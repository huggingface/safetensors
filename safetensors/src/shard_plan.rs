//! Tensor Parallelism plan for safetensors.

use core::{fmt::Display, str::FromStr};

use crate::lib::HashMap;

use crate::{tensor::TensorInfo, Dtype};

/// Errors that can occur during shard plan parsing and resolution
#[derive(Debug)]
pub enum ShardPlanError {
    /// Invalid shard strategy string
    InvalidStrategy(String),
    /// Sharding is not supported for unaligned dtypes
    UnalignedDtype(Dtype),
    /// Requested tensor for shard slice computation is not present in shard plan
    TensorNotInPlan(String),
}

impl Display for ShardPlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShardPlanError::InvalidStrategy(s) => write!(f, "Invalid shard strategy: {}", s),
            ShardPlanError::UnalignedDtype(dtype) => {
                write!(f, "Sharding not supported for unaligned dtype: {:?}", dtype)
            }
            ShardPlanError::TensorNotInPlan(name) => {
                write!(f, "Requested tensor '{}' is not in shard_plan", name)
            }
        }
    }
}

/// Sharding strategies for tensor parallelism
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardStrategy {
    /// Shard dim -2 for 2D+ tensors or dim -1 for 1D tensors
    /// Produces contiguous byte ranges in row-major layout
    Colwise,
    /// Shard dim -2 for 1D tensors or replicate for 1D tensors
    /// Non-contiguous bytes requires full tensor load followed by narrowing on device
    Rowwise,
    /// Full copy to all ranks, no sharding
    Replicate,
}

impl FromStr for ShardStrategy {
    type Err = ShardPlanError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "colwise" => Ok(ShardStrategy::Colwise),
            "rowwise" => Ok(ShardStrategy::Rowwise),
            "replicate" => Ok(ShardStrategy::Replicate),
            _ => Err(ShardPlanError::InvalidStrategy(s.to_string())),
        }
    }
}

fn name_to_pattern(name: &str) -> String {
    let mut result = String::with_capacity(name.len());
    for (i, seg) in name.split('.').enumerate() {
        if i > 0 {
            result.push('.');
        }
        if !seg.is_empty() && seg.bytes().all(|b| b.is_ascii_digit()) {
            result.push('*');
        } else {
            result.push_str(seg);
        }
    }
    result
}

/// A pattern entry with shard strategy and optional rank filter
#[derive(Debug, Clone)]
pub struct ShardPatternConfig {
    /// The sharding strategy for this pattern
    pub strategy: ShardStrategy,
    /// Optional rank filter. None means all ranks, Some(vec) means only ranks in the vec.
    pub ranks: Option<Vec<usize>>,
}

impl ShardPatternConfig {
    /// Create a new shard pattern config with the given strategy and optional rank filter
    pub fn new(strategy: ShardStrategy, ranks: Option<Vec<usize>>) -> Self {
        Self { strategy, ranks }
    }

    /// Check if the given rank is included in this pattern's rank filter
    pub fn includes_rank(&self, rank: usize) -> bool {
        match &self.ranks {
            None => true,
            Some(ranks) => ranks.contains(&rank),
        }
    }
}

/// Represents the byte range and shape of a shard for a given tensor, computed from the sharding
/// strategy
#[derive(Debug, Clone)]
pub enum ShardSlice {
    /// Contiguous byte range that can be loaded directly
    Contiguous {
        /// Shape of the shard tensor
        shape: Vec<usize>,
        /// Start byte offset of the shard in the file
        start: usize,
        /// End byte offset of the shard in the file
        end: usize,
    },
    /// Non-contiguous byte range that requires loading the full tensor and then narrowing on
    /// device
    NarrowAfterLoad {
        /// Shape of the narrowed tensor
        shape: Vec<usize>,
        /// Dimension to narrow on after loading the full tensor
        dim: usize,
        /// Start index of the narrow slice along the narrow_dim
        start: usize,
        /// Length of the narrow slice along the narrow_dim
        len: usize,
    },
    /// No sharding, full copy of the tensor to all ranks
    FullCopy {
        /// Output shape (same as original tensor shape)
        shape: Vec<usize>,
    },
    /// Tensor is not loaded for the current rank
    Skip,
}

impl ShardSlice {
    /// Returns the output shape after applying this slice.
    /// Returns None for Skip (no tensor produced).
    pub fn shape(&self) -> Option<&[usize]> {
        match self {
            Self::Contiguous { shape, .. } => Some(shape),
            Self::NarrowAfterLoad { shape, .. } => Some(shape),
            Self::FullCopy { shape } => Some(shape),
            _ => None,
        }
    }
}

/// Represents a sharding plan for distributing tensors across multiple ranks
#[derive(Debug)]
pub struct ShardPlan {
    patterns: HashMap<String, ShardPatternConfig>,
    world_size: usize,
}

impl ShardPlan {
    /// Create a new shard plan from a given mapping of patterns
    pub fn new(patterns: HashMap<String, ShardPatternConfig>, world_size: usize) -> Self {
        Self {
            patterns,
            world_size,
        }
    }

    /// Resolve the sharding configuration for the given tensor name
    /// Matches the tensor name against the patterns in the plan, returning the config of the
    /// longest matching pattern. Patterns can include '*' as a wildcard for numeric segments,
    /// allowing for flexible matching of tensor names with varying numeric indices.
    pub fn resolve(&self, tensor_name: &str) -> Option<ShardPatternConfig> {
        let pattern_to_match = name_to_pattern(tensor_name);
        self.patterns
            .iter()
            .filter(|(pattern, _)| pattern_to_match.starts_with(pattern.as_str()))
            .max_by_key(|(pattern, _)| pattern.len())
            .map(|(_, config)| config.clone())
    }

    /// Compute a shard slice for the given tensor and rank
    pub fn compute_slice(
        &self,
        tensor_name: &str,
        info: &TensorInfo,
        rank: usize,
    ) -> Result<ShardSlice, ShardPlanError> {
        if let Dtype::F4 | Dtype::F6_E2M3 | Dtype::F6_E3M2 = info.dtype {
            return Err(ShardPlanError::UnalignedDtype(info.dtype));
        }

        let config = self
            .resolve(tensor_name)
            .ok_or_else(|| ShardPlanError::TensorNotInPlan(tensor_name.to_owned()))?;

        if !config.includes_rank(rank) {
            return Ok(ShardSlice::Skip);
        }

        match (info.shape.len(), config.strategy) {
            (1, ShardStrategy::Colwise) => {
                let dim_size = info.shape[0];
                let elem_bytes = info.dtype.bitsize() / 8;

                let (start_idx, end_idx) = shard_range(dim_size, rank, self.world_size);
                let len = end_idx - start_idx;
                let shape = vec![len];

                let byte_start = start_idx * elem_bytes;
                let byte_end = end_idx * elem_bytes;

                Ok(ShardSlice::Contiguous {
                    shape,
                    start: byte_start,
                    end: byte_end,
                })
            }
            (0, _) | (1, ShardStrategy::Rowwise) | (_, ShardStrategy::Replicate) => {
                Ok(ShardSlice::FullCopy {
                    shape: info.shape.clone(),
                })
            }
            (_, ShardStrategy::Colwise) => {
                let shard_dim = info.shape.len() - 2;
                let dim_size = info.shape[shard_dim];
                let elem_bytes = info.dtype.bitsize() / 8;
                // this will always be the last dimension's length
                let row_bytes = info.shape[shard_dim + 1] * elem_bytes;

                let (start_idx, end_idx) = shard_range(dim_size, rank, self.world_size);
                let mut shape = info.shape.clone();
                shape[shard_dim] = end_idx - start_idx;

                let byte_start = start_idx * row_bytes;
                let byte_end = end_idx * row_bytes;

                Ok(ShardSlice::Contiguous {
                    shape,
                    start: byte_start,
                    end: byte_end,
                })
            }
            (_, ShardStrategy::Rowwise) => {
                let shard_dim = info.shape.len() - 1;
                let dim_size = info.shape[shard_dim];

                let (start_idx, end_idx) = shard_range(dim_size, rank, self.world_size);

                let len = end_idx - start_idx;
                let mut shape = info.shape.clone();
                shape[shard_dim] = len;

                Ok(ShardSlice::NarrowAfterLoad {
                    shape,
                    dim: shard_dim,
                    start: start_idx,
                    len,
                })
            }
        }
    }
}

fn shard_range(dim_size: usize, rank: usize, world_size: usize) -> (usize, usize) {
    let shard_size = dim_size.div_ceil(world_size);
    let start = rank * shard_size;
    let end = start.saturating_add(shard_size).min(dim_size);
    (start, end)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ti(dtype: Dtype, shape: Vec<usize>) -> TensorInfo {
        let elem_bytes = dtype.bitsize() / 8;
        let n_bytes: usize = shape.iter().product::<usize>() * elem_bytes;
        TensorInfo {
            dtype,
            shape,
            data_offsets: (0, n_bytes),
        }
    }

    #[test]
    fn pattern_replaces_digit_segments() {
        assert_eq!(
            name_to_pattern("layers.0.self_attn.q_proj.weight"),
            "layers.*.self_attn.q_proj.weight"
        );
    }

    #[test]
    fn pattern_replaces_multi_digit() {
        assert_eq!(
            name_to_pattern("layers.31.mlp.down_proj.bias"),
            "layers.*.mlp.down_proj.bias"
        );
    }

    #[test]
    fn pattern_preserves_mixed_segments() {
        assert_eq!(name_to_pattern("model.w1.weight"), "model.w1.weight");
    }

    #[test]
    fn pattern_no_digits() {
        assert_eq!(name_to_pattern("lm_head.weight"), "lm_head.weight");
    }

    #[test]
    fn pattern_hyphenated_suffix() {
        assert_eq!(
            name_to_pattern("layers.0.self_attn.q_proj-bias"),
            "layers.*.self_attn.q_proj-bias"
        );
    }

    #[test]
    fn pattern_multiple_numeric_segments() {
        assert_eq!(
            name_to_pattern("model.layers.0.experts.3.gate_proj.weight"),
            "model.layers.*.experts.*.gate_proj.weight"
        );
    }

    #[test]
    fn parse_colwise() {
        assert_eq!(
            "colwise".parse::<ShardStrategy>().unwrap(),
            ShardStrategy::Colwise
        );
    }

    #[test]
    fn parse_rowwise() {
        assert_eq!(
            "rowwise".parse::<ShardStrategy>().unwrap(),
            ShardStrategy::Rowwise
        );
    }

    #[test]
    fn parse_replicate() {
        assert_eq!(
            "replicate".parse::<ShardStrategy>().unwrap(),
            ShardStrategy::Replicate
        );
    }

    #[test]
    fn parse_invalid() {
        assert!("colwise_rep".parse::<ShardStrategy>().is_err());
    }

    #[test]
    fn range_even_split() {
        assert_eq!(shard_range(4096, 0, 4), (0, 1024));
        assert_eq!(shard_range(4096, 1, 4), (1024, 2048));
        assert_eq!(shard_range(4096, 2, 4), (2048, 3072));
        assert_eq!(shard_range(4096, 3, 4), (3072, 4096));
    }

    #[test]
    fn range_uneven_split() {
        // ceil(10/3) = 4
        assert_eq!(shard_range(10, 0, 3), (0, 4));
        assert_eq!(shard_range(10, 1, 3), (4, 8));
        assert_eq!(shard_range(10, 2, 3), (8, 10));
    }

    #[test]
    fn resolve_dot_weight_suffix() {
        let plan = ShardPlan::new(
            HashMap::from([(
                "layers.*.self_attn.q_proj".into(),
                ShardPatternConfig::new(ShardStrategy::Colwise, None),
            )]),
            4,
        );
        assert_eq!(
            plan.resolve("layers.0.self_attn.q_proj.weight")
                .map(|c| c.strategy),
            Some(ShardStrategy::Colwise),
        );
    }

    #[test]
    fn resolve_dot_bias_suffix() {
        let plan = ShardPlan::new(
            HashMap::from([(
                "layers.*.self_attn.q_proj".into(),
                ShardPatternConfig::new(ShardStrategy::Colwise, None),
            )]),
            4,
        );
        assert_eq!(
            plan.resolve("layers.31.self_attn.q_proj.bias")
                .map(|c| c.strategy),
            Some(ShardStrategy::Colwise),
        );
    }

    #[test]
    fn resolve_hyphen_suffix() {
        let plan = ShardPlan::new(
            HashMap::from([(
                "layers.*.self_attn.q_proj".into(),
                ShardPatternConfig::new(ShardStrategy::Colwise, None),
            )]),
            4,
        );
        assert_eq!(
            plan.resolve("layers.0.self_attn.q_proj-bias")
                .map(|c| c.strategy),
            Some(ShardStrategy::Colwise),
        );
    }

    #[test]
    fn resolve_longest_prefix_wins() {
        let plan = ShardPlan::new(
            HashMap::from([
                (
                    "layers.*".into(),
                    ShardPatternConfig::new(ShardStrategy::Replicate, None),
                ),
                (
                    "layers.*.self_attn.q_proj".into(),
                    ShardPatternConfig::new(ShardStrategy::Colwise, None),
                ),
            ]),
            4,
        );
        assert_eq!(
            plan.resolve("layers.0.self_attn.q_proj.weight")
                .map(|c| c.strategy),
            Some(ShardStrategy::Colwise),
        );
    }

    #[test]
    fn resolve_no_match() {
        let plan = ShardPlan::new(
            HashMap::from([(
                "layers.*.self_attn.q_proj".into(),
                ShardPatternConfig::new(ShardStrategy::Colwise, None),
            )]),
            4,
        );
        assert!(plan.resolve("model.embed_tokens.weight").is_none());
    }

    #[test]
    fn resolve_multiple_strategies() {
        let plan = ShardPlan::new(
            HashMap::from([
                (
                    "layers.*.self_attn.q_proj".into(),
                    ShardPatternConfig::new(ShardStrategy::Colwise, None),
                ),
                (
                    "layers.*.self_attn.o_proj".into(),
                    ShardPatternConfig::new(ShardStrategy::Rowwise, None),
                ),
                (
                    "layers.*.layer_norm".into(),
                    ShardPatternConfig::new(ShardStrategy::Replicate, None),
                ),
            ]),
            4,
        );
        assert_eq!(
            plan.resolve("layers.0.self_attn.q_proj.weight")
                .map(|c| c.strategy),
            Some(ShardStrategy::Colwise),
        );
        assert_eq!(
            plan.resolve("layers.0.self_attn.o_proj.weight")
                .map(|c| c.strategy),
            Some(ShardStrategy::Rowwise),
        );
        assert_eq!(
            plan.resolve("layers.0.layer_norm.weight")
                .map(|c| c.strategy),
            Some(ShardStrategy::Replicate),
        );
    }

    #[test]
    fn colwise_2d_rank0() {
        let plan = ShardPlan::new(
            HashMap::from([(
                "layers.*.self_attn.q_proj".into(),
                ShardPatternConfig::new(ShardStrategy::Colwise, None),
            )]),
            4,
        );
        let info = ti(Dtype::F16, vec![4096, 4096]);

        match plan
            .compute_slice("layers.0.self_attn.q_proj.weight", &info, 0)
            .unwrap()
        {
            ShardSlice::Contiguous { shape, start, end } => {
                assert_eq!(shape, vec![1024, 4096]);
                assert_eq!(start, 0);
                assert_eq!(end, 1024 * 4096 * 2);
            }
            other => panic!(
                "Expected Contiguous, got {:?}",
                std::mem::discriminant(&other)
            ),
        }
    }

    #[test]
    fn colwise_2d_rank3() {
        let plan = ShardPlan::new(
            HashMap::from([(
                "layers.*.self_attn.q_proj".into(),
                ShardPatternConfig::new(ShardStrategy::Colwise, None),
            )]),
            4,
        );
        let info = ti(Dtype::F16, vec![4096, 4096]);

        match plan
            .compute_slice("layers.0.self_attn.q_proj.weight", &info, 3)
            .unwrap()
        {
            ShardSlice::Contiguous { shape, start, end } => {
                assert_eq!(shape, vec![1024, 4096]);
                assert_eq!(start, 3072 * 4096 * 2);
                assert_eq!(end, 4096 * 4096 * 2);
            }
            other => panic!(
                "Expected Contiguous, got {:?}",
                std::mem::discriminant(&other)
            ),
        }
    }

    #[test]
    fn colwise_2d_uneven_last_rank() {
        let plan = ShardPlan::new(
            HashMap::from([(
                "proj".into(),
                ShardPatternConfig::new(ShardStrategy::Colwise, None),
            )]),
            3,
        );
        // [10, 4] F32: ceil(10/3)=4, rank 2 → rows [8,10)
        let info = ti(Dtype::F32, vec![10, 4]);

        match plan.compute_slice("proj.weight", &info, 2).unwrap() {
            ShardSlice::Contiguous { shape, start, end } => {
                assert_eq!(shape, vec![2, 4]);
                assert_eq!(start, 8 * 4 * 4);
                assert_eq!(end, 10 * 4 * 4);
            }
            other => panic!(
                "Expected Contiguous, got {:?}",
                std::mem::discriminant(&other)
            ),
        }
    }

    #[test]
    fn colwise_2d_both_ranks_cover_full_tensor() {
        let plan = ShardPlan::new(
            HashMap::from([(
                "proj".into(),
                ShardPatternConfig::new(ShardStrategy::Colwise, None),
            )]),
            2,
        );
        let info = ti(Dtype::F32, vec![8, 4]);
        let row_bytes = 4 * 4;

        match (
            plan.compute_slice("proj.weight", &info, 0).unwrap(),
            plan.compute_slice("proj.weight", &info, 1).unwrap(),
        ) {
            (
                ShardSlice::Contiguous {
                    shape: s0,
                    start: st0,
                    end: e0,
                },
                ShardSlice::Contiguous {
                    shape: s1,
                    start: st1,
                    end: e1,
                },
            ) => {
                assert_eq!(s0, vec![4, 4]);
                assert_eq!(st0, 0);
                assert_eq!(e0, 4 * row_bytes);
                assert_eq!(s1, vec![4, 4]);
                assert_eq!(st1, e0); // contiguous
                assert_eq!(e1, 8 * row_bytes);
            }
            _ => panic!("Expected both Contiguous"),
        }
    }

    #[test]
    fn colwise_all_ranks_cover_full_dimension() {
        let plan = ShardPlan::new(
            HashMap::from([(
                "proj".into(),
                ShardPatternConfig::new(ShardStrategy::Colwise, None),
            )]),
            4,
        );
        let info = ti(Dtype::F32, vec![100, 64]);
        let row_bytes = 64 * 4;

        let mut total_rows = 0;
        let mut prev_end = 0;
        for rank in 0..4 {
            match plan.compute_slice("proj.weight", &info, rank).unwrap() {
                ShardSlice::Contiguous { shape, start, end } => {
                    assert_eq!(shape[1], 64);
                    assert_eq!(start, prev_end);
                    assert_eq!(end - start, shape[0] * row_bytes);
                    total_rows += shape[0];
                    prev_end = end;
                }
                _ => panic!("Expected Contiguous for rank {rank}"),
            }
        }
        assert_eq!(total_rows, 100);
        assert_eq!(prev_end, 100 * row_bytes);
    }

    #[test]
    fn colwise_1d_rank0() {
        let plan = ShardPlan::new(
            HashMap::from([(
                "layers.*.self_attn.q_proj".into(),
                ShardPatternConfig::new(ShardStrategy::Colwise, None),
            )]),
            4,
        );
        let info = ti(Dtype::F16, vec![4096]);

        match plan
            .compute_slice("layers.0.self_attn.q_proj.bias", &info, 0)
            .unwrap()
        {
            ShardSlice::Contiguous { shape, start, end } => {
                assert_eq!(shape, vec![1024]);
                assert_eq!(start, 0);
                assert_eq!(end, 1024 * 2);
            }
            other => panic!(
                "Expected Contiguous, got {:?}",
                std::mem::discriminant(&other)
            ),
        }
    }

    #[test]
    fn colwise_1d_last_rank() {
        let plan = ShardPlan::new(
            HashMap::from([(
                "layers.*.self_attn.q_proj".into(),
                ShardPatternConfig::new(ShardStrategy::Colwise, None),
            )]),
            4,
        );
        let info = ti(Dtype::F16, vec![4096]);

        match plan
            .compute_slice("layers.0.self_attn.q_proj.bias", &info, 3)
            .unwrap()
        {
            ShardSlice::Contiguous { shape, start, end } => {
                assert_eq!(shape, vec![1024]);
                assert_eq!(start, 3072 * 2);
                assert_eq!(end, 4096 * 2);
            }
            other => panic!(
                "Expected Contiguous, got {:?}",
                std::mem::discriminant(&other)
            ),
        }
    }

    #[test]
    fn colwise_3d_shards_dim_minus_2() {
        let plan = ShardPlan::new(
            HashMap::from([(
                "proj".into(),
                ShardPatternConfig::new(ShardStrategy::Colwise, None),
            )]),
            2,
        );
        // [2, 8, 4] F32: dim -2 = dim 1, row_bytes = 4 * 4 = 16
        let info = ti(Dtype::F32, vec![2, 8, 4]);

        match plan.compute_slice("proj.weight", &info, 0).unwrap() {
            ShardSlice::Contiguous { shape, start, end } => {
                assert_eq!(shape, vec![2, 4, 4]);
                assert_eq!(start, 0);
                assert_eq!(end, 4 * 4 * 4); // 4 rows * 4 cols * 4 bytes
            }
            other => panic!(
                "Expected Contiguous, got {:?}",
                std::mem::discriminant(&other)
            ),
        }
    }

    #[test]
    fn rowwise_2d_rank0() {
        let plan = ShardPlan::new(
            HashMap::from([(
                "layers.*.mlp.down_proj".into(),
                ShardPatternConfig::new(ShardStrategy::Rowwise, None),
            )]),
            2,
        );
        let info = ti(Dtype::F16, vec![4096, 11008]);

        match plan
            .compute_slice("layers.0.mlp.down_proj.weight", &info, 0)
            .unwrap()
        {
            ShardSlice::NarrowAfterLoad {
                shape,
                dim,
                start,
                len,
            } => {
                assert_eq!(shape, vec![4096, 5504]);
                assert_eq!(dim, 1);
                assert_eq!(start, 0);
                assert_eq!(len, 5504);
            }
            other => panic!(
                "Expected NarrowAfterLoad, got {:?}",
                std::mem::discriminant(&other)
            ),
        }
    }

    #[test]
    fn rowwise_2d_rank1() {
        let plan = ShardPlan::new(
            HashMap::from([(
                "layers.*.mlp.down_proj".into(),
                ShardPatternConfig::new(ShardStrategy::Rowwise, None),
            )]),
            2,
        );
        let info = ti(Dtype::F16, vec![4096, 11008]);

        match plan
            .compute_slice("layers.0.mlp.down_proj.weight", &info, 1)
            .unwrap()
        {
            ShardSlice::NarrowAfterLoad {
                shape,
                dim,
                start,
                len,
            } => {
                assert_eq!(shape, vec![4096, 5504]);
                assert_eq!(dim, 1);
                assert_eq!(start, 5504);
                assert_eq!(len, 5504);
            }
            other => panic!(
                "Expected NarrowAfterLoad, got {:?}",
                std::mem::discriminant(&other)
            ),
        }
    }

    #[test]
    fn rowwise_2d_uneven() {
        let plan = ShardPlan::new(
            HashMap::from([(
                "proj".into(),
                ShardPatternConfig::new(ShardStrategy::Rowwise, None),
            )]),
            3,
        );
        // [4, 10] F32: ceil(10/3)=4, rank 2 → cols [8,10)
        let info = ti(Dtype::F32, vec![4, 10]);

        match plan.compute_slice("proj.weight", &info, 2).unwrap() {
            ShardSlice::NarrowAfterLoad {
                shape,
                dim,
                start,
                len,
            } => {
                assert_eq!(shape, vec![4, 2]);
                assert_eq!(dim, 1);
                assert_eq!(start, 8);
                assert_eq!(len, 2);
            }
            other => panic!(
                "Expected NarrowAfterLoad, got {:?}",
                std::mem::discriminant(&other)
            ),
        }
    }

    #[test]
    fn rowwise_all_ranks_cover_full_dimension() {
        let plan = ShardPlan::new(
            HashMap::from([(
                "proj".into(),
                ShardPatternConfig::new(ShardStrategy::Rowwise, None),
            )]),
            4,
        );
        let info = ti(Dtype::F32, vec![64, 100]);

        let mut total_cols = 0;
        let mut prev_end = 0;
        for rank in 0..4 {
            match plan.compute_slice("proj.weight", &info, rank).unwrap() {
                ShardSlice::NarrowAfterLoad {
                    shape,
                    dim,
                    start,
                    len,
                } => {
                    assert_eq!(dim, 1);
                    assert_eq!(shape[0], 64);
                    assert_eq!(start, prev_end);
                    assert_eq!(len, shape[1]);
                    total_cols += len;
                    prev_end = start + len;
                }
                _ => panic!("Expected NarrowAfterLoad for rank {rank}"),
            }
        }
        assert_eq!(total_cols, 100);
    }

    #[test]
    fn rowwise_1d_replicated() {
        let plan = ShardPlan::new(
            HashMap::from([(
                "layers.*.mlp.down_proj".into(),
                ShardPatternConfig::new(ShardStrategy::Rowwise, None),
            )]),
            4,
        );
        let info = ti(Dtype::F16, vec![4096]);

        let result = plan
            .compute_slice("layers.0.mlp.down_proj.bias", &info, 0)
            .unwrap();
        assert!(matches!(result, ShardSlice::FullCopy { .. }));
    }

    #[test]
    fn replicate_2d() {
        let plan = ShardPlan::new(
            HashMap::from([(
                "layers.*.layer_norm".into(),
                ShardPatternConfig::new(ShardStrategy::Replicate, None),
            )]),
            4,
        );
        let info = ti(Dtype::F16, vec![4096, 4096]);

        let result = plan
            .compute_slice("layers.0.layer_norm.weight", &info, 0)
            .unwrap();
        assert!(matches!(result, ShardSlice::FullCopy { .. }));
    }

    #[test]
    fn replicate_1d() {
        let plan = ShardPlan::new(
            HashMap::from([(
                "layers.*.layer_norm".into(),
                ShardPatternConfig::new(ShardStrategy::Replicate, None),
            )]),
            4,
        );
        let info = ti(Dtype::F16, vec![4096]);

        let result = plan
            .compute_slice("layers.0.layer_norm.weight", &info, 0)
            .unwrap();
        assert!(matches!(result, ShardSlice::FullCopy { .. }));
    }

    #[test]
    fn no_match_errors() {
        let plan = ShardPlan::new(
            HashMap::from([(
                "layers.*.self_attn.q_proj".into(),
                ShardPatternConfig::new(ShardStrategy::Colwise, None),
            )]),
            4,
        );
        let info = ti(Dtype::F16, vec![32000, 4096]);

        let result = plan.compute_slice("lm_head.weight", &info, 0);
        assert!(matches!(result, Err(ShardPlanError::TensorNotInPlan(_))));
    }

    #[test]
    fn scalar_full_copy() {
        let plan = ShardPlan::new(
            HashMap::from([(
                "layers.*.norm".into(),
                ShardPatternConfig::new(ShardStrategy::Colwise, None),
            )]),
            4,
        );
        let info = TensorInfo {
            dtype: Dtype::F32,
            shape: vec![],
            data_offsets: (0, 4),
        };

        let result = plan
            .compute_slice("layers.0.norm.weight", &info, 0)
            .unwrap();
        assert!(matches!(result, ShardSlice::FullCopy { .. }));
    }

    #[test]
    fn f4_errors() {
        let plan = ShardPlan::new(
            HashMap::from([(
                "proj".into(),
                ShardPatternConfig::new(ShardStrategy::Colwise, None),
            )]),
            2,
        );
        let info = TensorInfo {
            dtype: Dtype::F4,
            shape: vec![8, 4],
            data_offsets: (0, 16),
        };
        assert!(plan.compute_slice("proj.weight", &info, 0).is_err());
    }

    #[test]
    fn f6_errors() {
        let plan = ShardPlan::new(
            HashMap::from([(
                "proj".into(),
                ShardPatternConfig::new(ShardStrategy::Colwise, None),
            )]),
            2,
        );
        let info = TensorInfo {
            dtype: Dtype::F6_E2M3,
            shape: vec![8, 4],
            data_offsets: (0, 24),
        };
        assert!(plan.compute_slice("proj.weight", &info, 0).is_err());
    }

    #[test]
    fn colwise_bf16() {
        let plan = ShardPlan::new(
            HashMap::from([(
                "proj".into(),
                ShardPatternConfig::new(ShardStrategy::Colwise, None),
            )]),
            2,
        );
        let info = ti(Dtype::BF16, vec![8, 4]);

        match plan.compute_slice("proj.weight", &info, 0).unwrap() {
            ShardSlice::Contiguous { shape, start, end } => {
                assert_eq!(shape, vec![4, 4]);
                assert_eq!(start, 0);
                assert_eq!(end, 4 * 4 * 2);
            }
            other => panic!(
                "Expected Contiguous, got {:?}",
                std::mem::discriminant(&other)
            ),
        }
    }

    #[test]
    fn colwise_f8() {
        let plan = ShardPlan::new(
            HashMap::from([(
                "proj".into(),
                ShardPatternConfig::new(ShardStrategy::Colwise, None),
            )]),
            2,
        );
        let info = ti(Dtype::F8_E4M3, vec![8, 4]);

        match plan.compute_slice("proj.weight", &info, 0).unwrap() {
            ShardSlice::Contiguous { shape, start, end } => {
                assert_eq!(shape, vec![4, 4]);
                assert_eq!(start, 0);
                assert_eq!(end, 4 * 4);
            }
            other => panic!(
                "Expected Contiguous, got {:?}",
                std::mem::discriminant(&other)
            ),
        }
    }

    #[test]
    fn rank_filter_skip() {
        let plan = ShardPlan::new(
            HashMap::from([(
                "experts.*.gate".into(),
                ShardPatternConfig::new(ShardStrategy::Colwise, Some(vec![0, 2])),
            )]),
            4,
        );
        let info = ti(Dtype::F16, vec![4096, 4096]);

        assert!(matches!(
            plan.compute_slice("experts.0.gate.weight", &info, 0)
                .unwrap(),
            ShardSlice::Contiguous { .. }
        ));

        assert!(matches!(
            plan.compute_slice("experts.0.gate.weight", &info, 1)
                .unwrap(),
            ShardSlice::Skip
        ));

        assert!(matches!(
            plan.compute_slice("experts.0.gate.weight", &info, 2)
                .unwrap(),
            ShardSlice::Contiguous { .. }
        ));

        assert!(matches!(
            plan.compute_slice("experts.0.gate.weight", &info, 3)
                .unwrap(),
            ShardSlice::Skip
        ));
    }

    #[test]
    fn includes_rank_none_matches_all() {
        let cfg = ShardPatternConfig::new(ShardStrategy::Colwise, None);
        assert!(cfg.includes_rank(0));
        assert!(cfg.includes_rank(100));
    }

    #[test]
    fn includes_rank_some_filters() {
        let cfg = ShardPatternConfig::new(ShardStrategy::Colwise, Some(vec![1, 3]));
        assert!(!cfg.includes_rank(0));
        assert!(cfg.includes_rank(1));
        assert!(!cfg.includes_rank(2));
        assert!(cfg.includes_rank(3));
    }
}
