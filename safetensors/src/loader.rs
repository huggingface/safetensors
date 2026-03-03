//! TODO:

use core::{error::Error, fmt::Display};
use std::path::Path;

use hashbrown::HashMap;

use crate::{
    index::IndexParsingError,
    shard_plan::{ShardPlan, ShardPlanError, ShardSlice},
    tensor::{TensorInfo, N_LEN},
    SafeTensorError, SafeTensors,
};

/// Tensor loading errors
#[derive(Debug)]
pub enum LoaderError {
    /// hmll error
    Hmll(hmll::Error),
    /// Safetensors error
    SafeTensors(SafeTensorError),
    /// Duplicate tensor name found in source files
    DuplicateTensor(String),
    /// error parsing .index.json error
    IndexParsing(IndexParsingError),
    /// Queried tensor does not exist in the tensor_index
    TensorNotFound(String),
    /// Sharding error
    ShardPlan(ShardPlanError),
}

impl Display for LoaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoaderError::Hmll(err) => write!(f, "hmll error: {}", err),
            LoaderError::SafeTensors(err) => write!(f, "safetensors error: {}", err),
            LoaderError::DuplicateTensor(name) => write!(f, "duplicate tensor '{}' found", name),
            LoaderError::IndexParsing(err) => write!(f, ".index.json parsing err: {}", err),
            LoaderError::TensorNotFound(name) => {
                write!(
                    f,
                    "Tensor '{}' was not found in parsed header metadata",
                    name
                )
            }
            LoaderError::ShardPlan(err) => write!(f, "sharding error: {}", err),
        }
    }
}

impl Error for LoaderError {}

impl From<hmll::Error> for LoaderError {
    fn from(err: hmll::Error) -> Self {
        LoaderError::Hmll(err)
    }
}

impl From<SafeTensorError> for LoaderError {
    fn from(err: SafeTensorError) -> Self {
        LoaderError::SafeTensors(err)
    }
}

impl From<IndexParsingError> for LoaderError {
    fn from(err: IndexParsingError) -> Self {
        LoaderError::IndexParsing(err)
    }
}

impl From<ShardPlanError> for LoaderError {
    fn from(err: ShardPlanError) -> Self {
        LoaderError::ShardPlan(err)
    }
}

/// TODO:
#[derive(Debug)]
pub struct ShardContext {
    plan: ShardPlan,
    rank: usize,
}

/// TODO:
#[derive(Debug)]
pub struct TensorLoader {
    loader: hmll::WeightLoader,
    source_offsets: Vec<usize>,
    tensor_index: HashMap<String, (usize, TensorInfo)>,
    shard_ctx: Option<ShardContext>,
}

/// Result of fetching a tensor, including optional sharding info
#[derive(Debug)]
pub struct FetchResult {
    /// The tensor data buffer
    pub buffer: hmll::Buffer,
    /// Tensor metadata (dtype, shape, offsets)
    pub info: TensorInfo,
    /// If sharding was applied, the slice info for post-processing
    pub shard_slice: Option<ShardSlice>,
}

impl TensorLoader {
    /// Create a tensor loader from a list of source files
    pub fn open(paths: &[impl AsRef<Path>], device: hmll::Device) -> Result<Self, LoaderError> {
        let sources = paths
            .iter()
            .map(hmll::Source::open)
            .collect::<hmll::Result<Vec<_>>>()?;

        // XXX: we only want to use hmll for IoUring backend, so we hard code it for the moment
        let loader = hmll::WeightLoader::new(sources, device, hmll::LoaderKind::IoUring)?;

        let (source_offsets, tensor_index) = paths.iter().enumerate().try_fold(
            (Vec::with_capacity(paths.len()), HashMap::new()),
            |(mut offsets, mut tensor_index), (source_idx, path)| {
                let (header_size, metadata) = SafeTensors::read_metadata_from_file(path)?;
                offsets.push(N_LEN + header_size);

                for (name, info) in metadata.tensors() {
                    if tensor_index
                        .insert(name.clone(), (source_idx, info.clone()))
                        .is_some()
                    {
                        return Err(LoaderError::DuplicateTensor(name));
                    }
                }
                Ok((offsets, tensor_index))
            },
        )?;

        Ok(Self {
            loader,
            source_offsets,
            tensor_index,
            shard_ctx: None,
        })
    }

    /// Create a tensor loader from an .index.json file
    pub fn open_index(path: impl AsRef<Path>, device: hmll::Device) -> Result<Self, LoaderError> {
        use crate::index::parse_index;

        let parent = path.as_ref().parent().unwrap_or_else(|| Path::new("."));

        let index = parse_index(&path)?;
        let paths = index.unique_source_files(Some(parent));

        Self::open(&paths, device)
    }

    /// Configure sharding for tensor parallel loading
    pub fn with_shard_plan(mut self, plan: ShardPlan, rank: usize) -> Self {
        self.shard_ctx = Some(ShardContext { plan, rank });

        self
    }

    /// Get the tensor's data from it's source file
    pub fn fetch_tensor(&mut self, name: &str) -> Result<FetchResult, LoaderError> {
        let (source_idx, info) = self
            .tensor_index
            .get(name)
            .ok_or_else(|| LoaderError::TensorNotFound(name.to_owned()))?;

        let source_offset = self.source_offsets[*source_idx];
        let (tensor_start, tensor_end) = info.data_offsets;
        let (fetch_start, fetch_end, shard_slice) = match &self.shard_ctx {
            Some(ctx) => {
                let slice = ctx.plan.compute_slice(name, info, ctx.rank)?;
                match slice {
                    ShardSlice::Contiguous { start, end, .. } => {
                        let abs_start = source_offset + tensor_start + start;
                        let abs_end = source_offset + tensor_end + end;
                        (abs_start, abs_end, Some(slice))
                    }
                    ShardSlice::NarrowAfterLoad { .. } | ShardSlice::FullCopy => {
                        let abs_start = source_offset + tensor_start;
                        let abs_end = source_offset + tensor_end;
                        (abs_start, abs_end, Some(slice))
                    }
                }
            }
            None => {
                let abs_start = source_offset + tensor_start;
                let abs_end = source_offset + tensor_end;
                (abs_start, abs_end, None)
            }
        };

        let buffer = self.loader.fetch(fetch_start..fetch_end, *source_idx)?;

        Ok(FetchResult {
            buffer,
            info: info.clone(),
            shard_slice,
        })
    }

    /// Fetch tensor data in batch for performance
    pub fn fetch_tensors(
        &mut self,
        names: &[&str],
    ) -> Result<Vec<(String, FetchResult)>, LoaderError> {
        if names.is_empty() {
            return Ok(Vec::new());
        }

        struct FetchItem<'a> {
            name: &'a str,
            range: hmll::Range,
            info: &'a TensorInfo,
            shard_slice: Option<ShardSlice>,
        }

        let mut by_source: HashMap<usize, Vec<FetchItem>> = HashMap::new();

        for name in names {
            let (source_idx, info) = self
                .tensor_index
                .get(*name)
                .ok_or_else(|| LoaderError::TensorNotFound((*name).to_owned()))?;
            let source_offset = self.source_offsets[*source_idx];
            let (tensor_start, tensor_end) = info.data_offsets;

            let (fetch_start, fetch_end, shard_slice) = match &self.shard_ctx {
                Some(ctx) => {
                    let slice = ctx.plan.compute_slice(name, info, ctx.rank)?;
                    match slice {
                        ShardSlice::Contiguous { start, end, .. } => {
                            let abs_start = source_offset + tensor_start + start;
                            let abs_end = source_offset + tensor_end + end;
                            (abs_start, abs_end, Some(slice))
                        }
                        ShardSlice::NarrowAfterLoad { .. } | ShardSlice::FullCopy => {
                            let abs_start = source_offset + tensor_start;
                            let abs_end = source_offset + tensor_end;
                            (abs_start, abs_end, Some(slice))
                        }
                    }
                }
                None => {
                    let abs_start = source_offset + tensor_start;
                    let abs_end = source_offset + tensor_end;
                    (abs_start, abs_end, None)
                }
            };

            by_source.entry(*source_idx).or_default().push(FetchItem {
                name,
                info,
                range: (fetch_start..fetch_end).into(),
                shard_slice,
            });
        }

        let mut results = Vec::with_capacity(names.len());
        for (source_idx, items) in by_source {
            let ranges: Vec<hmll::Range> = items.iter().map(|item| item.range).collect();
            let buffers = self.loader.fetchv(&ranges, source_idx)?;

            for (item, buffer) in items.into_iter().zip(buffers) {
                results.push((
                    item.name.to_owned(),
                    FetchResult {
                        buffer,
                        info: item.info.clone(),
                        shard_slice: item.shard_slice,
                    },
                ));
            }
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{serialize_to_file, tensor::Dtype};
    use tempfile::{tempdir, NamedTempFile};

    fn create_test_safetensor(tensors: Vec<(&str, Dtype, Vec<usize>, Vec<u8>)>) -> NamedTempFile {
        let file = NamedTempFile::new().expect("Failed to create temp file");
        let data: Vec<(&str, crate::tensor::TensorView<'_>)> = tensors
            .iter()
            .map(|(name, dtype, shape, data)| {
                (
                    *name,
                    crate::tensor::TensorView::new(*dtype, shape.clone(), data).unwrap(),
                )
            })
            .collect();

        serialize_to_file(data, None, file.path()).expect("Failed to serialize");
        file
    }

    #[test]
    fn test_open_single_file() {
        let data = vec![0u8; 16]; // 4 f32s
        let file = create_test_safetensor(vec![("weight", Dtype::F32, vec![2, 2], data)]);

        let loader = TensorLoader::open(&[file.path()], hmll::Device::Cpu);
        assert!(loader.is_ok());

        let loader = loader.unwrap();
        assert_eq!(loader.source_offsets.len(), 1);
        assert_eq!(loader.tensor_index.len(), 1);
        assert!(loader.tensor_index.contains_key("weight"));
    }

    #[test]
    fn test_open_multiple_files() {
        let data1 = vec![0u8; 16];
        let data2 = vec![1u8; 32];

        let file1 = create_test_safetensor(vec![("tensor_a", Dtype::F32, vec![2, 2], data1)]);
        let file2 = create_test_safetensor(vec![("tensor_b", Dtype::F32, vec![2, 4], data2)]);

        let loader = TensorLoader::open(&[file1.path(), file2.path()], hmll::Device::Cpu);
        assert!(loader.is_ok());

        let loader = loader.unwrap();
        assert_eq!(loader.source_offsets.len(), 2);
        assert_eq!(loader.tensor_index.len(), 2);

        // Verify source indices
        let (idx_a, _) = loader.tensor_index.get("tensor_a").unwrap();
        let (idx_b, _) = loader.tensor_index.get("tensor_b").unwrap();
        assert_eq!(*idx_a, 0);
        assert_eq!(*idx_b, 1);
    }

    #[test]
    fn test_open_duplicate_tensor_error() {
        let data = vec![0u8; 16];

        let file1 =
            create_test_safetensor(vec![("shared_name", Dtype::F32, vec![2, 2], data.clone())]);
        let file2 = create_test_safetensor(vec![("shared_name", Dtype::F32, vec![2, 2], data)]);

        let result = TensorLoader::open(&[file1.path(), file2.path()], hmll::Device::Cpu);
        assert!(result.is_err());

        match result.unwrap_err() {
            LoaderError::DuplicateTensor(name) => assert_eq!(name, "shared_name"),
            e => panic!("Expected DuplicateTensor error, got {:?}", e),
        }
    }

    #[test]
    fn test_fetch_tensor() {
        let data: Vec<u8> = (0..16).collect(); // 4 f32s with distinct bytes
        let file = create_test_safetensor(vec![("weight", Dtype::F32, vec![2, 2], data.clone())]);

        let mut loader = TensorLoader::open(&[file.path()], hmll::Device::Cpu).unwrap();

        let result = loader.fetch_tensor("weight").unwrap();

        assert_eq!(result.info.dtype, Dtype::F32);
        assert_eq!(result.info.shape, vec![2, 2]);
        assert_eq!(result.buffer.len(), 16);
        assert_eq!(result.buffer.as_slice().unwrap(), data.as_slice());
        assert!(result.shard_slice.is_none());
    }

    #[test]
    fn test_fetch_tensor_not_found() {
        let data = vec![0u8; 16];
        let file = create_test_safetensor(vec![("weight", Dtype::F32, vec![2, 2], data)]);

        let mut loader = TensorLoader::open(&[file.path()], hmll::Device::Cpu).unwrap();

        let result = loader.fetch_tensor("nonexistent");
        assert!(result.is_err());

        match result.unwrap_err() {
            LoaderError::TensorNotFound(name) => assert_eq!(name, "nonexistent"),
            e => panic!("Expected TensorNotFound error, got {:?}", e),
        }
    }

    #[test]
    fn test_fetch_tensor_from_multiple_files() {
        let data1: Vec<u8> = (0..16).collect();
        let data2: Vec<u8> = (100..132).collect();

        let file1 =
            create_test_safetensor(vec![("tensor_a", Dtype::F32, vec![2, 2], data1.clone())]);
        let file2 =
            create_test_safetensor(vec![("tensor_b", Dtype::F32, vec![2, 4], data2.clone())]);

        let mut loader =
            TensorLoader::open(&[file1.path(), file2.path()], hmll::Device::Cpu).unwrap();

        let result_a = loader.fetch_tensor("tensor_a").unwrap();
        let result_b = loader.fetch_tensor("tensor_b").unwrap();

        assert_eq!(result_a.info.shape, vec![2, 2]);
        assert_eq!(result_b.info.shape, vec![2, 4]);
        assert_eq!(result_a.buffer.as_slice().unwrap(), data1.as_slice());
        assert_eq!(result_b.buffer.as_slice().unwrap(), data2.as_slice());
    }

    #[test]
    fn test_fetch_multiple_tensors_same_file() {
        let data1: Vec<u8> = (0..16).collect();
        let data2: Vec<u8> = (50..82).collect();

        let file = create_test_safetensor(vec![
            ("weight", Dtype::F32, vec![2, 2], data1.clone()),
            ("bias", Dtype::F32, vec![8], data2.clone()),
        ]);

        let mut loader = TensorLoader::open(&[file.path()], hmll::Device::Cpu).unwrap();

        assert_eq!(loader.tensor_index.len(), 2);

        let result_w = loader.fetch_tensor("weight").unwrap();
        let result_b = loader.fetch_tensor("bias").unwrap();

        assert_eq!(result_w.buffer.as_slice().unwrap(), data1.as_slice());
        assert_eq!(result_b.buffer.as_slice().unwrap(), data2.as_slice());
    }

    #[test]
    fn test_open_index() {
        let dir = tempdir().expect("Failed to create temp dir");

        // Create two safetensors files
        let data1: Vec<u8> = (0..16).collect();
        let data2: Vec<u8> = (100..132).collect();

        let path1 = dir.path().join("model-00001-of-00002.safetensors");
        let path2 = dir.path().join("model-00002-of-00002.safetensors");

        serialize_to_file(
            vec![(
                "layer1.weight",
                crate::tensor::TensorView::new(Dtype::F32, vec![2, 2], &data1).unwrap(),
            )],
            None,
            &path1,
        )
        .unwrap();

        serialize_to_file(
            vec![(
                "layer2.weight",
                crate::tensor::TensorView::new(Dtype::F32, vec![2, 4], &data2).unwrap(),
            )],
            None,
            &path2,
        )
        .unwrap();

        // Create index.json
        let index_path = dir.path().join("model.safetensors.index.json");
        let index_content = r#"{
            "metadata": {"total_size": 48},
            "weight_map": {
                "layer1.weight": "model-00001-of-00002.safetensors",
                "layer2.weight": "model-00002-of-00002.safetensors"
            }
        }"#;
        std::fs::write(&index_path, index_content).unwrap();

        let mut loader = TensorLoader::open_index(&index_path, hmll::Device::Cpu).unwrap();

        assert_eq!(loader.tensor_index.len(), 2);

        let result1 = loader.fetch_tensor("layer1.weight").unwrap();
        let result2 = loader.fetch_tensor("layer2.weight").unwrap();

        assert_eq!(result1.buffer.as_slice().unwrap(), data1.as_slice());
        assert_eq!(result2.buffer.as_slice().unwrap(), data2.as_slice());
    }

    #[test]
    fn test_open_nonexistent_file() {
        let result = TensorLoader::open(&["/nonexistent/path.safetensors"], hmll::Device::Cpu);
        assert!(result.is_err());
    }

    #[test]
    fn test_fetch_tensors_empty() {
        let data = vec![0u8; 16];
        let file = create_test_safetensor(vec![("weight", Dtype::F32, vec![2, 2], data)]);

        let mut loader = TensorLoader::open(&[file.path()], hmll::Device::Cpu).unwrap();

        let results = loader.fetch_tensors(&[]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_fetch_tensors_single() {
        let data: Vec<u8> = (0..16).collect();
        let file = create_test_safetensor(vec![("weight", Dtype::F32, vec![2, 2], data.clone())]);

        let mut loader = TensorLoader::open(&[file.path()], hmll::Device::Cpu).unwrap();

        let results = loader.fetch_tensors(&["weight"]).unwrap();
        assert_eq!(results.len(), 1);

        let (name, result) = &results[0];
        assert_eq!(name, "weight");
        assert_eq!(result.info.dtype, Dtype::F32);
        assert_eq!(result.info.shape, vec![2, 2]);
        assert_eq!(result.buffer.as_slice().unwrap(), data.as_slice());
        assert!(result.shard_slice.is_none());
    }

    #[test]
    fn test_fetch_tensors_multiple_same_file() {
        let data1: Vec<u8> = (0..16).collect();
        let data2: Vec<u8> = (50..82).collect();
        let data3: Vec<u8> = (100..108).collect();

        let file = create_test_safetensor(vec![
            ("weight", Dtype::F32, vec![2, 2], data1.clone()),
            ("bias", Dtype::F32, vec![8], data2.clone()),
            ("scale", Dtype::F32, vec![2], data3.clone()),
        ]);

        let mut loader = TensorLoader::open(&[file.path()], hmll::Device::Cpu).unwrap();

        let results = loader.fetch_tensors(&["weight", "bias", "scale"]).unwrap();
        assert_eq!(results.len(), 3);

        // Results may be out of order, so collect into a map
        let result_map: HashMap<&str, &FetchResult> = results
            .iter()
            .map(|(name, result)| (name.as_str(), result))
            .collect();

        let result_w = result_map.get("weight").unwrap();
        let result_b = result_map.get("bias").unwrap();
        let result_s = result_map.get("scale").unwrap();

        assert_eq!(result_w.buffer.as_slice().unwrap(), data1.as_slice());
        assert_eq!(result_b.buffer.as_slice().unwrap(), data2.as_slice());
        assert_eq!(result_s.buffer.as_slice().unwrap(), data3.as_slice());
    }

    #[test]
    fn test_fetch_tensors_multiple_files() {
        let data1: Vec<u8> = (0..16).collect();
        let data2: Vec<u8> = (100..132).collect();

        let file1 =
            create_test_safetensor(vec![("tensor_a", Dtype::F32, vec![2, 2], data1.clone())]);
        let file2 =
            create_test_safetensor(vec![("tensor_b", Dtype::F32, vec![2, 4], data2.clone())]);

        let mut loader =
            TensorLoader::open(&[file1.path(), file2.path()], hmll::Device::Cpu).unwrap();

        let results = loader.fetch_tensors(&["tensor_a", "tensor_b"]).unwrap();
        assert_eq!(results.len(), 2);

        let result_map: HashMap<&str, &FetchResult> = results
            .iter()
            .map(|(name, result)| (name.as_str(), result))
            .collect();

        let result_a = result_map.get("tensor_a").unwrap();
        let result_b = result_map.get("tensor_b").unwrap();

        assert_eq!(result_a.info.shape, vec![2, 2]);
        assert_eq!(result_b.info.shape, vec![2, 4]);
        assert_eq!(result_a.buffer.as_slice().unwrap(), data1.as_slice());
        assert_eq!(result_b.buffer.as_slice().unwrap(), data2.as_slice());
    }

    #[test]
    fn test_fetch_tensors_not_found() {
        let data = vec![0u8; 16];
        let file = create_test_safetensor(vec![("weight", Dtype::F32, vec![2, 2], data)]);

        let mut loader = TensorLoader::open(&[file.path()], hmll::Device::Cpu).unwrap();

        let result = loader.fetch_tensors(&["weight", "nonexistent"]);
        assert!(result.is_err());

        match result.unwrap_err() {
            LoaderError::TensorNotFound(name) => assert_eq!(name, "nonexistent"),
            e => panic!("Expected TensorNotFound error, got {:?}", e),
        }
    }

    #[test]
    fn test_fetch_tensors_partial_from_multiple_files() {
        // Test fetching subset of tensors from a multi-file model
        let data1: Vec<u8> = (0..16).collect();
        let data2: Vec<u8> = (20..36).collect();
        let data3: Vec<u8> = (100..132).collect();

        let file1 = create_test_safetensor(vec![
            ("layer1.weight", Dtype::F32, vec![2, 2], data1.clone()),
            ("layer1.bias", Dtype::F32, vec![2, 2], data2.clone()),
        ]);
        let file2 = create_test_safetensor(vec![(
            "layer2.weight",
            Dtype::F32,
            vec![2, 4],
            data3.clone(),
        )]);

        let mut loader =
            TensorLoader::open(&[file1.path(), file2.path()], hmll::Device::Cpu).unwrap();

        // Only fetch layer1.weight and layer2.weight (skip layer1.bias)
        let results = loader
            .fetch_tensors(&["layer1.weight", "layer2.weight"])
            .unwrap();
        assert_eq!(results.len(), 2);

        let result_map: HashMap<&str, &FetchResult> = results
            .iter()
            .map(|(name, result)| (name.as_str(), result))
            .collect();

        let result1 = result_map.get("layer1.weight").unwrap();
        let result2 = result_map.get("layer2.weight").unwrap();

        assert_eq!(result1.buffer.as_slice().unwrap(), data1.as_slice());
        assert_eq!(result2.buffer.as_slice().unwrap(), data3.as_slice());
    }

    #[test]
    fn test_open_index_fetch_tensors() {
        let dir = tempdir().expect("Failed to create temp dir");

        // Create three safetensors files
        let data1: Vec<u8> = (0..16).collect();
        let data2: Vec<u8> = (50..66).collect();
        let data3: Vec<u8> = (100..132).collect();

        let path1 = dir.path().join("model-00001-of-00003.safetensors");
        let path2 = dir.path().join("model-00002-of-00003.safetensors");
        let path3 = dir.path().join("model-00003-of-00003.safetensors");

        serialize_to_file(
            vec![(
                "layer1.weight",
                crate::tensor::TensorView::new(Dtype::F32, vec![2, 2], &data1).unwrap(),
            )],
            None,
            &path1,
        )
        .unwrap();

        serialize_to_file(
            vec![(
                "layer2.weight",
                crate::tensor::TensorView::new(Dtype::F32, vec![2, 2], &data2).unwrap(),
            )],
            None,
            &path2,
        )
        .unwrap();

        serialize_to_file(
            vec![(
                "layer3.weight",
                crate::tensor::TensorView::new(Dtype::F32, vec![2, 4], &data3).unwrap(),
            )],
            None,
            &path3,
        )
        .unwrap();

        // Create index.json
        let index_path = dir.path().join("model.safetensors.index.json");
        let index_content = r#"{
            "metadata": {"total_size": 64},
            "weight_map": {
                "layer1.weight": "model-00001-of-00003.safetensors",
                "layer2.weight": "model-00002-of-00003.safetensors",
                "layer3.weight": "model-00003-of-00003.safetensors"
            }
        }"#;
        std::fs::write(&index_path, index_content).unwrap();

        let mut loader = TensorLoader::open_index(&index_path, hmll::Device::Cpu).unwrap();

        assert_eq!(loader.tensor_index.len(), 3);

        // Batch fetch all tensors
        let results = loader
            .fetch_tensors(&["layer1.weight", "layer2.weight", "layer3.weight"])
            .unwrap();
        assert_eq!(results.len(), 3);

        let result_map: HashMap<&str, &FetchResult> = results
            .iter()
            .map(|(name, result)| (name.as_str(), result))
            .collect();

        let result1 = result_map.get("layer1.weight").unwrap();
        let result2 = result_map.get("layer2.weight").unwrap();
        let result3 = result_map.get("layer3.weight").unwrap();

        assert_eq!(result1.info.shape, vec![2, 2]);
        assert_eq!(result2.info.shape, vec![2, 2]);
        assert_eq!(result3.info.shape, vec![2, 4]);

        assert_eq!(result1.buffer.as_slice().unwrap(), data1.as_slice());
        assert_eq!(result2.buffer.as_slice().unwrap(), data2.as_slice());
        assert_eq!(result3.buffer.as_slice().unwrap(), data3.as_slice());
    }
}
