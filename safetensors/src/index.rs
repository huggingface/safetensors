//! Safetensors index file support for sharded models.
//!
//! Large models are often split across multiple safetensors files (shards).
//! The `model.safetensors.index.json` file maps tensor names to their shard files.
//!
//! # Example Index File
//!
//! ```json
//! {
//!   "metadata": {"total_size": 28966928384},
//!   "weight_map": {
//!     "lm_head.weight": "model-00006-of-00006.safetensors",
//!     "model.embed_tokens.weight": "model-00001-of-00006.safetensors",
//!     "model.layers.0.mlp.gate_proj.weight": "model-00001-of-00006.safetensors"
//!   }
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io;
use std::path::Path;

/// Error type for index file operations.
#[derive(Debug)]
pub enum IndexError {
    /// Failed to read the index file.
    Io(io::Error),
    /// Failed to parse the index JSON.
    Json(serde_json::Error),
    /// Tensor not found in weight map.
    TensorNotFound(String),
}

impl std::fmt::Display for IndexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IndexError::Io(e) => write!(f, "failed to read index file: {e}"),
            IndexError::Json(e) => write!(f, "failed to parse index JSON: {e}"),
            IndexError::TensorNotFound(name) => write!(f, "tensor not found in index: {name}"),
        }
    }
}

impl std::error::Error for IndexError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            IndexError::Io(e) => Some(e),
            IndexError::Json(e) => Some(e),
            IndexError::TensorNotFound(_) => None,
        }
    }
}

impl From<io::Error> for IndexError {
    fn from(e: io::Error) -> Self {
        IndexError::Io(e)
    }
}

impl From<serde_json::Error> for IndexError {
    fn from(e: serde_json::Error) -> Self {
        IndexError::Json(e)
    }
}

/// Result type for index operations.
pub type Result<T> = std::result::Result<T, IndexError>;

/// Metadata from the index file.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IndexMetadata {
    /// Total size of all tensors in bytes.
    #[serde(default)]
    pub total_size: Option<u64>,

    /// Additional metadata fields (preserved for round-tripping).
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// Safetensors index file for sharded models.
///
/// This structure represents the `model.safetensors.index.json` file format
/// used by HuggingFace transformers for sharded models.
///
/// # Example
///
/// ```no_run
/// use safetensors::index::SafetensorsIndex;
///
/// let index = SafetensorsIndex::load("model.safetensors.index.json")?;
///
/// // List all shards
/// for shard in index.shards() {
///     println!("Shard: {shard}");
/// }
///
/// // Find which shard contains a tensor
/// if let Some(shard) = index.get_shard("model.layers.0.weight") {
///     println!("Tensor is in: {shard}");
/// }
/// # Ok::<(), safetensors::index::IndexError>(())
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetensorsIndex {
    /// Metadata about the model.
    #[serde(default)]
    pub metadata: IndexMetadata,

    /// Maps tensor names to shard filenames.
    pub weight_map: HashMap<String, String>,
}

impl SafetensorsIndex {
    /// Load an index from a JSON file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the `model.safetensors.index.json` file
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Self::from_json(&content)
    }

    /// Parse an index from JSON string.
    pub fn from_json(json: &str) -> Result<Self> {
        Ok(serde_json::from_str(json)?)
    }

    /// Serialize the index to JSON string.
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Get the shard filename for a tensor.
    ///
    /// Returns `None` if the tensor is not in the weight map.
    #[inline]
    pub fn get_shard(&self, tensor_name: &str) -> Option<&str> {
        self.weight_map.get(tensor_name).map(|s| s.as_str())
    }

    /// Get all unique shard filenames, sorted alphabetically.
    pub fn shards(&self) -> Vec<&str> {
        let mut shards: Vec<_> = self.weight_map.values().map(|s| s.as_str()).collect();
        shards.sort();
        shards.dedup();
        shards
    }

    /// Get the number of unique shards.
    pub fn num_shards(&self) -> usize {
        self.shards().len()
    }

    /// Get the number of tensors in the index.
    #[inline]
    pub fn num_tensors(&self) -> usize {
        self.weight_map.len()
    }

    /// Get all tensor names.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.weight_map.keys().map(|s| s.as_str())
    }

    /// Group tensors by their shard file.
    ///
    /// Returns a map from shard filename to list of tensor names.
    pub fn tensors_by_shard(&self) -> HashMap<&str, Vec<&str>> {
        let mut result: HashMap<&str, Vec<&str>> = HashMap::new();
        for (tensor, shard) in &self.weight_map {
            result
                .entry(shard.as_str())
                .or_default()
                .push(tensor.as_str());
        }
        result
    }

    /// Check if a tensor exists in the index.
    #[inline]
    pub fn contains_tensor(&self, tensor_name: &str) -> bool {
        self.weight_map.contains_key(tensor_name)
    }

    /// Get the total size in bytes, if available in metadata.
    #[inline]
    pub fn total_size(&self) -> Option<u64> {
        self.metadata.total_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXAMPLE_INDEX: &str = r#"{
        "metadata": {"total_size": 14000000000},
        "weight_map": {
            "lm_head.weight": "model-00002-of-00002.safetensors",
            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
            "model.layers.0.self_attn.k_proj.weight": "model-00001-of-00002.safetensors",
            "model.layers.1.self_attn.q_proj.weight": "model-00002-of-00002.safetensors"
        }
    }"#;

    #[test]
    fn test_parse_index() {
        let index = SafetensorsIndex::from_json(EXAMPLE_INDEX).unwrap();

        assert_eq!(index.total_size(), Some(14000000000));
        assert_eq!(index.num_tensors(), 5);
        assert_eq!(index.num_shards(), 2);
    }

    #[test]
    fn test_get_shard() {
        let index = SafetensorsIndex::from_json(EXAMPLE_INDEX).unwrap();

        assert_eq!(
            index.get_shard("lm_head.weight"),
            Some("model-00002-of-00002.safetensors")
        );
        assert_eq!(
            index.get_shard("model.embed_tokens.weight"),
            Some("model-00001-of-00002.safetensors")
        );
        assert_eq!(index.get_shard("nonexistent"), None);
    }

    #[test]
    fn test_shards() {
        let index = SafetensorsIndex::from_json(EXAMPLE_INDEX).unwrap();
        let shards = index.shards();

        assert_eq!(shards.len(), 2);
        assert!(shards.contains(&"model-00001-of-00002.safetensors"));
        assert!(shards.contains(&"model-00002-of-00002.safetensors"));
    }

    #[test]
    fn test_tensors_by_shard() {
        let index = SafetensorsIndex::from_json(EXAMPLE_INDEX).unwrap();
        let by_shard = index.tensors_by_shard();

        assert_eq!(by_shard.len(), 2);

        let shard1 = by_shard.get("model-00001-of-00002.safetensors").unwrap();
        assert_eq!(shard1.len(), 3);
        assert!(shard1.contains(&"model.embed_tokens.weight"));

        let shard2 = by_shard.get("model-00002-of-00002.safetensors").unwrap();
        assert_eq!(shard2.len(), 2);
        assert!(shard2.contains(&"lm_head.weight"));
    }

    #[test]
    fn test_round_trip() {
        let index = SafetensorsIndex::from_json(EXAMPLE_INDEX).unwrap();
        let json = index.to_json().unwrap();
        let index2 = SafetensorsIndex::from_json(&json).unwrap();

        assert_eq!(index.weight_map, index2.weight_map);
        assert_eq!(index.total_size(), index2.total_size());
    }

    #[test]
    fn test_missing_metadata() {
        let json = r#"{"weight_map": {"a": "shard.safetensors"}}"#;
        let index = SafetensorsIndex::from_json(json).unwrap();

        assert_eq!(index.total_size(), None);
        assert_eq!(index.num_tensors(), 1);
    }
}
