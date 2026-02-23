//! module for loading sharded models. We call a shard a single safetensors file that contains a
//! subset of the tensors comprising a model.

use std::{fmt::Display, fs::File, io::BufReader, path::Path};

use hashbrown::HashMap;
use serde::Deserialize;

/// Index file parsing errors
#[derive(Debug)]
pub enum IndexParsingError {
    /// fs errors
    FileReadError(std::io::Error),
    /// json errors
    JsonParseError(serde_json::Error),
}

impl Display for IndexParsingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IndexParsingError::FileReadError(e) => write!(f, "Failed to read index file: {}", e),
            IndexParsingError::JsonParseError(e) => write!(f, "Failed to parse index JSON: {}", e),
        }
    }
}

impl From<std::io::Error> for IndexParsingError {
    fn from(err: std::io::Error) -> Self {
        IndexParsingError::FileReadError(err)
    }
}

impl From<serde_json::Error> for IndexParsingError {
    fn from(err: serde_json::Error) -> Self {
        IndexParsingError::JsonParseError(err)
    }
}

/// Metadata from the index file.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct IndexMetadata {
    /// Total size of all tensors in bytes.
    #[serde(default)]
    total_size: Option<u64>,

    /// Additional metadata fields (preserved for round-tripping).
    #[serde(flatten)]
    extra: HashMap<String, String>,
}

/// Struct representing the index file for sharded models
#[derive(Deserialize)]
pub struct Index {
    metadata: IndexMetadata,
    weight_map: HashMap<String, String>,
}

impl Index {
    /// Get the shard file path for a given tensor name
    pub fn get_tensor_shard(&self, tensor_name: &str) -> Option<&String> {
        self.weight_map.get(tensor_name)
    }

    /// Get total size of all tensors in bytes, if available
    pub fn total_size(&self) -> Option<u64> {
        self.metadata.total_size
    }

    /// Get metadata value for the given key
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.extra.get(key)
    }
}

/// Parse an index file
pub fn parse_index(file: impl AsRef<Path>) -> Result<Index, IndexParsingError> {
    let file = BufReader::new(File::open(file)?);
    let index: Index = serde_json::from_reader(file)?;
    Ok(index)
}
