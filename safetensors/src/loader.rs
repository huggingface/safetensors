//! TODO:

use core::{error::Error, fmt::Display};
use std::path::Path;

use hashbrown::HashMap;

use crate::{
    index::IndexParsingError,
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

/// TODO:
#[derive(Debug)]
pub struct TensorLoader {
    loader: hmll::WeightLoader,
    source_offsets: Vec<usize>,
    tensor_index: HashMap<String, (usize, TensorInfo)>,
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

    /// Get the tensor's data from it's source file
    pub fn fetch_tensor(&mut self, name: &str) -> Result<(hmll::Buffer, TensorInfo), LoaderError> {
        let (source_idx, info) = self
            .tensor_index
            .get(name)
            .ok_or_else(|| LoaderError::TensorNotFound(name.to_owned()))?;

        let source_offset = self.source_offsets[*source_idx];
        let (start, end) = info.data_offsets;
        let abs_start = start + source_offset;
        let abs_end = end + source_offset;

        let buffer = self.loader.fetch(abs_start..abs_end, *source_idx)?;

        Ok((buffer, info.clone()))
    }

    /// Fetch tensor data in batch for performance
    pub fn fetch_tensors(
        &mut self,
        names: &[&str],
    ) -> Result<Vec<(String, hmll::Buffer, TensorInfo)>, LoaderError> {
        if names.is_empty() {
            return Ok(Vec::new());
        }

        let mut by_source: HashMap<usize, Vec<(&str, hmll::Range, &TensorInfo)>> = HashMap::new();
        for name in names {
            let (source_idx, info) = self
                .tensor_index
                .get(*name)
                .ok_or_else(|| LoaderError::TensorNotFound((*name).to_owned()))?;
            let source_offset = self.source_offsets[*source_idx];
            let (start, end) = info.data_offsets;
            let range = hmll::Range::new(source_offset + start, source_offset + end);

            by_source
                .entry(*source_idx)
                .or_default()
                .push((name, range, info));
        }

        let mut results = Vec::with_capacity(names.len());
        for (source_idx, tensors) in by_source {
            let ranges: Vec<hmll::Range> = tensors.iter().map(|(_, r, _)| *r).collect();
            let buffers = self.loader.fetchv(&ranges, source_idx)?;
            for ((name, _, info), buffer) in tensors.into_iter().zip(buffers) {
                results.push((name.to_owned(), buffer, info.clone()));
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

        let (buffer, info) = loader.fetch_tensor("weight").unwrap();

        assert_eq!(info.dtype, Dtype::F32);
        assert_eq!(info.shape, vec![2, 2]);
        assert_eq!(buffer.len(), 16);
        assert_eq!(buffer.as_slice().unwrap(), data.as_slice());
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

        let (buf_a, info_a) = loader.fetch_tensor("tensor_a").unwrap();
        let (buf_b, info_b) = loader.fetch_tensor("tensor_b").unwrap();

        assert_eq!(info_a.shape, vec![2, 2]);
        assert_eq!(info_b.shape, vec![2, 4]);
        assert_eq!(buf_a.as_slice().unwrap(), data1.as_slice());
        assert_eq!(buf_b.as_slice().unwrap(), data2.as_slice());
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

        let (buf_w, _) = loader.fetch_tensor("weight").unwrap();
        let (buf_b, _) = loader.fetch_tensor("bias").unwrap();

        assert_eq!(buf_w.as_slice().unwrap(), data1.as_slice());
        assert_eq!(buf_b.as_slice().unwrap(), data2.as_slice());
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

        let (buf1, _) = loader.fetch_tensor("layer1.weight").unwrap();
        let (buf2, _) = loader.fetch_tensor("layer2.weight").unwrap();

        assert_eq!(buf1.as_slice().unwrap(), data1.as_slice());
        assert_eq!(buf2.as_slice().unwrap(), data2.as_slice());
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

        let (name, buffer, info) = &results[0];
        assert_eq!(name, "weight");
        assert_eq!(info.dtype, Dtype::F32);
        assert_eq!(info.shape, vec![2, 2]);
        assert_eq!(buffer.as_slice().unwrap(), data.as_slice());
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
        let result_map: HashMap<&str, (&hmll::Buffer, &TensorInfo)> = results
            .iter()
            .map(|(name, buf, info)| (name.as_str(), (buf, info)))
            .collect();

        let (buf_w, _) = result_map.get("weight").unwrap();
        let (buf_b, _) = result_map.get("bias").unwrap();
        let (buf_s, _) = result_map.get("scale").unwrap();

        assert_eq!(buf_w.as_slice().unwrap(), data1.as_slice());
        assert_eq!(buf_b.as_slice().unwrap(), data2.as_slice());
        assert_eq!(buf_s.as_slice().unwrap(), data3.as_slice());
    }

    #[test]
    fn test_fetch_tensors_multiple_files() {
        let data1: Vec<u8> = (0..16).collect();
        let data2: Vec<u8> = (100..132).collect();

        let file1 = create_test_safetensor(vec![("tensor_a", Dtype::F32, vec![2, 2], data1.clone())]);
        let file2 = create_test_safetensor(vec![("tensor_b", Dtype::F32, vec![2, 4], data2.clone())]);

        let mut loader =
            TensorLoader::open(&[file1.path(), file2.path()], hmll::Device::Cpu).unwrap();

        let results = loader.fetch_tensors(&["tensor_a", "tensor_b"]).unwrap();
        assert_eq!(results.len(), 2);

        let result_map: HashMap<&str, (&hmll::Buffer, &TensorInfo)> = results
            .iter()
            .map(|(name, buf, info)| (name.as_str(), (buf, info)))
            .collect();

        let (buf_a, info_a) = result_map.get("tensor_a").unwrap();
        let (buf_b, info_b) = result_map.get("tensor_b").unwrap();

        assert_eq!(info_a.shape, vec![2, 2]);
        assert_eq!(info_b.shape, vec![2, 4]);
        assert_eq!(buf_a.as_slice().unwrap(), data1.as_slice());
        assert_eq!(buf_b.as_slice().unwrap(), data2.as_slice());
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
        let file2 = create_test_safetensor(vec![("layer2.weight", Dtype::F32, vec![2, 4], data3.clone())]);

        let mut loader =
            TensorLoader::open(&[file1.path(), file2.path()], hmll::Device::Cpu).unwrap();

        // Only fetch layer1.weight and layer2.weight (skip layer1.bias)
        let results = loader
            .fetch_tensors(&["layer1.weight", "layer2.weight"])
            .unwrap();
        assert_eq!(results.len(), 2);

        let result_map: HashMap<&str, (&hmll::Buffer, &TensorInfo)> = results
            .iter()
            .map(|(name, buf, info)| (name.as_str(), (buf, info)))
            .collect();

        let (buf1, _) = result_map.get("layer1.weight").unwrap();
        let (buf2, _) = result_map.get("layer2.weight").unwrap();

        assert_eq!(buf1.as_slice().unwrap(), data1.as_slice());
        assert_eq!(buf2.as_slice().unwrap(), data3.as_slice());
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

        let result_map: HashMap<&str, (&hmll::Buffer, &TensorInfo)> = results
            .iter()
            .map(|(name, buf, info)| (name.as_str(), (buf, info)))
            .collect();

        let (buf1, info1) = result_map.get("layer1.weight").unwrap();
        let (buf2, info2) = result_map.get("layer2.weight").unwrap();
        let (buf3, info3) = result_map.get("layer3.weight").unwrap();

        assert_eq!(info1.shape, vec![2, 2]);
        assert_eq!(info2.shape, vec![2, 2]);
        assert_eq!(info3.shape, vec![2, 4]);

        assert_eq!(buf1.as_slice().unwrap(), data1.as_slice());
        assert_eq!(buf2.as_slice().unwrap(), data2.as_slice());
        assert_eq!(buf3.as_slice().unwrap(), data3.as_slice());
    }
}
