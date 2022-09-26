//! Module Containing the most important structures
use crate::slice::{InvalidSlice, SliceIterator, TensorIndexer};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};

/// Possible errors that could occur while reading
/// A Safetensor file.
#[derive(Debug)]
pub enum SafeTensorError {
    /// The header is an invalid UTF-8 string and cannot be read.
    InvalidHeader,
    /// The header does contain a valid string, but it is not valid JSON.
    InvalidHeaderDeserialization,
    /// The tensor name was not found in the archive
    TensorNotFound,
    /// Invalid information between shape, dtype and the proposed offsets in the file
    TensorInvalidInfo,
}

fn prepare<'hash, 'data>(
    data: &'hash HashMap<String, TensorView<'data>>,
    data_info: &'hash Option<HashMap<String, String>>,
) -> (Metadata, Vec<&'hash TensorView<'data>>, usize) {
    let mut tensors: Vec<&TensorView> = vec![];
    let mut hmetadata = HashMap::new();
    let mut offset = 0;
    for (name, tensor) in data {
        let n = tensor.data.len();
        let tensor_info = TensorInfo {
            dtype: tensor.dtype,
            shape: tensor.shape.to_vec(),
            data_offsets: (offset, offset + n),
        };
        offset += n;
        hmetadata.insert(name.to_string(), tensor_info);
        tensors.push(tensor);
    }

    let metadata: Metadata = Metadata::new(data_info.clone(), hmetadata);

    (metadata, tensors, offset)
}

/// Serialize to an owned byte buffer the dictionnary of tensors.
pub fn serialize(
    data: &HashMap<String, TensorView>,
    data_info: &Option<HashMap<String, String>>,
) -> Vec<u8> {
    let (metadata, tensors, offset) = prepare(data, data_info);
    let metadata_buf = serde_json::to_string(&metadata).unwrap().into_bytes();
    let expected_size = 8 + metadata_buf.len() + offset;
    let mut buffer: Vec<u8> = Vec::with_capacity(expected_size);
    let n: u64 = metadata_buf.len() as u64;
    buffer.extend(&n.to_le_bytes().to_vec());
    buffer.extend(&metadata_buf);
    for tensor in tensors {
        buffer.extend(tensor.data);
    }
    buffer
}

/// Serialize to a regular file the dictionnary of tensors.
/// Writing directly to file reduces the need to allocate the whole amount to
/// memory.
pub fn serialize_to_file(
    data: &HashMap<String, TensorView>,
    data_info: &Option<HashMap<String, String>>,
    filename: &str,
) -> Result<(), std::io::Error> {
    let (metadata, tensors, _) = prepare(data, data_info);
    let metadata_buf = serde_json::to_string(&metadata).unwrap().into_bytes();
    let n: u64 = metadata_buf.len() as u64;
    let mut f = BufWriter::new(File::create(filename)?);
    f.write_all(n.to_le_bytes().as_ref())?;
    f.write_all(&metadata_buf)?;
    for tensor in tensors {
        f.write_all(tensor.data)?;
    }
    f.flush()?;
    Ok(())
}

/// A structure owning some metadata to lookup tensors on a shared `data`
/// byte-buffer (not owned).
pub struct SafeTensors<'data> {
    metadata: Metadata,
    data: &'data [u8],
}

impl<'data> SafeTensors<'data> {
    /// Given a byte-buffer representing the whole safetensor file
    /// parses it and returns the Deserialized form (No Tensor allocation).
    pub fn deserialize<'in_data>(buffer: &'in_data [u8]) -> Result<Self, SafeTensorError>
    where
        'in_data: 'data,
    {
        let arr: [u8; 8] = [
            buffer[0], buffer[1], buffer[2], buffer[3], buffer[4], buffer[5], buffer[6], buffer[7],
        ];
        let n = u64::from_le_bytes(arr) as usize;
        let string =
            std::str::from_utf8(&buffer[8..8 + n]).map_err(|_| SafeTensorError::InvalidHeader)?;
        let metadata: Metadata = serde_json::from_str(string)
            .map_err(|_| SafeTensorError::InvalidHeaderDeserialization)?;
        let data = &buffer[n + 8..];
        Ok(Self { metadata, data })
    }

    /// Allow the user to iterate over tensors within the SafeTensors.
    /// The tensors returned are merely views and the data is not owned by this
    /// structure.
    pub fn tensors(&self) -> Vec<(String, TensorView<'_>)> {
        let mut tensors = vec![];
        for (name, info) in &self.metadata.tensors {
            let tensorview = TensorView {
                dtype: info.dtype,
                shape: info.shape.clone(),
                data: &self.data[info.data_offsets.0..info.data_offsets.1],
            };
            tensors.push((name.to_string(), tensorview));
        }
        tensors
    }

    /// Allow the user to get a specific tensor within the SafeTensors.
    /// The tensor returned is merely a view and the data is not owned by this
    /// structure.
    pub fn tensor(&self, tensor_name: &str) -> Result<TensorView<'_>, SafeTensorError> {
        if let Some(info) = &self.metadata.tensors.get(tensor_name) {
            Ok(TensorView {
                dtype: info.dtype,
                shape: info.shape.clone(),
                data: &self.data[info.data_offsets.0..info.data_offsets.1],
            })
        } else {
            Err(SafeTensorError::TensorNotFound)
        }
    }

    /// Allow the user to get a specific tensor within the SafeTensors.
    /// The tensor returned is merely a view and the data is not owned by this
    /// structure.
    pub fn names(&self) -> Vec<&'_ String> {
        self.metadata.tensors.iter().map(|(name, _)| name).collect()
    }
}

/// The stuct representing the header of safetensor files which allow
/// indexing into the raw byte-buffer array and how to interpret it.
#[derive(Debug, Deserialize, Serialize)]
pub struct Metadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "__metadata__")]
    metadata: Option<HashMap<String, String>>,
    #[serde(flatten)]
    tensors: HashMap<String, TensorInfo>,
}

impl Metadata {
    fn new(
        metadata: Option<HashMap<String, String>>,
        tensors: HashMap<String, TensorInfo>,
    ) -> Self {
        Self { metadata, tensors }
    }

    /// Gives back the tensor metadata
    pub fn tensors(&self) -> &HashMap<String, TensorInfo> {
        &self.tensors
    }

    /// Gives back the tensor metadata
    pub fn metadata(&self) -> &Option<HashMap<String, String>> {
        &self.metadata
    }
}

/// A view of a Tensor within the file.
/// Contains references to data within the full byte-buffer
/// And is thus a readable view of a single tensor
#[derive(Debug)]
pub struct TensorView<'data> {
    pub(crate) dtype: Dtype,
    pub(crate) shape: Vec<usize>,
    pub(crate) data: &'data [u8],
}

impl<'data> TensorView<'data> {
    /// Create new tensor view
    pub fn new(
        dtype: Dtype,
        shape: Vec<usize>,
        data: &'data [u8],
    ) -> Result<Self, SafeTensorError> {
        let nelements: usize = shape.iter().product();
        let nbytes = nelements * dtype.size();
        if data.len() != nbytes {
            println!("{:?} != {:?}", data.len(), nbytes);
            Err(SafeTensorError::TensorInvalidInfo)
        } else {
            Ok(Self { dtype, shape, data })
        }
    }
    /// The current tensor dtype
    pub fn get_dtype(&self) -> Dtype {
        self.dtype
    }

    /// The current tensor shape
    pub fn get_shape(&'data self) -> &'data [usize] {
        &self.shape
    }

    /// The current tensor byte-buffer
    pub fn get_data(&self) -> &'data [u8] {
        self.data
    }

    /// The various pieces of the data buffer according to the asked slice
    pub fn get_sliced_data(
        &'data self,
        slices: Vec<TensorIndexer>,
    ) -> Result<SliceIterator<'data>, InvalidSlice> {
        SliceIterator::new(self, slices)
    }
}

/// A single tensor information.
/// Endianness is assumed to be little endian
/// Ordering is assumed to be 'C'.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct TensorInfo {
    /// The type of each element of the tensor
    pub dtype: Dtype,
    /// The shape of the tensor
    pub shape: Vec<usize>,
    /// The offsets to find the data within the byte-buffer array.
    pub data_offsets: (usize, usize),
}

/// The various available dtypes
#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Dtype {
    /// Boolan type
    BOOL,
    /// Unsigned byte
    U8,
    /// Signed byte
    I8,
    /// Signed integer (16-bit)
    I16,
    /// Unsigned integer (16-bit)
    U16,
    /// Signed integer (32-bit)
    I32,
    /// Unsigned integer (32-bit)
    U32,
    /// Signed integer (64-bit)
    I64,
    /// Unsigned integer (64-bit)
    U64,
    /// Half-precision floating point
    F16,
    /// Brain floating point
    BF16,
    /// Floating point (32-bit)
    F32,
    /// Floating point (64-bit)
    F64,
}

impl Dtype {
    /// Gives out the size (in bytes) of 1 element of this dtype.
    pub fn size(&self) -> usize {
        match self {
            Dtype::BOOL => 1,
            Dtype::U8 => 1,
            Dtype::I8 => 1,
            Dtype::I16 => 2,
            Dtype::U16 => 2,
            Dtype::I32 => 4,
            Dtype::U32 => 4,
            Dtype::I64 => 8,
            Dtype::U64 => 8,
            Dtype::F16 => 2,
            Dtype::BF16 => 2,
            Dtype::F32 => 4,
            Dtype::F64 => 8,
        }
    }
}

// /// A struct representing a Tensor, the byte-buffer is not owned
// /// but dtype a shape are.
// pub struct Tensor<'data> {
//     shape: Vec<usize>,
//     dtype: Dtype,
//     data: &'data [u8],
// }
//
// impl<'a> Tensor<'a> {
//     /// Simple Tensor creation.
//     pub fn new(data: &'a [u8], dtype: Dtype, shape: Vec<usize>) -> Self {
//         Self { data, dtype, shape }
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::slice::IndexOp;

    #[test]
    fn test_serialization() {
        let data: Vec<u8> = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let shape = vec![1, 2, 3];
        let attn_0 = TensorView::new(Dtype::F32, shape, &data).unwrap();
        let metadata: HashMap<String, TensorView> =
            [("attn.0".to_string(), attn_0)].into_iter().collect();

        let out = serialize(&metadata, &None);
        let _parsed = SafeTensors::deserialize(&out).unwrap();
    }

    #[test]
    fn test_slicing() {
        let data: Vec<u8> = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let attn_0 = TensorView {
            dtype: Dtype::F32,
            shape: vec![1, 2, 3],
            data: &data,
        };
        let metadata: HashMap<String, TensorView> =
            [("attn.0".to_string(), attn_0)].into_iter().collect();

        let out = serialize(&metadata, &None);
        let parsed = SafeTensors::deserialize(&out).unwrap();

        let out_buffer: Vec<u8> = parsed
            .tensor("attn.0")
            .unwrap()
            .slice((.., ..1))
            .unwrap()
            .flat_map(|b| b.to_vec())
            .collect();
        assert_eq!(out_buffer, vec![0u8, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 64]);
        assert_eq!(
            out_buffer,
            vec![0.0f32, 1.0, 2.0]
                .into_iter()
                .flat_map(|f| f.to_le_bytes())
                .collect::<Vec<_>>()
        );
        let out_buffer: Vec<u8> = parsed
            .tensor("attn.0")
            .unwrap()
            .slice((.., .., ..1))
            .unwrap()
            .flat_map(|b| b.to_vec())
            .collect();
        assert_eq!(out_buffer, vec![0u8, 0, 0, 0, 0, 0, 64, 64]);
        assert_eq!(
            out_buffer,
            vec![0.0f32, 3.0]
                .into_iter()
                .flat_map(|f| f.to_le_bytes())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_gpt2() {
        gpt2_like(12, "gpt2");
    }

    #[test]
    fn test_gpt2_tiny() {
        gpt2_like(6, "gpt2_tiny");
    }

    fn gpt2_like(n_heads: usize, model_id: &str) {
        let mut tensors_desc = vec![];
        tensors_desc.push(("wte".to_string(), vec![50257, 768]));
        tensors_desc.push(("wpe".to_string(), vec![1024, 768]));
        for i in 0..n_heads {
            tensors_desc.push((format!("h.{}.ln_1.weight", i), vec![768]));
            tensors_desc.push((format!("h.{}.ln_1.bias", i), vec![768]));
            tensors_desc.push((format!("h.{}.attn.bias", i), vec![1, 1, 1024, 1024]));
            tensors_desc.push((format!("h.{}.attn.c_attn.weight", i), vec![768, 2304]));
            tensors_desc.push((format!("h.{}.attn.c_attn.bias", i), vec![2304]));
            tensors_desc.push((format!("h.{}.attn.c_proj.weight", i), vec![768, 768]));
            tensors_desc.push((format!("h.{}.attn.c_proj.bias", i), vec![768]));
            tensors_desc.push((format!("h.{}.ln_2.weight", i), vec![768]));
            tensors_desc.push((format!("h.{}.ln_2.bias", i), vec![768]));
            tensors_desc.push((format!("h.{}.mlp.c_fc.weight", i), vec![768, 3072]));
            tensors_desc.push((format!("h.{}.mlp.c_fc.bias", i), vec![3072]));
            tensors_desc.push((format!("h.{}.mlp.c_proj.weight", i), vec![3072, 768]));
            tensors_desc.push((format!("h.{}.mlp.c_proj.bias", i), vec![768]));
        }
        tensors_desc.push(("ln_f.weight".to_string(), vec![768]));
        tensors_desc.push(("ln_f.bias".to_string(), vec![768]));

        let dtype = Dtype::F32;
        let n: usize = tensors_desc
            .iter()
            .map(|(_, shape)| shape.iter().product::<usize>())
            .sum::<usize>()
            * dtype.size(); // 4
        let all_data = vec![0; n];
        let mut metadata: HashMap<String, TensorView> = HashMap::new();
        let mut offset = 0;
        for (name, shape) in tensors_desc {
            let n: usize = shape.iter().product();
            let buffer = &all_data[offset..offset + n * dtype.size()];
            let tensor = TensorView::new(dtype, shape, buffer).unwrap();
            metadata.insert(name, tensor);
            offset += n;
        }

        let filename = format!("./out_{}.bin", model_id);

        let out = serialize(&metadata, &None);

        std::fs::write(&filename, out).unwrap();

        let raw = std::fs::read(&filename).unwrap();
        let _deserialized = SafeTensors::deserialize(&raw).unwrap();
        std::fs::remove_file(&filename).unwrap();
    }

    #[test]
    fn test_deserialization() {
        let serialized = b"<\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";

        let loaded = SafeTensors::deserialize(serialized).unwrap();

        assert_eq!(loaded.names(), vec!["test"]);
        let tensor = loaded.tensor("test").unwrap();
        assert_eq!(tensor.get_shape(), vec![2, 2]);
        assert_eq!(tensor.get_dtype(), Dtype::I32);
        // 16 bytes
        assert_eq!(tensor.get_data(), b"\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0");
    }
}
