#![deny(missing_docs)]
//!# Safetensors
//!
//!This repository implements a new simple format for storing tensors
//!safely (as opposed to pickle) and that is still fast (zero-copy).
//!
//!##Format
//!
//!8 bytes: `N`, a u64 int, containing the size of the header
//!N bytes: a JSON utf-8 string representing the header.
//!         The header is a dict like {"TENSOR_NAME": {"dtype": "float16", "shape": [1, 16, 256], "offsets": (X, Y)}}, where X and Y are the offsets in the byte buffer of the tensor data
//!Rest of the file: byte-buffer.
//!
//!
//!## Yet another format ?
//!
//!The main rationale for this crate is to remove the need to use
//!`pickle` on `PyTorch` which is used by default.
//!There are other formats out there used by machine learning and more general
//!formats.
//!
//!
//!Let's take a look at alternatives and why this format is deemed interesting.
//!This is my very personal and probably biased view:
//!
//!| Format | Safe | Zero-copy | Lazy loading | No file size limit | Layout control | Flexibility |
//!| --- | --- | --- | --- | --- | --- | --- |
//!| pickle (PyTorch) | ✗ | ✗ | ✗ | 🗸 | ✗ | 🗸 |
//!| H5 (Tensorflow) | 🗸 | ✗ | 🗸 | 🗸 | ~ | ~ |
//!| SavedModel (Tensorflow) | 🗸? | ✗? | ✗ | 🗸  | 🗸 | ✗ | 🗸 |
//!| MsgPack (flax) | 🗸 | 🗸 | ✗ | 🗸 | ✗ | ✗ | ~ |
//!| Protobuf (ONNX) | 🗸 | ✗ | ✗ | ✗ | ✗ | ✗ | ~ |
//!| Cap'n'Proto | 🗸  | 🗸 | ~ | 🗸  | 🗸 | ~ | ~ |
//!| SafeTensors | 🗸 | 🗸 | 🗸 | 🗸 | 🗸 | ✗ |
//!
//!- Safe: Can I use a file randomly downloaded and expect not to run arbitrary code ?
//!- Zero-copy: Does reading the file require more memory than the original file ?
//!- Lazy loading: Can I inspect the file without loading everything ? And loading only
//!some tensors in it without scanning the whole file (distributed setting) ?
//!- Layout control: Lazy loading, is not necessarily enough since if the information about tensors is spread out in your file, then even if the information is lazily accessible you might have to access most of your file to read the available tensors (incurring many DISK -> RAM copies). Controlling layout to keep fast access to single tensors is important.
//!- No file size limit: Is there a limit to the file size ?
//!- Flexibility: Can I save custom code in the format and be able to use it later with zero extra code ? (~ means we can store more than pure tensors, but no custom code)
//!
//!
//!## Main oppositions
//!
//!- Pickle: Unsafe, runs arbitrary code
//!- H5: Apparently now discouraged for TF/Keras. Seems like a great fit otherwise actually. Some classic user after free issues: <https://www.cvedetails.com/vulnerability-list/vendor_id-15991/product_id-35054/Hdfgroup-Hdf5.html>. On a very different level than pickle security wise. Also 210k lines of code vs ~400 lines for this lib currently.
//!- SavedModel: Tensorflow specific (it contains TF graph information).
//!- MsgPack: No layout control to enable lazy loading (important for loading specific parts in distributed setting)
//!- Protobuf: Hard 2Go max file size limit
//!- Cap'n'proto: Float16 support is not present [link](https://capnproto.org/language.html#built-in-types) so using a manual wrapper over a byte-buffer would be necessary. Layout control seems possible but not trivial as buffers have limitations [link](https://stackoverflow.com/questions/48458839/capnproto-maximum-filesize).
//!
//!## Notes
//!
//!- Zero-copy: No format is really zero-copy in ML, it needs to go from disk to RAM/GPU RAM (that takes time). Also
//!    In PyTorch/numpy, you need a mutable buffer, and we don't really want to mutate a mmaped file, so 1 copy is really necessary to use the thing freely in user code. That being said, zero-copy is achievable in Rust if it's wanted and safety can be guaranteed by some other means.
//!    SafeTensors is not zero-copy for the header. The choice of JSON is pretty arbitrary, but since deserialization is <<< of the time required to load the actual tensor data and is readable I went that way, (also space is <<< to the tensor data).
//!
//!- Endianness: Little-endian. This can be modified later, but it feels really unecessary atm
//!- Order: 'C' or row-major. This seems to have won. We can add that information later if needed.
//!- Stride: No striding, all tensors need to be packed before being serialized. I have yet to see a case where it seems useful to have a strided tensor stored in serialized format
//!
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};

pub mod slice;
use crate::slice::{InvalidSlice, SliceIterator, TensorIndexer};

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
}

fn prepare<'hash, 'data>(
    data: &'hash HashMap<String, Tensor<'data>>,
) -> (Metadata, Vec<&'hash Tensor<'data>>, usize) {
    let mut tensors: Vec<&Tensor> = vec![];
    let mut hmetadata = HashMap::new();
    let mut offset = 0;
    for (name, tensor) in data {
        let n = tensor.data.len();
        let tensor_info = TensorInfo {
            dtype: tensor.dtype.clone(),
            shape: tensor.shape.clone(),
            data_offsets: (offset, offset + n),
        };
        offset += n;
        hmetadata.insert(name.to_string(), tensor_info);
        tensors.push(tensor);
    }

    let metadata: Metadata = Metadata(hmetadata);

    (metadata, tensors, offset)
}

/// Serialize to an owned byte buffer the dictionnary of tensors.
pub fn serialize(data: &HashMap<String, Tensor>) -> Vec<u8> {
    let (metadata, tensors, offset) = prepare(data);
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

/// Serialize to an regular file the dictionnary of tensors.
/// Writing directly to file reduces the need to allocate the whole amount to
/// memory.
pub fn serialize_to_file(
    data: &HashMap<String, Tensor>,
    filename: &str,
) -> Result<(), std::io::Error> {
    let (metadata, tensors, _) = prepare(data);
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
    offset: usize,
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
        // TODO validate metadata
        // - Rest of buffer is used
        // - Indidivual tensors have the appropriate declared size
        Ok(Self {
            metadata,
            offset: n + 8,
            data: buffer,
        })
    }

    /// Allow the user to iterate over tensors within the SafeTensors.
    /// The tensors returned are merely views and the data is not owned by this
    /// structure.
    pub fn tensors(&self) -> Vec<(String, TensorView<'_>)> {
        let mut tensors = vec![];
        for (name, info) in &self.metadata.0 {
            let tensorview = TensorView {
                dtype: &info.dtype,
                shape: &info.shape,
                data: &self.data
                    [info.data_offsets.0 + self.offset..info.data_offsets.1 + self.offset],
            };
            tensors.push((name.to_string(), tensorview));
        }
        tensors
    }

    /// Allow the user to get a specific tensor within the SafeTensors.
    /// The tensor returned is merely a view and the data is not owned by this
    /// structure.
    pub fn tensor(&self, tensor_name: &str) -> Result<TensorView<'_>, SafeTensorError> {
        for (name, info) in &self.metadata.0 {
            if name == tensor_name {
                return Ok(TensorView {
                    dtype: &info.dtype,
                    shape: &info.shape,
                    data: &self.data
                        [info.data_offsets.0 + self.offset..info.data_offsets.1 + self.offset],
                });
            }
        }
        Err(SafeTensorError::TensorNotFound)
    }

    /// Allow the user to get a specific tensor within the SafeTensors.
    /// The tensor returned is merely a view and the data is not owned by this
    /// structure.
    pub fn names(&self) -> Vec<&'_ String> {
        self.metadata.0.iter().map(|(name, _)| name).collect()
    }
}

/// The stuct representing the header of safetensor files which allow
/// indexing into the raw byte-buffer array and how to interpret it.
#[derive(Debug, Deserialize, Serialize)]
pub struct Metadata(pub HashMap<String, TensorInfo>);

/// A view of a Tensor within the file.
/// Contains references to data within the full byte-buffer
/// And is thus a readable view of a single tensor
#[derive(Debug)]
pub struct TensorView<'data> {
    dtype: &'data Dtype,
    shape: &'data [usize],
    data: &'data [u8],
}

impl<'data> TensorView<'data> {
    /// Create new tensor view
    pub fn new(dtype: &'data Dtype, shape: &'data [usize], data: &'data [u8]) -> Self {
        Self { dtype, shape, data }
    }
    /// The current tensor dtype
    pub fn get_dtype(&self) -> &'data Dtype {
        self.dtype
    }

    /// The current tensor shape
    pub fn get_shape(&self) -> &'data [usize] {
        self.shape
    }

    /// The current tensor byte-buffer
    pub fn get_data(&self) -> &'data [u8] {
        self.data
    }
}

impl<'data> TensorView<'data> {
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
#[derive(Debug, Deserialize, Serialize, Clone)]
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

/// A struct representing a Tensor, the byte-buffer is not owned
/// but dtype a shape are.
pub struct Tensor<'data> {
    shape: Vec<usize>,
    dtype: Dtype,
    data: &'data [u8],
}

impl<'a> Tensor<'a> {
    /// Simple Tensor creation.
    pub fn new(data: &'a [u8], dtype: Dtype, shape: Vec<usize>) -> Self {
        Self { data, dtype, shape }
    }
}

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
        let attn_0 = Tensor {
            dtype: Dtype::F32,
            shape: vec![1, 2, 3],
            data: &data,
        };
        let metadata: HashMap<String, Tensor> =
            [("attn.0".to_string(), attn_0)].into_iter().collect();

        let out = serialize(&metadata);
        let _parsed = SafeTensors::deserialize(&out).unwrap();
    }

    #[test]
    fn test_slicing() {
        let data: Vec<u8> = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let attn_0 = Tensor {
            dtype: Dtype::F32,
            shape: vec![1, 2, 3],
            data: &data,
        };
        let metadata: HashMap<String, Tensor> =
            [("attn.0".to_string(), attn_0)].into_iter().collect();

        let out = serialize(&metadata);
        let parsed = SafeTensors::deserialize(&out).unwrap();

        let out_buffer: Vec<u8> = parsed
            .tensor("attn.0")
            .unwrap()
            .i((.., ..1))
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
            .i((.., .., ..1))
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
    fn test_gpt2_medium() {
        gpt2_like(24, "gpt2_medium");
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

        let n: usize = tensors_desc
            .iter()
            .map(|item| item.1.iter().product::<usize>())
            .sum::<usize>()
            * 4; // 4 == float32
        let all_data = vec![0; n];
        let mut metadata: HashMap<String, Tensor> = HashMap::new();
        let mut offset = 0;
        for (name, shape) in tensors_desc {
            let n: usize = shape.iter().product();
            let buffer = &all_data[offset..offset + n];
            let tensor = Tensor::new(buffer, Dtype::F32, shape);
            metadata.insert(name, tensor);
            offset += n;
        }

        let filename = format!("./out_{}.bin", model_id);

        let out = serialize(&metadata);

        std::fs::write(&filename, out).unwrap();

        let raw = std::fs::read(&filename).unwrap();
        let _deserialized = SafeTensors::deserialize(&raw).unwrap();
        std::fs::remove_file(&filename).unwrap();
    }
}
