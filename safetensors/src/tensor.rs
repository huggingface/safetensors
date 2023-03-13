//! Module Containing the most important structures
use crate::slice::{InvalidSlice, SliceIterator, TensorIndexer};
use serde::{ser::SerializeMap, Deserialize, Deserializer, Serialize, Serializer};
use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

const MAX_HEADER_SIZE: usize = 100_000_000;

/// Possible errors that could occur while reading
/// A Safetensor file.
#[derive(Debug)]
pub enum SafeTensorError {
    /// The header is an invalid UTF-8 string and cannot be read.
    InvalidHeader,
    /// The header does contain a valid string, but it is not valid JSON.
    InvalidHeaderDeserialization,
    /// The header is large than 100Mo which is considered too large (Might evolve in the future).
    HeaderTooLarge,
    /// The header is smaller than 8 bytes
    HeaderTooSmall,
    /// The header length is invalid
    InvalidHeaderLength,
    /// The tensor name was not found in the archive
    TensorNotFound(String),
    /// Invalid information between shape, dtype and the proposed offsets in the file
    TensorInvalidInfo,
    /// The offsets declared for tensor with name `String` in the header are invalid
    InvalidOffset(String),
    /// IoError
    IoError(std::io::Error),
    /// The follow tensor cannot be created because the buffer size doesn't match shape + dtype
    InvalidTensorView(Dtype, Vec<usize>, usize),
}

impl From<std::io::Error> for SafeTensorError {
    fn from(error: std::io::Error) -> SafeTensorError {
        SafeTensorError::IoError(error)
    }
}

impl std::fmt::Display for SafeTensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for SafeTensorError {}

struct PreparedData {
    n: u64,
    header_bytes: Vec<u8>,
    offset: usize,
}

/// The trait necessary to enable safetensors to serialize a tensor
/// If you have an owned tensor like this:
///
/// ```rust
/// use safetensors::tensor::{View, Dtype};
/// use std::borrow::Cow;
/// struct Tensor{ dtype: MyDtype, shape: Vec<usize>, data: Vec<u8>}
///
/// # type MyDtype = Dtype;
/// impl<'data> View for &'data Tensor{
///    fn dtype(&self) -> Dtype{
///        self.dtype.into()
///    }
///    fn shape(&self) -> &[usize]{
///         &self.shape
///    }
///    fn data(&self) -> Cow<[u8]>{
///        (&self.data).into()
///    }
///    fn data_len(&self) -> usize{
///        self.data.len()
///    }
/// }
/// ```
///
/// For a borrowed tensor:
///
/// ```rust
/// use safetensors::tensor::{View, Dtype};
/// use std::borrow::Cow;
/// struct Tensor<'data>{ dtype: MyDtype, shape: Vec<usize>, data: &'data[u8]}
///
/// # type MyDtype = Dtype;
/// impl<'data> View for Tensor<'data>{
///    fn dtype(&self) -> Dtype{
///        self.dtype.into()
///    }
///    fn shape(&self) -> &[usize]{
///         &self.shape
///    }
///    fn data(&self) -> Cow<[u8]>{
///        self.data.into()
///    }
///    fn data_len(&self) -> usize{
///        self.data.len()
///    }
/// }
/// ```
///
/// Now if you have some unknown buffer that could be on GPU for instance,
/// you can implement the trait to return an owned local buffer containing the data
/// on CPU (needed to write on disk)
/// ```rust
/// use safetensors::tensor::{View, Dtype};
/// use std::borrow::Cow;
///
/// # type MyDtype = Dtype;
/// # type OpaqueGpu = Vec<u8>;
/// struct Tensor{ dtype: MyDtype, shape: Vec<usize>, data: OpaqueGpu }
///
/// impl View for Tensor{
///    fn dtype(&self) -> Dtype{
///        self.dtype.into()
///    }
///    fn shape(&self) -> &[usize]{
///         &self.shape
///    }
///    fn data(&self) -> Cow<[u8]>{
///        // This copies data from GPU to CPU.
///        let data: Vec<u8> = self.data.to_vec();
///        data.into()
///    }
///    fn data_len(&self) -> usize{
///        let n: usize = self.shape.iter().product();
///        let bytes_per_element = self.dtype.size();
///        n * bytes_per_element
///    }
/// }
/// ```
pub trait View {
    /// The `Dtype` of the tensor
    fn dtype(&self) -> Dtype;
    /// The shape of the tensor
    fn shape(&self) -> &[usize];
    /// The data of the tensor
    fn data(&self) -> Cow<[u8]>;
    /// The length of the data, in bytes.
    /// This is necessary as this might be faster to get than `data().len()`
    /// for instance for tensors residing in GPU.
    fn data_len(&self) -> usize;
}

fn prepare<S: AsRef<str> + Ord + std::fmt::Display, V: View, I: IntoIterator<Item = (S, V)>>(
    data: I,
    data_info: &Option<HashMap<String, String>>,
    // ) -> Result<(Metadata, Vec<&'hash TensorView<'data>>, usize), SafeTensorError> {
) -> Result<(PreparedData, Vec<V>), SafeTensorError> {
    // Make sure we're sorting by descending dtype alignment
    // Then by name
    let mut data: Vec<_> = data.into_iter().collect();
    data.sort_by(|(lname, left), (rname, right)| {
        right.dtype().cmp(&left.dtype()).then(lname.cmp(rname))
    });

    let mut tensors: Vec<V> = Vec::with_capacity(data.len());
    let mut hmetadata = Vec::with_capacity(data.len());
    let mut offset = 0;
    let data: Vec<_> = data.into_iter().collect();
    for (name, tensor) in data {
        let n = tensor.data_len();
        let tensor_info = TensorInfo {
            dtype: tensor.dtype(),
            shape: tensor.shape().to_vec(),
            data_offsets: (offset, offset + n),
        };
        offset += n;
        hmetadata.push((name.to_string(), tensor_info));
        tensors.push(tensor);
    }

    let metadata: Metadata = Metadata::new(data_info.clone(), hmetadata)?;
    let mut metadata_buf = serde_json::to_string(&metadata).unwrap().into_bytes();
    // Force alignment
    let extra = metadata_buf.len() % 8;
    metadata_buf.extend(vec![b' '; extra]);

    let n: u64 = metadata_buf.len() as u64;

    Ok((
        PreparedData {
            n,
            header_bytes: metadata_buf,
            offset,
        },
        tensors,
    ))
}

/// Serialize to an owned byte buffer the dictionnary of tensors.
pub fn serialize<
    S: AsRef<str> + Ord + std::fmt::Display,
    V: View,
    I: IntoIterator<Item = (S, V)>,
>(
    data: I,
    data_info: &Option<HashMap<String, String>>,
) -> Result<Vec<u8>, SafeTensorError> {
    let (
        PreparedData {
            n,
            header_bytes,
            offset,
        },
        tensors,
    ) = prepare(data, data_info)?;
    let expected_size = 8 + header_bytes.len() + offset;
    let mut buffer: Vec<u8> = Vec::with_capacity(expected_size);
    buffer.extend(&n.to_le_bytes().to_vec());
    buffer.extend(&header_bytes);
    for tensor in tensors {
        buffer.extend(tensor.data().as_ref());
    }
    Ok(buffer)
}

/// Serialize to a regular file the dictionnary of tensors.
/// Writing directly to file reduces the need to allocate the whole amount to
/// memory.
pub fn serialize_to_file<
    S: AsRef<str> + Ord + std::fmt::Display,
    V: View,
    I: IntoIterator<Item = (S, V)>,
>(
    data: I,
    data_info: &Option<HashMap<String, String>>,
    filename: &Path,
) -> Result<(), SafeTensorError> {
    let (
        PreparedData {
            n, header_bytes, ..
        },
        tensors,
    ) = prepare(data, data_info)?;
    let mut f = BufWriter::new(File::create(filename)?);
    f.write_all(n.to_le_bytes().as_ref())?;
    f.write_all(&header_bytes)?;
    for tensor in tensors {
        f.write_all(tensor.data().as_ref())?;
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
    /// parses the header, and returns the size of the header + the parsed data.
    pub fn read_metadata<'in_data>(
        buffer: &'in_data [u8],
    ) -> Result<(usize, Metadata), SafeTensorError>
    where
        'in_data: 'data,
    {
        let buffer_len = buffer.len();
        if buffer_len < 8 {
            return Err(SafeTensorError::HeaderTooSmall);
        }
        let arr: [u8; 8] = [
            buffer[0], buffer[1], buffer[2], buffer[3], buffer[4], buffer[5], buffer[6], buffer[7],
        ];
        let n: usize = u64::from_le_bytes(arr)
            .try_into()
            .map_err(|_| SafeTensorError::HeaderTooLarge)?;
        if n > MAX_HEADER_SIZE {
            return Err(SafeTensorError::HeaderTooLarge);
        }

        let stop = n
            .checked_add(8)
            .ok_or(SafeTensorError::InvalidHeaderLength)?;
        if stop > buffer_len {
            return Err(SafeTensorError::InvalidHeaderLength);
        }
        let string =
            std::str::from_utf8(&buffer[8..stop]).map_err(|_| SafeTensorError::InvalidHeader)?;
        let metadata: Metadata = serde_json::from_str(string)
            .map_err(|_| SafeTensorError::InvalidHeaderDeserialization)?;
        metadata.validate()?;
        Ok((n, metadata))
    }
    /// Given a byte-buffer representing the whole safetensor file
    /// parses it and returns the Deserialized form (No Tensor allocation).
    ///
    /// ```
    /// use safetensors::SafeTensors;
    /// use memmap2::MmapOptions;
    /// use std::fs::File;
    ///
    /// let filename = "model.safetensors";
    /// # use std::io::Write;
    /// # let serialized = b"<\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
    /// # File::create(filename).unwrap().write(serialized).unwrap();
    /// let file = File::open(filename).unwrap();
    /// let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };
    /// let tensors = SafeTensors::deserialize(&buffer).unwrap();
    /// let tensor = tensors
    ///         .tensor("test")
    ///         .unwrap();
    /// ```
    pub fn deserialize<'in_data>(buffer: &'in_data [u8]) -> Result<Self, SafeTensorError>
    where
        'in_data: 'data,
    {
        let (n, metadata) = SafeTensors::read_metadata(buffer)?;
        let data = &buffer[n + 8..];
        Ok(Self { metadata, data })
    }

    /// Allow the user to iterate over tensors within the SafeTensors.
    /// The tensors returned are merely views and the data is not owned by this
    /// structure.
    pub fn tensors(&self) -> Vec<(String, TensorView<'_>)> {
        let mut tensors = vec![];
        for (name, &index) in &self.metadata.index_map {
            let info = &self.metadata.tensors[index];
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
        if let Some(index) = &self.metadata.index_map.get(tensor_name) {
            if let Some(info) = &self.metadata.tensors.get(**index) {
                Ok(TensorView {
                    dtype: info.dtype,
                    shape: info.shape.clone(),
                    data: &self.data[info.data_offsets.0..info.data_offsets.1],
                })
            } else {
                Err(SafeTensorError::TensorNotFound(tensor_name.to_string()))
            }
        } else {
            Err(SafeTensorError::TensorNotFound(tensor_name.to_string()))
        }
    }

    /// Allow the user to get a specific tensor within the SafeTensors.
    /// The tensor returned is merely a view and the data is not owned by this
    /// structure.
    pub fn names(&self) -> Vec<&'_ String> {
        self.metadata.index_map.keys().collect()
    }
}

/// The stuct representing the header of safetensor files which allow
/// indexing into the raw byte-buffer array and how to interpret it.
#[derive(Debug)]
pub struct Metadata {
    metadata: Option<HashMap<String, String>>,
    tensors: Vec<TensorInfo>,
    index_map: HashMap<String, usize>,
}

/// Helper struct used only for serialization deserialization
#[derive(Serialize, Deserialize)]
struct HashMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "__metadata__")]
    metadata: Option<HashMap<String, String>>,
    #[serde(flatten)]
    tensors: HashMap<String, TensorInfo>,
}

impl<'de> Deserialize<'de> for Metadata {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let hashdata: HashMetadata = HashMetadata::deserialize(deserializer)?;
        let (metadata, tensors) = (hashdata.metadata, hashdata.tensors);
        let mut tensors: Vec<_> = tensors.into_iter().collect();
        // We need to sort by offsets
        // Previous versions might have a different ordering
        // Than we expect (Not aligned ordered, but purely name ordered,
        // or actually any order).
        tensors.sort_by(|(_, left), (_, right)| left.data_offsets.cmp(&right.data_offsets));
        Metadata::new(metadata, tensors).map_err(serde::de::Error::custom)
    }
}

impl Serialize for Metadata {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut names = vec![""; self.index_map.len()];
        for (name, index) in &self.index_map {
            names[*index] = name;
        }

        let tensors: Vec<_> = names.iter().zip(self.tensors.iter()).collect();
        let mut map = serializer.serialize_map(Some(tensors.len()))?;
        if let Some(metadata) = &self.metadata {
            map.serialize_entry("__metadata__", metadata)?;
        }
        for (name, info) in tensors {
            map.serialize_entry(&name, &info)?;
        }
        map.end()
    }
}

impl Metadata {
    fn new(
        metadata: Option<HashMap<String, String>>,
        tensors: Vec<(String, TensorInfo)>,
    ) -> Result<Self, SafeTensorError> {
        let mut index_map = HashMap::new();

        let tensors: Vec<_> = tensors
            .into_iter()
            .enumerate()
            .map(|(index, (k, tensor))| {
                index_map.insert(k, index);
                tensor
            })
            .collect();

        let metadata = Self {
            metadata,
            tensors,
            index_map,
        };
        // metadata.validate()?;
        Ok(metadata)
    }

    fn validate(&self) -> Result<(), SafeTensorError> {
        let mut start = 0;
        for (i, info) in self.tensors.iter().enumerate() {
            let (s, e) = info.data_offsets;
            if s != start || e < s {
                let tensor_name = self
                    .index_map
                    .iter()
                    .filter_map(|(name, &index)| if index == i { Some(&name[..]) } else { None })
                    .next()
                    .unwrap_or("no_tensor");
                return Err(SafeTensorError::InvalidOffset(tensor_name.to_string()));
            }
            start = e;
            let nelements: usize = info.shape.iter().product();
            let nbytes = nelements * info.dtype.size();
            if (e - s) != nbytes {
                return Err(SafeTensorError::TensorInvalidInfo);
            }
        }

        Ok(())
    }

    /// Gives back the tensor metadata
    pub fn tensors(&self) -> HashMap<String, &TensorInfo> {
        self.index_map
            .iter()
            .map(|(tensor_name, index)| (tensor_name.clone(), &self.tensors[*index]))
            .collect()
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
    dtype: Dtype,
    shape: Vec<usize>,
    data: &'data [u8],
}

impl<'data> View for &TensorView<'data> {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<[u8]> {
        self.data.into()
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

impl<'data> TensorView<'data> {
    /// Create new tensor view
    pub fn new(
        dtype: Dtype,
        shape: Vec<usize>,
        data: &'data [u8],
    ) -> Result<Self, SafeTensorError> {
        let n = data.len();
        let n_elements: usize = shape.iter().product();
        if n != n_elements * dtype.size() {
            Err(SafeTensorError::InvalidTensorView(dtype, shape, n))
        } else {
            Ok(Self { dtype, shape, data })
        }
    }
    /// The current tensor dtype
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    /// The current tensor shape
    pub fn shape(&'data self) -> &'data [usize] {
        &self.shape
    }

    /// The current tensor byte-buffer
    pub fn data(&self) -> &'data [u8] {
        self.data
    }

    /// The various pieces of the data buffer according to the asked slice
    pub fn sliced_data(
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

/// The various available dtypes. They MUST be in increasing alignment order
#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq, Ord, PartialOrd)]
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
    /// Half-precision floating point
    F16,
    /// Brain floating point
    BF16,
    /// Signed integer (32-bit)
    I32,
    /// Unsigned integer (32-bit)
    U32,
    /// Floating point (32-bit)
    F32,
    /// Floating point (64-bit)
    F64,
    /// Signed integer (64-bit)
    I64,
    /// Unsigned integer (64-bit)
    U64,
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

        let out = serialize(&metadata, &None).unwrap();
        assert_eq!(
            out,
            [
                64, 0, 0, 0, 0, 0, 0, 0, 123, 34, 97, 116, 116, 110, 46, 48, 34, 58, 123, 34, 100,
                116, 121, 112, 101, 34, 58, 34, 70, 51, 50, 34, 44, 34, 115, 104, 97, 112, 101, 34,
                58, 91, 49, 44, 50, 44, 51, 93, 44, 34, 100, 97, 116, 97, 95, 111, 102, 102, 115,
                101, 116, 115, 34, 58, 91, 48, 44, 50, 52, 93, 125, 125, 0, 0, 0, 0, 0, 0, 128, 63,
                0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64, 0, 0, 160, 64
            ]
        );
        let _parsed = SafeTensors::deserialize(&out).unwrap();
    }

    #[test]
    fn test_serialization_forced_alignement() {
        let data: Vec<u8> = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let shape = vec![1, 2, 3];
        let attn_0 = TensorView::new(Dtype::F32, shape, &data).unwrap();
        let metadata: HashMap<String, TensorView> =
            // Smaller string to force misalignment compared to previous test.
            [("attn0".to_string(), attn_0)].into_iter().collect();

        let out = serialize(&metadata, &None).unwrap();
        assert_eq!(
            out,
            [
                70, 0, 0, 0, 0, 0, 0, 0, 123, 34, 97, 116, 116, 110, 48, 34, 58, 123, 34, 100, 116,
                121, 112, 101, 34, 58, 34, 70, 51, 50, 34, 44, 34, 115, 104, 97, 112, 101, 34, 58,
                91, 49, 44, 50, 44, 51, 93, 44, 34, 100, 97, 116, 97, 95, 111, 102, 102, 115, 101,
                // All the 32 are forcing alignement of the tensor data for casting to f32, f64
                // etc..
                116, 115, 34, 58, 91, 48, 44, 50, 52, 93, 125, 125, 32, 32, 32, 32, 32, 32, 32, 0,
                0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64, 0, 0, 160, 64
            ]
        );
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

        let out = serialize(&metadata, &None).unwrap();
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
            tensors_desc.push((format!("h.{i}.ln_1.weight"), vec![768]));
            tensors_desc.push((format!("h.{i}.ln_1.bias"), vec![768]));
            tensors_desc.push((format!("h.{i}.attn.bias"), vec![1, 1, 1024, 1024]));
            tensors_desc.push((format!("h.{i}.attn.c_attn.weight"), vec![768, 2304]));
            tensors_desc.push((format!("h.{i}.attn.c_attn.bias"), vec![2304]));
            tensors_desc.push((format!("h.{i}.attn.c_proj.weight"), vec![768, 768]));
            tensors_desc.push((format!("h.{i}.attn.c_proj.bias"), vec![768]));
            tensors_desc.push((format!("h.{i}.ln_2.weight"), vec![768]));
            tensors_desc.push((format!("h.{i}.ln_2.bias"), vec![768]));
            tensors_desc.push((format!("h.{i}.mlp.c_fc.weight"), vec![768, 3072]));
            tensors_desc.push((format!("h.{i}.mlp.c_fc.bias"), vec![3072]));
            tensors_desc.push((format!("h.{i}.mlp.c_proj.weight"), vec![3072, 768]));
            tensors_desc.push((format!("h.{i}.mlp.c_proj.bias"), vec![768]));
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

        let filename = format!("./out_{model_id}.bin");

        let out = serialize(&metadata, &None).unwrap();

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
        assert_eq!(tensor.shape(), vec![2, 2]);
        assert_eq!(tensor.dtype(), Dtype::I32);
        // 16 bytes
        assert_eq!(tensor.data(), b"\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0");
    }

    #[test]
    fn test_json_attack() {
        let mut tensors = HashMap::new();
        let dtype = Dtype::F32;
        let shape = vec![2, 2];
        let data_offsets = (0, 16);
        for i in 0..10 {
            tensors.insert(
                format!("weight_{i}"),
                TensorInfo {
                    dtype,
                    shape: shape.clone(),
                    data_offsets,
                },
            );
        }

        let metadata = HashMetadata {
            metadata: None,
            tensors,
        };
        let serialized = serde_json::to_string(&metadata).unwrap();
        let serialized = serialized.as_bytes();

        let n = serialized.len();

        let filename = "out.safetensors";
        let mut f = BufWriter::new(File::create(filename).unwrap());
        f.write_all(n.to_le_bytes().as_ref()).unwrap();
        f.write_all(serialized).unwrap();
        f.write_all(b"\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0").unwrap();
        f.flush().unwrap();

        let reloaded = std::fs::read(filename).unwrap();
        match SafeTensors::deserialize(&reloaded) {
            Err(SafeTensorError::InvalidOffset(_)) => {
                // Yes we have the correct error, name of the tensor is random though
            }
            Err(err) => panic!("Unexpected error {err:?}"),
            Ok(_) => panic!("This should not be able to be deserialized"),
        }
    }

    #[test]
    fn test_header_too_large() {
        let serialized = b"<\x00\x00\x00\x00\xff\xff\xff{\"test\":{\"dtype\":\"I32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";

        match SafeTensors::deserialize(serialized) {
            Err(SafeTensorError::HeaderTooLarge) => {
                // Yes we have the correct error
            }
            _ => panic!("This should not be able to be deserialized"),
        }
    }
}
