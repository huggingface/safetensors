//! Module Containing the most important structures
use crate::lib::{Cow, HashMap, String, ToString, Vec};
use crate::slice::{InvalidSlice, SliceIterator, TensorIndexer};
use serde::{ser::SerializeMap, Deserialize, Deserializer, Serialize, Serializer};
#[cfg(feature = "std")]
use std::io::Write;

const MAX_HEADER_SIZE: usize = 100_000_000;

/// Possible errors that could occur while reading
/// A Safetensor file.
#[derive(Debug)]
pub enum SafeTensorError {
    /// The header is an invalid UTF-8 string and cannot be read.
    InvalidHeader,
    /// The header's first byte is not the expected `{`.
    InvalidHeaderStart,
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
    #[cfg(feature = "std")]
    IoError(std::io::Error),
    /// JSON error
    JsonError(serde_json::Error),
    /// The follow tensor cannot be created because the buffer size doesn't match shape + dtype
    InvalidTensorView(Dtype, Vec<usize>, usize),
    /// The metadata is invalid because the data offsets of the tensor does not
    /// fully cover the buffer part of the file. The last offset **must** be
    /// the end of the file.
    MetadataIncompleteBuffer,
    /// The metadata contains information (shape or shape * dtype size) which lead to an
    /// arithmetic overflow. This is most likely an error in the file.
    ValidationOverflow,
}

#[cfg(feature = "std")]
impl From<std::io::Error> for SafeTensorError {
    fn from(error: std::io::Error) -> SafeTensorError {
        SafeTensorError::IoError(error)
    }
}

impl From<serde_json::Error> for SafeTensorError {
    fn from(error: serde_json::Error) -> SafeTensorError {
        SafeTensorError::JsonError(error)
    }
}

impl core::fmt::Display for SafeTensorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(not(feature = "std"))]
impl core::error::Error for SafeTensorError {}

#[cfg(feature = "std")]
impl std::error::Error for SafeTensorError {}

struct PreparedData {
    n: u64,
    header_bytes: Vec<u8>,
    offset: usize,
}

/// Packs two u4 values (represented as u8, must be <= 15) into a single u8.
/// Low nibble value goes into the lower 4 bits, high nibble value into the upper 4 bits.
#[inline]
fn pack_u4(low: u8, high: u8) -> u8 {
    let high_nibble_area_mask: u8 = 0b11110000;
    let low_nibble_area_mask: u8 = 0b00001111;
    let value_mask: u8 = 0b00001111;

    let high_part = ((high & value_mask) << 4) & high_nibble_area_mask;
    let low_part = (low & value_mask) & low_nibble_area_mask;

    high_part | low_part
}

/// Unpacks a single u8 into two u4 values (represented as u8).
/// Returns a tuple (low_nibble_value, high_nibble_value).
#[inline]
fn unpack_u4(packed: u8) -> (u8, u8) {
    let low_nibble_mask: u8 = 0b00001111;

    let low_value = packed & low_nibble_mask;
    let high_value = (packed >> 4) & low_nibble_mask;

    (low_value, high_value)
}

/// Packs two i4 values (represented as i8, must be between -8 and 7) into a single u8.
/// Low nibble value goes into the lower 4 bits, high nibble value into the upper 4 bits.
#[inline]
fn pack_i4(low: i8, high: i8) -> u8 {
    let high_nibble_area_mask: u8 = 0b11110000;
    let low_nibble_area_mask: u8 = 0b00001111;
    let value_mask: u8 = 0b00001111;

    let low_bits = (low as u8) & value_mask;
    let high_bits = (high as u8) & value_mask;

    let high_part = (high_bits << 4) & high_nibble_area_mask;
    let low_part = low_bits & low_nibble_area_mask;

    high_part | low_part
}

/// Unpacks a single u8 into two i4 values (represented as i8).
/// Interprets the nibbles as 4-bit two's complement integers.
/// Returns a tuple (low_nibble_value, high_nibble_value).
#[inline]
fn unpack_i4(packed: u8) -> (i8, i8) {
    let nibble_mask: u8 = 0b00001111;
    let sign_bit_mask: u8 = 0b00001000;
    let sign_extend_mask: u8 = 0b11110000; // Mask to OR for negative sign extension

    let low_bits = packed & nibble_mask;
    let high_bits = (packed >> 4) & nibble_mask;

    let low_signed = if (low_bits & sign_bit_mask) != 0 {
        (low_bits | sign_extend_mask) as i8
    } else {
        low_bits as i8
    };

    let high_signed = if (high_bits & sign_bit_mask) != 0 {
        (high_bits | sign_extend_mask) as i8
    } else {
        high_bits as i8
    };

    (low_signed, high_signed)
}

/// Computes the number of rows and columns for a given shape.
/// Returns a tuple (rows, cols).
fn compute_rows_cols(shape: &[usize]) -> (usize, usize) {
    let dims = shape.len();
    match dims {
        0 => (0, 0),
        1 => (1, shape[0]),
        _ => {
            let cols = shape[dims - 1];
            let rows = shape.iter().take(dims - 1).product();
            (rows, cols)
        }
    }
}

/// Serializes a tensor of type INT4 into a byte buffer.
/// Uses padding to ensure shape is a multiple of 2.
pub fn serialize_i4(shape: &[usize], data: &[i8]) -> Vec<u8> {
    let (rows, cols) = compute_rows_cols(shape);
    let mut buffer: Vec<u8> = Vec::with_capacity(rows * (cols / 2 + cols % 2));
    for row in 0..rows {
        for col in (0..cols).step_by(2) {
            let idx = row * cols + col;
            let a = data[idx];
            let b = if col + 1 < cols { data[idx + 1] } else { 0 };
            let packed = pack_i4(b, a);
            buffer.push(packed);
        }
    }

    buffer
}

/// Deserializes a byte buffer into a tensor of type INT4.
/// Assumes the buffer is correctly formatted.
pub fn deserialize_i4(shape: &[usize], packed: &[u8]) -> Vec<i8> {
    let (rows, cols) = compute_rows_cols(shape);
    let count = rows * cols;
    let mut buffer: Vec<i8> = Vec::with_capacity(count);

    for idx in 0..count {
        let row = idx / cols;
        let raw_idx = idx + row * cols % 2;

        let byte = packed[raw_idx / 2];
        let (low, high) = unpack_i4(byte);

        let val = if raw_idx % 2 == 0 { high } else { low };
        buffer.push(val);
    }

    buffer
}

/// Serializes a tensor of type UINT4 into a byte buffer.
/// Assumes the tensor is correctly formatted.
pub fn serialize_u4(shape: &[usize], data: &[u8]) -> Vec<u8> {
    let (rows, cols) = compute_rows_cols(shape);
    let mut buffer: Vec<u8> = Vec::with_capacity(rows * (cols / 2 + cols % 2));
    for row in 0..rows {
        for col in (0..cols).step_by(2) {
            let idx = row * cols + col;
            let a = data[idx];
            let b = if col + 1 < cols { data[idx + 1] } else { 0 };
            let packed = pack_u4(b, a);
            buffer.push(packed);
        }
    }

    buffer
}

/// Deserializes a tensor of type UINT4 from a byte buffer.
/// Assumes the buffer is correctly formatted.
pub fn deserialize_u4(shape: &[usize], packed: &[u8]) -> Vec<u8> {
    let (rows, cols) = compute_rows_cols(shape);
    let count = rows * cols;
    let mut buffer: Vec<u8> = Vec::with_capacity(count);

    for idx in 0..count {
        let row = idx / cols;
        let raw_idx = idx + row * cols % 2;

        let byte = packed[raw_idx / 2];
        let (low, high) = unpack_u4(byte);

        let val = if raw_idx % 2 == 0 { high } else { low };
        buffer.push(val);
    }

    buffer
}

/// Computes the size in bytes of a tensor with the given data type and shape.
/// This function takes into account specific details about storing INT4 tensors.
pub fn compute_size(dtype: Dtype, shape: &[usize]) -> usize {
    let n_elements: usize = shape.iter().product();
    match dtype {
        Dtype::PackedI4 | Dtype::PackedU4 => {
            let (rows, cols) = compute_rows_cols(&shape);

            let mut bits_per_row = cols * dtype.size_in_bits();
            if cols % 2 != 0 {
                bits_per_row += dtype.size_in_bits()
            }

            let bytes_per_row = bits_per_row / 8;
            bytes_per_row * rows
        }
        _ => n_elements * dtype.size_in_bits() / 8,
    }
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
///        let bytes_per_element = self.dtype.size_in_bits() / 8;
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

fn prepare<S: AsRef<str> + Ord + core::fmt::Display, V: View, I: IntoIterator<Item = (S, V)>>(
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
    let mut metadata_buf = serde_json::to_string(&metadata)?.into_bytes();
    // Force alignment to 8 bytes.
    let extra = (8 - metadata_buf.len() % 8) % 8;
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
    S: AsRef<str> + Ord + core::fmt::Display,
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
#[cfg(feature = "std")]
pub fn serialize_to_file<
    S: AsRef<str> + Ord + core::fmt::Display,
    V: View,
    I: IntoIterator<Item = (S, V)>,
>(
    data: I,
    data_info: &Option<HashMap<String, String>>,
    filename: &std::path::Path,
) -> Result<(), SafeTensorError> {
    let (
        PreparedData {
            n, header_bytes, ..
        },
        tensors,
    ) = prepare(data, data_info)?;
    let mut f = std::io::BufWriter::new(std::fs::File::create(filename)?);
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
#[derive(Debug)]
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
            core::str::from_utf8(&buffer[8..stop]).map_err(|_| SafeTensorError::InvalidHeader)?;
        // Assert the string starts with {
        // NOTE: Add when we move to 0.4.0
        // if !string.starts_with('{') {
        //     return Err(SafeTensorError::InvalidHeaderStart);
        // }
        let metadata: Metadata = serde_json::from_str(string)
            .map_err(|_| SafeTensorError::InvalidHeaderDeserialization)?;
        let buffer_end = metadata.validate()?;
        if buffer_end + 8 + n != buffer_len {
            return Err(SafeTensorError::MetadataIncompleteBuffer);
        }
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

    /// Returns the tensors contained within the SafeTensors.
    /// The tensors returned are merely views and the data is not owned by this
    /// structure.
    pub fn tensors(&self) -> Vec<(String, TensorView<'data>)> {
        let mut tensors = Vec::with_capacity(self.metadata.index_map.len());
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

    /// Returns an iterator over the tensors contained within the SafeTensors.
    /// The tensors returned are merely views and the data is not owned by this
    /// structure.
    pub fn iter<'a>(&'a self) -> impl Iterator<Item = (&'a str, TensorView<'data>)> {
        self.metadata.index_map.iter().map(|(name, &idx)| {
            let info = &self.metadata.tensors[idx];
            (
                name.as_str(),
                TensorView {
                    dtype: info.dtype,
                    shape: info.shape.clone(),
                    data: &self.data[info.data_offsets.0..info.data_offsets.1],
                },
            )
        })
    }

    /// Allow the user to get a specific tensor within the SafeTensors.
    /// The tensor returned is merely a view and the data is not owned by this
    /// structure.
    pub fn tensor(&self, tensor_name: &str) -> Result<TensorView<'data>, SafeTensorError> {
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

    /// Return the names of the tensors within the SafeTensors.
    /// These are used as keys to access to the actual tensors, that can be
    /// retrieved using the tensor method.
    pub fn names(&self) -> Vec<&'_ String> {
        self.metadata.index_map.keys().collect()
    }

    /// Return how many tensors are currently stored within the SafeTensors.
    #[inline]
    pub fn len(&self) -> usize {
        self.metadata.tensors.len()
    }

    /// Indicate if the SafeTensors contains or not any tensor.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.metadata.tensors.is_empty()
    }
}

/// The stuct representing the header of safetensor files which allow
/// indexing into the raw byte-buffer array and how to interpret it.
#[derive(Debug, Clone)]
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
        let length = if let Some(metadata) = &self.metadata {
            metadata.len()
        } else {
            0
        };
        let mut map = serializer.serialize_map(Some(tensors.len() + length))?;
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
        let mut index_map = HashMap::with_capacity(tensors.len());

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

    fn validate(&self) -> Result<usize, SafeTensorError> {
        let mut start = 0;
        for (i, info) in self.tensors.iter().enumerate() {
            let (s, e) = info.data_offsets;
            if s != start || e < s {
                let tensor_name = self
                    .index_map
                    .iter()
                    .find_map(|(name, &index)| if index == i { Some(&name[..]) } else { None })
                    .unwrap_or("no_tensor");
                return Err(SafeTensorError::InvalidOffset(tensor_name.to_string()));
            }
            start = e;
            let nelements: usize = info
                .shape
                .iter()
                .cloned()
                .try_fold(1usize, usize::checked_mul)
                .ok_or(SafeTensorError::ValidationOverflow)?;
            let nbytes = match info.dtype {
                Dtype::PackedI4 | Dtype::PackedU4 => {
                    let (rows, cols) = compute_rows_cols(&info.shape);

                    let mut bits_per_row = cols * info.dtype.size_in_bits();
                    if cols % 2 != 0 {
                        bits_per_row += info.dtype.size_in_bits()
                    }

                    let bytes_per_row = bits_per_row / 8;

                    rows.checked_mul(bytes_per_row)
                        .ok_or(SafeTensorError::ValidationOverflow)?
                }
                _ => nelements
                    .checked_mul(info.dtype.size_in_bits() / 8)
                    .ok_or(SafeTensorError::ValidationOverflow)?,
            };
            if (e - s) != nbytes {
                return Err(SafeTensorError::TensorInvalidInfo);
            }
        }
        Ok(start)
    }

    /// Gives back the tensor metadata
    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        let index = self.index_map.get(name)?;
        self.tensors.get(*index)
    }

    /// Gives back the tensor metadata
    pub fn tensors(&self) -> HashMap<String, &TensorInfo> {
        self.index_map
            .iter()
            .map(|(tensor_name, index)| (tensor_name.clone(), &self.tensors[*index]))
            .collect()
    }

    /// Gives back the tensor names ordered by offset
    pub fn offset_keys(&self) -> Vec<String> {
        let mut index_vec: Vec<_> = self.index_map.iter().collect();
        index_vec.sort_by_key(|a| a.1);
        index_vec.into_iter().map(|a| a.0.clone()).collect()
    }

    /// Gives back the tensor metadata
    pub fn metadata(&self) -> &Option<HashMap<String, String>> {
        &self.metadata
    }
}

/// A view of a Tensor within the file.
/// Contains references to data within the full byte-buffer
/// And is thus a readable view of a single tensor
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct TensorView<'data> {
    dtype: Dtype,
    shape: Vec<usize>,
    data: &'data [u8],
}

impl View for &TensorView<'_> {
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
        compute_size(self.dtype, &self.shape)
    }
}

impl View for TensorView<'_> {
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
        compute_size(self.dtype, &self.shape)
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
        let size_in_bytes = compute_size(dtype, &shape);
        if dtype != Dtype::PackedI4 && dtype != Dtype::PackedU4 && n != size_in_bytes {
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
        slices: &[TensorIndexer],
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
    /// FP8 <https://arxiv.org/pdf/2209.05433.pdf>_
    #[allow(non_camel_case_types)]
    F8_E5M2,
    /// FP8 <https://arxiv.org/pdf/2209.05433.pdf>_
    #[allow(non_camel_case_types)]
    F8_E4M3,
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
    /// Packed unsigned integer (4-bit)
    PackedU4,
    /// Packed signed integer (4-bit)
    PackedI4,
}

impl Dtype {
    /// Gives out the size (in bits) of 1 element of this dtype.
    pub fn size_in_bits(&self) -> usize {
        match self {
            Dtype::BOOL => 8,
            Dtype::PackedU4 => 4,
            Dtype::PackedI4 => 4,
            Dtype::U8 => 8,
            Dtype::I8 => 8,
            Dtype::F8_E5M2 => 8,
            Dtype::F8_E4M3 => 8,
            Dtype::I16 => 16,
            Dtype::U16 => 16,
            Dtype::I32 => 32,
            Dtype::U32 => 32,
            Dtype::I64 => 64,
            Dtype::U64 => 64,
            Dtype::F16 => 16,
            Dtype::BF16 => 16,
            Dtype::F32 => 32,
            Dtype::F64 => 64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::slice::IndexOp;
    use proptest::prelude::*;
    #[cfg(not(feature = "std"))]
    extern crate std;
    use std::io::Write;

    const MAX_DIMENSION: usize = 8;
    const MAX_SIZE: usize = 8;
    const MAX_TENSORS: usize = 8;

    fn arbitrary_dtype() -> impl Strategy<Value = Dtype> {
        prop_oneof![
            Just(Dtype::BOOL),
            Just(Dtype::U8),
            Just(Dtype::I8),
            Just(Dtype::I16),
            Just(Dtype::U16),
            Just(Dtype::I32),
            Just(Dtype::U32),
            Just(Dtype::I64),
            Just(Dtype::U64),
            Just(Dtype::F16),
            Just(Dtype::BF16),
            Just(Dtype::F32),
            Just(Dtype::F64),
        ]
    }

    fn arbitrary_shape() -> impl Strategy<Value = Vec<usize>> {
        // We do not allow empty shapes or 0 sizes.
        (1..MAX_DIMENSION).prop_flat_map(|length| prop::collection::vec(1..MAX_SIZE, length))
    }

    fn arbitrary_metadata() -> impl Strategy<Value = Metadata> {
        // We generate at least one tensor.
        (1..MAX_TENSORS)
            .prop_flat_map(|size| {
                // Returns a strategy generating `size` data types and shapes.
                (
                    prop::collection::vec(arbitrary_dtype(), size),
                    prop::collection::vec(arbitrary_shape(), size),
                )
            })
            .prop_map(|(dtypes, shapes)| {
                // Returns a valid metadata object for a random (length, dtypes, shapes) triple.
                let mut start = 0;
                let tensors: Vec<TensorInfo> = dtypes
                    .iter()
                    .zip(shapes)
                    .map(|(dtype, shape)| {
                        // This cannot overflow because the size of
                        // the vector and elements are so small.
                        let length: usize = shape.iter().product();
                        let end = start + length * dtype.size_in_bits() / 8;
                        let tensor = TensorInfo {
                            dtype: *dtype,
                            shape,
                            data_offsets: (start, end),
                        };
                        start = end;
                        tensor
                    })
                    .collect();
                let index_map = (0..tensors.len())
                    .map(|index| (format!("t.{index}"), index))
                    .collect();
                Metadata {
                    metadata: None,
                    tensors,
                    index_map,
                }
            })
    }

    /// This method returns the size of the data corresponding to the metadata. It
    /// assumes that `metadata` contains at least one tensor, and that tensors are
    /// ordered by offset in `metadata.tensors`.
    ///
    /// # Panics
    ///
    /// This method will panic if `metadata` does not contain any tensors.
    fn data_size(metadata: &Metadata) -> usize {
        metadata.tensors.last().unwrap().data_offsets.1
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]

        #[test]
        fn test_indexing(metadata in arbitrary_metadata()) {
            let data = vec![0u8; data_size(&metadata)];
            let tensors = SafeTensors { metadata, data: &data };
            for name in tensors.names() {
                assert!(tensors.tensor(name).is_ok());
            }
        }

        #[test]
        fn test_roundtrip(metadata in arbitrary_metadata()) {
            let data: Vec<u8> = (0..data_size(&metadata)).map(|x| x as u8).collect();
            let before = SafeTensors { metadata, data: &data };
            let tensors = before.tensors();
            let bytes = serialize(tensors.iter().map(|(name, view)| (name.to_string(), view)), &None).unwrap();

            let after = SafeTensors::deserialize(&bytes).unwrap();

            // Check that the tensors are the same after deserialization.
            assert_eq!(before.names().len(), after.names().len());
            for name in before.names() {
                let tensor_before = before.tensor(name).unwrap();
                let tensor_after = after.tensor(name).unwrap();
                assert_eq!(tensor_after.data().as_ptr() as usize % (tensor_after.dtype().size_in_bits() / 8), 0);
                assert_eq!(tensor_before, tensor_after);
            }
        }
    }

    #[test]
    fn test_packing_unpacking_u4() {
        let low: u8 = 5; // 0101
        let high: u8 = 12; // 1100

        let packed_u4 = pack_u4(low, high); // Expected: 1100_0101 = 197
        assert_eq!(packed_u4, 197);
        let (unpacked_low, unpacked_high) = unpack_u4(packed_u4);

        assert_eq!(low, unpacked_low);
        assert_eq!(high, unpacked_high);
    }

    #[test]
    fn test_packing_unpacking_i4_positive() {
        let low: i8 = 3; // 0011
        let high: i8 = 7; // 0111

        let packed_i4 = pack_i4(low, high); // Expected: 0111_0011 = 115
        assert_eq!(packed_i4, 115);
        let (unpacked_low, unpacked_high) = unpack_i4(packed_i4);

        assert_eq!(low, unpacked_low);
        assert_eq!(high, unpacked_high);
    }

    #[test]
    fn test_packing_unpacking_i4_negative() {
        let low: i8 = -1; // 1111
        let high: i8 = -8; // 1000

        let packed_i4 = pack_i4(low, high); // Expected: 1000_1111 = 143
        assert_eq!(packed_i4, 143);
        let (unpacked_low, unpacked_high) = unpack_i4(packed_i4);

        assert_eq!(low, unpacked_low);
        assert_eq!(high, unpacked_high);
    }

    #[test]
    fn test_packing_unpacking_i4_mixed() {
        let low: i8 = 7; // 0111
        let high: i8 = -8; // 1000

        let packed_i4 = pack_i4(low, high); // Expected: 1000_0111 = 135
        assert_eq!(packed_i4, 135);
        let (unpacked_low, unpacked_high) = unpack_i4(packed_i4);

        assert_eq!(low, unpacked_low);
        assert_eq!(high, unpacked_high);
    }

    #[test]
    fn test_roundtrip_1d_i4() {
        let shape = vec![6];
        let data: Vec<i8> = vec![3, 7, -1, -8, 2, -5];

        let packed = serialize_i4(&shape, &data);
        assert_eq!(3, packed.len());
        assert_eq!(vec![55u8, 248, 43], packed);

        let unpacked = deserialize_i4(&shape, &packed);
        assert_eq!(data, unpacked);
    }

    #[test]
    fn test_roundtrip_2d_i4() {
        let shape = vec![2, 3];
        let data: Vec<i8> = vec![3, 7, -1, -8, 2, -5];

        let packed = serialize_i4(&shape, &data);
        assert_eq!(4, packed.len());
        assert_eq!(vec![55u8, 240, 130, 176], packed);

        let unpacked = deserialize_i4(&shape, &packed);
        assert_eq!(data, unpacked);
    }

    #[test]
    fn test_roundtrip_3d_i4() {
        let shape = vec![2, 2, 2];
        let data: Vec<i8> = vec![1, -2, 3, -4, 5, -6, 7, -8];

        let packed = serialize_i4(&shape, &data);
        assert_eq!(4, packed.len());
        assert_eq!(vec![30u8, 60, 90, 120], packed);

        let unpacked = deserialize_i4(&shape, &packed);
        assert_eq!(data, unpacked);
    }

    #[test]
    fn test_roundtrip_1d_u4() {
        let shape = vec![6];
        let data: Vec<u8> = vec![13, 7, 1, 8, 2, 15];

        let packed = serialize_u4(&shape, &data);
        assert_eq!(3, packed.len());
        assert_eq!(vec![215u8, 24, 47], packed);

        let unpacked = deserialize_u4(&shape, &packed);
        assert_eq!(data, unpacked);
    }

    #[test]
    fn test_roundtrip_2d_u4() {
        let shape = vec![2, 3];
        let data: Vec<u8> = vec![3, 15, 1, 8, 2, 12];

        let packed = serialize_u4(&shape, &data);
        assert_eq!(4, packed.len());
        assert_eq!(vec![63u8, 16, 130, 192], packed);

        let unpacked = deserialize_u4(&shape, &packed);
        assert_eq!(data, unpacked);
    }

    #[test]
    fn test_roundtrip_3d_u4() {
        let shape = vec![2, 2, 2];
        let data: Vec<u8> = vec![3, 15, 1, 8, 2, 12, 1, 7];

        let packed = serialize_u4(&shape, &data);
        assert_eq!(4, packed.len());
        assert_eq!(vec![63u8, 24, 44, 23], packed);

        let unpacked = deserialize_u4(&shape, &packed);
        assert_eq!(data, unpacked);
    }

    #[test]
    fn test_tensor_data_len_3d() {
        let shape = vec![1, 2, 3];
        let data: Vec<u8> = vec![3, 15, 1, 8, 2, 12];

        let attn_0 = TensorView::new(Dtype::PackedU4, shape, &data).unwrap();
        let len = attn_0.data_len();
        assert_eq!(4, len);
    }

    #[test]
    fn test_packed_u4_roundtrip_3d_tensor() {
        let shape = vec![1, 2, 3];
        let data: Vec<u8> = vec![3, 15, 1, 8, 2, 12]; //63u8, 16, 130, 192
        let packed: Vec<u8> = serialize_u4(&shape, &data);

        let attn_0 = TensorView::new(Dtype::PackedU4, shape, &packed).unwrap();
        let metadata: HashMap<String, TensorView> =
            [("attn.0".to_string(), attn_0)].into_iter().collect();

        let out = serialize(&metadata, &None).unwrap();
        println!("{:?}", out);

        let parsed = SafeTensors::deserialize(&out).unwrap();
        println!("{:?}", parsed);
    }

    #[test]
    fn test_packed_i4_roundtrip_3d_tensor() {
        let shape = vec![1, 2, 3];
        let data: Vec<i8> = vec![3, -2, 1, 7, -1, -7];
        let packed: Vec<u8> = serialize_i4(&shape, &data);

        let attn_0 = TensorView::new(Dtype::PackedI4, shape, &packed).unwrap();
        let metadata: HashMap<String, TensorView> =
            [("attn.0".to_string(), attn_0)].into_iter().collect();

        let out = serialize(&metadata, &None).unwrap();
        println!("{:?}", out);

        let parsed = SafeTensors::deserialize(&out).unwrap();
        println!("{:?}", parsed);
    }

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
    fn test_empty() {
        let tensors: HashMap<String, TensorView> = HashMap::new();

        let out = serialize(&tensors, &None).unwrap();
        assert_eq!(
            out,
            [8, 0, 0, 0, 0, 0, 0, 0, 123, 125, 32, 32, 32, 32, 32, 32]
        );
        let _parsed = SafeTensors::deserialize(&out).unwrap();

        let metadata: Option<HashMap<String, String>> = Some(
            [("framework".to_string(), "pt".to_string())]
                .into_iter()
                .collect(),
        );
        let out = serialize(&tensors, &metadata).unwrap();
        assert_eq!(
            out,
            [
                40, 0, 0, 0, 0, 0, 0, 0, 123, 34, 95, 95, 109, 101, 116, 97, 100, 97, 116, 97, 95,
                95, 34, 58, 123, 34, 102, 114, 97, 109, 101, 119, 111, 114, 107, 34, 58, 34, 112,
                116, 34, 125, 125, 32, 32, 32, 32, 32
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
        let shape = vec![1, 1, 2, 3];
        let attn_0 = TensorView::new(Dtype::F32, shape, &data).unwrap();
        let metadata: HashMap<String, TensorView> =
            // Smaller string to force misalignment compared to previous test.
            [("attn0".to_string(), attn_0)].into_iter().collect();

        let out = serialize(&metadata, &None).unwrap();
        assert_eq!(
            out,
            [
                72, 0, 0, 0, 0, 0, 0, 0, 123, 34, 97, 116, 116, 110, 48, 34, 58, 123, 34, 100, 116,
                121, 112, 101, 34, 58, 34, 70, 51, 50, 34, 44, 34, 115, 104, 97, 112, 101, 34, 58,
                91, 49, 44, 49, 44, 50, 44, 51, 93, 44, 34, 100, 97, 116, 97, 95, 111, 102, 102,
                // All the 32 are forcing alignement of the tensor data for casting to f32, f64
                // etc..
                115, 101, 116, 115, 34, 58, 91, 48, 44, 50, 52, 93, 125, 125, 32, 32, 32, 32, 32,
                32, 32, 0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64, 0, 0,
                160, 64
            ],
        );
        let parsed = SafeTensors::deserialize(&out).unwrap();
        let tensor = parsed.tensor("attn0").unwrap();
        assert_eq!(
            tensor.data().as_ptr() as usize % (tensor.dtype().size_in_bits() / 8),
            0
        );
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
            * dtype.size_in_bits()
            / 8; // 4
        let all_data = vec![0; n];
        let mut metadata = HashMap::with_capacity(tensors_desc.len());
        let mut offset = 0;
        for (name, shape) in tensors_desc {
            let n: usize = shape.iter().product();
            let buffer = &all_data[offset..offset + n * dtype.size_in_bits() / 8];
            let tensor = TensorView::new(dtype, shape, buffer).unwrap();
            metadata.insert(name, tensor);
            offset += n;
        }

        let filename = format!("./out_{model_id}.safetensors");

        let out = serialize(&metadata, &None).unwrap();
        std::fs::write(&filename, out).unwrap();
        let raw = std::fs::read(&filename).unwrap();
        let _deserialized = SafeTensors::deserialize(&raw).unwrap();
        std::fs::remove_file(&filename).unwrap();

        // File api
        #[cfg(feature = "std")]
        {
            serialize_to_file(&metadata, &None, std::path::Path::new(&filename)).unwrap();
            let raw = std::fs::read(&filename).unwrap();
            let _deserialized = SafeTensors::deserialize(&raw).unwrap();
            std::fs::remove_file(&filename).unwrap();
        }
    }

    #[test]
    fn test_empty_shapes_allowed() {
        let serialized = b"8\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[],\"data_offsets\":[0,4]}}\x00\x00\x00\x00";

        let loaded = SafeTensors::deserialize(serialized).unwrap();
        assert_eq!(loaded.names(), vec!["test"]);
        let tensor = loaded.tensor("test").unwrap();
        assert!(tensor.shape().is_empty());
        assert_eq!(tensor.dtype(), Dtype::I32);
        // 4 bytes
        assert_eq!(tensor.data(), b"\0\0\0\0");
    }

    #[test]
    fn test_deserialization() {
        let serialized = b"<\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";

        let loaded = SafeTensors::deserialize(serialized).unwrap();

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.names(), vec!["test"]);
        let tensor = loaded.tensor("test").unwrap();
        assert_eq!(tensor.shape(), vec![2, 2]);
        assert_eq!(tensor.dtype(), Dtype::I32);
        // 16 bytes
        assert_eq!(tensor.data(), b"\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0");
    }

    #[test]
    fn test_lifetimes() {
        let serialized = b"<\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";

        let tensor = {
            let loaded = SafeTensors::deserialize(serialized).unwrap();
            loaded.tensor("test").unwrap()
        };

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
        let mut f = std::io::BufWriter::new(std::fs::File::create(filename).unwrap());
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
    fn test_metadata_incomplete_buffer() {
        let serialized = b"<\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00extra_bogus_data_for_polyglot_file";

        match SafeTensors::deserialize(serialized) {
            Err(SafeTensorError::MetadataIncompleteBuffer) => {
                // Yes we have the correct error
            }
            _ => panic!("This should not be able to be deserialized"),
        }

        // Missing data in the buffer
        let serialized = b"<\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"; // <--- missing 2 bytes

        match SafeTensors::deserialize(serialized) {
            Err(SafeTensorError::MetadataIncompleteBuffer) => {
                // Yes we have the correct error
            }
            _ => panic!("This should not be able to be deserialized"),
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

    #[test]
    fn test_header_too_small() {
        let serialized = b"";
        match SafeTensors::deserialize(serialized) {
            Err(SafeTensorError::HeaderTooSmall) => {
                // Yes we have the correct error
            }
            _ => panic!("This should not be able to be deserialized"),
        }
    }

    #[test]
    fn test_invalid_header_length() {
        let serialized = b"<\x00\x00\x00\x00\x00\x00\x00";
        match SafeTensors::deserialize(serialized) {
            Err(SafeTensorError::InvalidHeaderLength) => {
                // Yes we have the correct error
            }
            _ => panic!("This should not be able to be deserialized"),
        }
    }

    #[test]
    fn test_invalid_header_non_utf8() {
        let serialized = b"\x01\x00\x00\x00\x00\x00\x00\x00\xff";
        match SafeTensors::deserialize(serialized) {
            Err(SafeTensorError::InvalidHeader) => {
                // Yes we have the correct error
            }
            _ => panic!("This should not be able to be deserialized"),
        }
    }

    #[test]
    fn test_invalid_header_not_json() {
        let serialized = b"\x01\x00\x00\x00\x00\x00\x00\x00{";
        match SafeTensors::deserialize(serialized) {
            Err(SafeTensorError::InvalidHeaderDeserialization) => {
                // Yes we have the correct error
            }
            _ => panic!("This should not be able to be deserialized"),
        }
    }

    #[test]
    /// Test that the JSON header may be trailing-padded with JSON whitespace characters.
    fn test_whitespace_padded_header() {
        let serialized = b"\x06\x00\x00\x00\x00\x00\x00\x00{}\x0D\x20\x09\x0A";
        let loaded = SafeTensors::deserialize(serialized).unwrap();
        assert_eq!(loaded.len(), 0);
    }

    // Reserver for 0.4.0
    // #[test]
    // /// Test that the JSON header must begin with a `{` character.
    // fn test_whitespace_start_padded_header_is_not_allowed() {
    //     let serialized = b"\x06\x00\x00\x00\x00\x00\x00\x00\x09\x0A{}\x0D\x20";
    //     match SafeTensors::deserialize(serialized) {
    //         Err(SafeTensorError::InvalidHeaderStart) => {
    //             // Correct error
    //         }
    //         _ => panic!("This should not be able to be deserialized"),
    //     }
    // }

    #[test]
    fn test_zero_sized_tensor() {
        let serialized = b"<\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[2,0],\"data_offsets\":[0, 0]}}";
        let loaded = SafeTensors::deserialize(serialized).unwrap();

        assert_eq!(loaded.names(), vec!["test"]);
        let tensor = loaded.tensor("test").unwrap();
        assert_eq!(tensor.shape(), vec![2, 0]);
        assert_eq!(tensor.dtype(), Dtype::I32);
        assert_eq!(tensor.data(), b"");
    }

    #[test]
    fn test_invalid_info() {
        let serialized = b"<\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[2,2],\"data_offsets\":[0, 4]}}";
        match SafeTensors::deserialize(serialized) {
            Err(SafeTensorError::TensorInvalidInfo) => {
                // Yes we have the correct error
            }
            _ => panic!("This should not be able to be deserialized"),
        }
    }

    #[test]
    fn test_validation_overflow() {
        // u64::MAX =  18_446_744_073_709_551_615u64
        // Overflow the shape calculation.
        let serialized = b"O\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[2,18446744073709551614],\"data_offsets\":[0,16]}}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
        match SafeTensors::deserialize(serialized) {
            Err(SafeTensorError::ValidationOverflow) => {
                // Yes we have the correct error
            }
            _ => panic!("This should not be able to be deserialized"),
        }
        // u64::MAX =  18_446_744_073_709_551_615u64
        // Overflow the num_elements * total shape.
        let serialized = b"N\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[2,9223372036854775807],\"data_offsets\":[0,16]}}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
        match SafeTensors::deserialize(serialized) {
            Err(SafeTensorError::ValidationOverflow) => {
                // Yes we have the correct error
            }
            _ => panic!("This should not be able to be deserialized"),
        }
    }
}
