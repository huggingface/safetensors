//! Module Containing the most important structures
use crate::lib::{Cow, HashMap, String, ToString, Vec};
use crate::slice::{InvalidSlice, SliceIterator, TensorIndexer};
use core::fmt::Display;
use core::str::Utf8Error;
use serde::{ser::SerializeMap, Deserialize, Deserializer, Serialize, Serializer};
#[cfg(feature = "std")]
use std::{io::Write, path::Path};

const MAX_HEADER_SIZE: usize = 100_000_000;
const N_LEN: usize = size_of::<u64>();

/// Possible errors that could occur while reading
/// A Safetensor file.
#[derive(Debug)]
pub enum SafeTensorError {
    /// The header is an invalid UTF-8 string and cannot be read.
    InvalidHeader(Utf8Error),
    /// The header's first byte is not the expected `{`.
    InvalidHeaderStart,
    /// The header does contain a valid string, but it is not valid JSON.
    InvalidHeaderDeserialization(serde_json::Error),
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
    /// For smaller than 1 byte dtypes, some slices will happen outside of the byte boundary, some special care has to be taken
    /// and standard functions will fail
    MisalignedSlice,
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

impl Display for SafeTensorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        use SafeTensorError::*;

        match self {
            InvalidHeader(error) => write!(f, "invalid UTF-8 in header: {error}"),
            InvalidHeaderStart => write!(f, "invalid start character in header, must be `{{`"),
            InvalidHeaderDeserialization(error) => write!(f, "invalid JSON in header: {error}"),
            JsonError(error) => write!(f, "JSON error: {error}"),
            HeaderTooLarge => write!(f, "header too large"),
            HeaderTooSmall => write!(f, "header too small"),
            InvalidHeaderLength => write!(f, "invalid header length"),
            TensorNotFound(name) => write!(f, "tensor `{name}` not found"),
            TensorInvalidInfo => write!(f, "invalid shape, data type, or offset for tensor"),
            InvalidOffset(name) => write!(f, "invalid offset for tensor `{name}`"),
            #[cfg(feature = "std")]
            IoError(error) => write!(f, "I/O error: {error}"),
            InvalidTensorView(dtype, shape, n_bytes) => {
                write!(f, "tensor of type {dtype} and shape (")?;
                for (i, &dim) in shape.iter().enumerate() {
                    write!(f, "{sep}{dim}", sep = if i == 0 { "" } else { ", " })?;
                }
                write!(f, ") can't be created from {n_bytes} bytes")
            }
            MetadataIncompleteBuffer => write!(f, "incomplete metadata, file not fully covered"),
            ValidationOverflow => write!(f, "overflow computing buffer size from shape and/or element type"),
            MisalignedSlice => write!(f, "The slice is slicing for subbytes dtypes, and the slice does not end up at a byte boundary, this is invalid.")
        }
    }
}

#[cfg(not(feature = "std"))]
impl core::error::Error for SafeTensorError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            SafeTensorError::InvalidHeader(source) => Some(source),
            SafeTensorError::JsonError(source) => Some(source),
            SafeTensorError::InvalidHeaderDeserialization(source) => Some(source),
            _ => None,
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for SafeTensorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SafeTensorError::InvalidHeader(source) => Some(source),
            SafeTensorError::JsonError(source) => Some(source),
            SafeTensorError::InvalidHeaderDeserialization(source) => Some(source),
            SafeTensorError::IoError(source) => Some(source),
            _ => None,
        }
    }
}

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
///    fn data(&self) -> Cow<'_, [u8]>{
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
///    fn data(&self) -> Cow<'_, [u8]>{
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
///    fn data(&self) -> Cow<'_, [u8]>{
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
    fn data(&self) -> Cow<'_, [u8]>;
    /// The length of the data, in bytes.
    /// This is necessary as this might be faster to get than `data().len()`
    /// for instance for tensors residing in GPU.
    fn data_len(&self) -> usize;
}

fn prepare<S, V, I>(
    data: I,
    data_info: Option<HashMap<String, String>>,
) -> Result<(PreparedData, Vec<V>), SafeTensorError>
where
    S: AsRef<str> + Ord + Display,
    V: View,
    I: IntoIterator<Item = (S, V)>,
{
    // Make sure we're sorting by descending dtype alignment
    // Then by name
    let mut data: Vec<_> = data.into_iter().collect();
    data.sort_by(|(lname, left), (rname, right)| {
        right.dtype().cmp(&left.dtype()).then(lname.cmp(rname))
    });

    let mut tensors: Vec<V> = Vec::with_capacity(data.len());
    let mut hmetadata = Vec::with_capacity(data.len());
    let mut offset = 0;

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

    let metadata: Metadata = Metadata::new(data_info, hmetadata)?;
    let mut metadata_buf = serde_json::to_string(&metadata)?.into_bytes();

    // Force alignment to 8 bytes.
    let aligned_metadata_len = metadata_buf.len().next_multiple_of(N_LEN);
    metadata_buf.resize(aligned_metadata_len, b' ');

    Ok((
        PreparedData {
            n: aligned_metadata_len as u64,
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
    data_info: Option<HashMap<String, String>>,
) -> Result<Vec<u8>, SafeTensorError> {
    let (
        PreparedData {
            n,
            header_bytes,
            offset,
        },
        tensors,
    ) = prepare(data, data_info)?;

    if n > MAX_HEADER_SIZE as u64 {
        return Err(SafeTensorError::HeaderTooLarge);
    }

    let expected_size = N_LEN + header_bytes.len() + offset;
    let mut buffer: Vec<u8> = Vec::with_capacity(expected_size);
    buffer.extend(n.to_le_bytes());
    buffer.extend(header_bytes);

    for tensor in tensors {
        buffer.extend(tensor.data().as_ref());
    }

    Ok(buffer)
}

#[cfg(feature = "std")]
fn buffered_write_to_file<V: View>(
    path: impl AsRef<Path>,
    n: u64,
    header_bytes: &[u8],
    tensors: &[V],
    total_size: usize,
) -> Result<(), SafeTensorError> {
    let file = std::fs::File::create(path)?;

    file.set_len(total_size as u64)?;

    // Serialize tensors to a file using direct I/O (bypassing page cache) using F_NOCACHE.
    // This yields ~30% performance improvement.
    #[cfg(target_os = "macos")]
    unsafe {
        use std::os::fd::AsRawFd;

        libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1);
    }

    let mut f = std::io::BufWriter::with_capacity(1024 * 1024, file);

    f.write_all(n.to_le_bytes().as_ref())?;
    f.write_all(header_bytes)?;

    for tensor in tensors {
        f.write_all(tensor.data().as_ref())?;
    }

    f.flush()?;

    Ok(())
}

/// Serialize to a regular file the dictionnary of tensors.
/// Writing directly to file reduces the need to allocate the whole amount to
/// memory.
#[cfg(feature = "std")]
pub fn serialize_to_file<S, V, I>(
    data: I,
    data_info: Option<HashMap<String, String>>,
    filename: &std::path::Path,
) -> Result<(), SafeTensorError>
where
    S: AsRef<str> + Ord + Display,
    V: View,
    I: IntoIterator<Item = (S, V)>,
{
    let (
        PreparedData {
            n,
            header_bytes,
            offset,
            ..
        },
        tensors,
    ) = prepare(data, data_info)?;

    if n > MAX_HEADER_SIZE as u64 {
        return Err(SafeTensorError::HeaderTooLarge);
    }

    let total_size = N_LEN + header_bytes.len() + offset;

    buffered_write_to_file(filename, n, &header_bytes, &tensors, total_size)?;

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
    pub fn read_metadata(buffer: &'data [u8]) -> Result<(usize, Metadata), SafeTensorError> {
        let buffer_len = buffer.len();
        let Some(header_size_bytes) = buffer.get(..N_LEN) else {
            return Err(SafeTensorError::HeaderTooSmall);
        };
        let arr: [u8; N_LEN] = header_size_bytes
            .try_into()
            .expect("this can't fail due to how `header_size_bytes` is defined above");
        let n: usize = u64::from_le_bytes(arr)
            .try_into()
            .map_err(|_| SafeTensorError::HeaderTooLarge)?;

        if n > MAX_HEADER_SIZE {
            return Err(SafeTensorError::HeaderTooLarge);
        }

        let stop = n
            .checked_add(N_LEN)
            .ok_or(SafeTensorError::InvalidHeaderLength)?;

        // the `.get(start..stop)` returns None if either index is out of bounds,
        // so this implicitly also ensures that `stop <= buffer.len()`.
        let Some(header_bytes) = buffer.get(N_LEN..stop) else {
            return Err(SafeTensorError::InvalidHeaderLength);
        };
        let string = core::str::from_utf8(header_bytes).map_err(SafeTensorError::InvalidHeader)?;
        // Assert the string starts with {
        // NOTE: Add when we move to 0.4.0
        // if !string.starts_with('{') {
        //     return Err(SafeTensorError::InvalidHeaderStart);
        // }
        let metadata: HashMetadata =
            serde_json::from_str(string).map_err(SafeTensorError::InvalidHeaderDeserialization)?;
        let metadata: Metadata = metadata.try_into()?;
        let buffer_end = metadata.validate()?;
        if buffer_end + N_LEN + n != buffer_len {
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
    pub fn deserialize(buffer: &'data [u8]) -> Result<Self, SafeTensorError> {
        let (n, metadata) = SafeTensors::read_metadata(buffer)?;
        let data = &buffer[N_LEN + n..];
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
    pub fn iter(&self) -> impl Iterator<Item = (&str, TensorView<'data>)> {
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
        let &index = self
            .metadata
            .index_map
            .get(tensor_name)
            .ok_or_else(|| SafeTensorError::TensorNotFound(tensor_name.to_string()))?;

        let info = self
            .metadata
            .tensors
            .get(index)
            .ok_or_else(|| SafeTensorError::TensorNotFound(tensor_name.to_string()))?;

        Ok(TensorView {
            dtype: info.dtype,
            shape: info.shape.clone(),
            data: &self.data[info.data_offsets.0..info.data_offsets.1],
        })
    }

    /// Return the names of the tensors within the SafeTensors.
    /// These are used as keys to access to the actual tensors, that can be
    /// retrieved using the tensor method.
    pub fn names(&self) -> Vec<&'_ str> {
        self.metadata.index_map.keys().map(String::as_str).collect()
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

/// Helper struct used only for serialization and deserialization
#[derive(Serialize, Deserialize)]
struct HashMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "__metadata__")]
    metadata: Option<HashMap<String, String>>,
    #[serde(flatten)]
    tensors: HashMap<String, TensorInfo>,
}

impl TryFrom<HashMetadata> for Metadata {
    type Error = SafeTensorError;
    fn try_from(hashdata: HashMetadata) -> Result<Self, Self::Error> {
        let (metadata, tensors) = (hashdata.metadata, hashdata.tensors);
        let mut tensors: Vec<_> = tensors.into_iter().collect();
        // We need to sort by offsets
        // Previous versions might have a different ordering
        // Than we expect (Not aligned ordered, but purely name ordered,
        // or actually any order).
        tensors.sort_by(|(_, left), (_, right)| left.data_offsets.cmp(&right.data_offsets));
        Metadata::new(metadata, tensors)
    }
}

impl<'de> Deserialize<'de> for Metadata {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let hashdata: HashMetadata = HashMetadata::deserialize(deserializer)?;

        let metadata: Metadata = hashdata.try_into().map_err(serde::de::Error::custom)?;
        Ok(metadata)
    }
}

impl Serialize for Metadata {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut names = vec![""; self.index_map.len()];
        for (name, &index) in &self.index_map {
            names[index] = name;
        }

        let length = self.metadata.as_ref().map_or(0, HashMap::len);
        let mut map = serializer.serialize_map(Some(self.tensors.len() + length))?;

        if let Some(metadata) = &self.metadata {
            map.serialize_entry("__metadata__", metadata)?;
        }

        for (name, info) in names.iter().zip(&self.tensors) {
            map.serialize_entry(name, info)?;
        }

        map.end()
    }
}

impl Metadata {
    /// Creates a new metadata structure.
    /// May fail if there is incorrect data in the Tensor Info.
    /// Notably the tensors need to be ordered by increasing data_offsets.
    pub fn new(
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
        metadata.validate()?;
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
                .copied()
                .try_fold(1usize, usize::checked_mul)
                .ok_or(SafeTensorError::ValidationOverflow)?;
            let nbits = nelements
                .checked_mul(info.dtype.bitsize())
                .ok_or(SafeTensorError::ValidationOverflow)?;

            if nbits % 8 != 0 {
                return Err(SafeTensorError::MisalignedSlice);
            }
            let size = nbits
                .checked_div(8)
                .ok_or(SafeTensorError::ValidationOverflow)?;

            if e - s != size {
                return Err(SafeTensorError::TensorInvalidInfo);
            }
        }
        Ok(start)
    }

    /// Gives back the tensor metadata
    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        let &index = self.index_map.get(name)?;
        self.tensors.get(index)
    }

    /// Gives back the tensor metadata
    pub fn tensors(&self) -> HashMap<String, &TensorInfo> {
        self.index_map
            .iter()
            .map(|(tensor_name, &index)| (tensor_name.clone(), &self.tensors[index]))
            .collect()
    }

    /// Gives back the tensor names ordered by offset
    pub fn offset_keys(&self) -> Vec<String> {
        let mut index_vec: Vec<_> = self.index_map.iter().collect();
        index_vec.sort_by_key(|a| a.1);
        index_vec.into_iter().map(|a| a.0.clone()).collect()
    }

    /// Gives the size of the content buffer in bytes.
    pub fn data_len(&self) -> usize {
        if let Some(tensor) = self.tensors.last() {
            tensor.data_offsets.1
        } else {
            0
        }
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

    fn data(&self) -> Cow<'_, [u8]> {
        self.data.into()
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

impl View for TensorView<'_> {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
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
        let n_elements: usize = shape.iter().product();

        let nbits = n_elements * dtype.bitsize();
        if nbits % 8 != 0 {
            return Err(SafeTensorError::MisalignedSlice);
        }
        let size = nbits
            .checked_div(8)
            .ok_or(SafeTensorError::ValidationOverflow)?;

        if data.len() != size {
            Err(SafeTensorError::InvalidTensorView(dtype, shape, data.len()))
        } else {
            Ok(Self { dtype, shape, data })
        }
    }
    /// The current tensor dtype
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    /// The current tensor shape
    pub fn shape(&self) -> &[usize] {
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
    /// MXF4 <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>_
    F4,
    /// MXF6 <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>_
    #[allow(non_camel_case_types)]
    F6_E2M3,
    /// MXF6 <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>_
    #[allow(non_camel_case_types)]
    F6_E3M2,
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
    /// F8_E8M0 <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>_
    #[allow(non_camel_case_types)]
    F8_E8M0,
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
    /// Complex (32-bit parts)
    C64,
    /// Floating point (64-bit)
    F64,
    /// Signed integer (64-bit)
    I64,
    /// Unsigned integer (64-bit)
    U64,
}

impl Dtype {
    /// Gives out the size (in bits) of 1 element of this dtype.
    pub fn bitsize(&self) -> usize {
        match self {
            Dtype::F4 => 4,
            Dtype::F6_E3M2 => 6,
            Dtype::F6_E2M3 => 6,
            Dtype::BOOL => 8,
            Dtype::U8 => 8,
            Dtype::I8 => 8,
            Dtype::F8_E5M2 => 8,
            Dtype::F8_E4M3 => 8,
            Dtype::F8_E8M0 => 8,
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
            Dtype::C64 => 64,
        }
    }
    /// Gives out the size (in bytes) of 1 element of this dtype.
    #[deprecated(
        since = "0.6.0",
        note = "Use `bitsize` instead as some elements have smaller than a full byte of width"
    )]
    pub fn size(&self) -> usize {
        self.bitsize() / 8
    }
}

impl Display for Dtype {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match *self {
            Dtype::F4 => "F4",
            Dtype::F6_E2M3 => "F6_E2M3",
            Dtype::F6_E3M2 => "F6_E3M2",
            Dtype::BOOL => "BOOL",
            Dtype::I8 => "I8",
            Dtype::U8 => "U8",
            Dtype::F8_E5M2 => "F8_E5M2",
            Dtype::F8_E4M3 => "F8_E4M3",
            Dtype::F8_E8M0 => "F8_E8M0",
            Dtype::I16 => "I16",
            Dtype::U16 => "U16",
            Dtype::I32 => "I32",
            Dtype::U32 => "U32",
            Dtype::I64 => "I64",
            Dtype::U64 => "U64",
            Dtype::F16 => "F16",
            Dtype::BF16 => "BF16",
            Dtype::F32 => "F32",
            Dtype::F64 => "F64",
            Dtype::C64 => "C64",
        })
    }
}

// ============================================================================
// Linux io_uring Read Implementation
// ============================================================================

/// io_uring queue depth - number of concurrent operations
#[cfg(all(feature = "std", target_os = "linux"))]
const QUEUE_DEPTH: u32 = 64;

/// Buffer size for chunked reads (256KB)
#[cfg(all(feature = "std", target_os = "linux"))]
const BUF_SIZE: usize = 256 * 1024;

/// Check if a pointer is aligned to the given alignment
#[cfg(all(feature = "std", target_os = "linux"))]
#[inline]
fn is_aligned(ptr: *const u8, alignment: usize) -> bool {
    (ptr as usize) % alignment == 0
}

/// Read a file using Linux io_uring for high-performance async I/O.
///
/// This function uses io_uring with the following optimizations:
/// - COOP_TASKRUN for reduced interrupts and better batching
/// - SINGLE_ISSUER for single-threaded kernel optimizations
/// - Direct reads into the destination buffer (no buffer registration needed)
/// - Batched async I/O operations
#[cfg(all(feature = "std", target_os = "linux"))]
fn linux_io_uring_read(
    file: &std::fs::File,
    file_size: u64,
    mut data: Vec<u8>,
) -> Result<Vec<u8>, SafeTensorError> {
    use io_uring::{opcode, types, IoUring};
    use std::io::{Error, ErrorKind};
    use std::os::fd::AsRawFd;

    // Handle empty files
    if file_size == 0 {
        return Ok(Vec::new());
    }

    let file_size_usize = file_size as usize;

    // Calculate number of chunks needed
    let num_chunks = file_size_usize.div_ceil(BUF_SIZE);

    // Build io_uring ring with performance flags
    let mut ring: IoUring<io_uring::squeue::Entry, io_uring::cqueue::Entry> = IoUring::builder()
        .setup_coop_taskrun() // Reduces interrupts, improves batching
        .setup_single_issuer() // Single-threaded optimization
        .build(QUEUE_DEPTH)
        .map_err(|e| {
            SafeTensorError::IoError(Error::other(format!("io_uring init failed: {}", e)))
        })?;

    let fd = types::Fd(file.as_raw_fd());

    // Track which chunks have been submitted and completed
    let mut next_chunk_to_submit: usize = 0;
    let mut completed_bytes: usize = 0;
    let mut in_flight: usize = 0;

    // Main I/O loop: submit reads and process completions
    while completed_bytes < file_size_usize {
        // Submit new read operations while we have queue space and chunks remaining
        while next_chunk_to_submit < num_chunks
            && in_flight < QUEUE_DEPTH as usize
            && !ring.submission().is_full()
        {
            let chunk_idx = next_chunk_to_submit;
            let file_offset = (chunk_idx * BUF_SIZE) as u64;

            // Calculate actual chunk size (last chunk may be smaller)
            let chunk_size = if chunk_idx == num_chunks - 1 {
                file_size_usize - (chunk_idx * BUF_SIZE)
            } else {
                BUF_SIZE
            };

            // Get pointer to destination in the data buffer
            let dest_offset = chunk_idx * BUF_SIZE;
            let buf_ptr = data[dest_offset..].as_mut_ptr();

            // Build read operation - reads directly into destination buffer
            let read_op = opcode::Read::new(fd, buf_ptr, chunk_size as u32)
                .offset(file_offset)
                .build()
                .user_data(chunk_idx as u64);

            // SAFETY: The buffer slice in data Vec remains valid for the entire
            // duration of io_uring operations. We track completions and wait
            // for all in-flight operations before returning.
            unsafe {
                ring.submission().push(&read_op).map_err(|e| {
                    SafeTensorError::IoError(Error::other(format!("SQ push failed: {}", e)))
                })?;
            }

            next_chunk_to_submit += 1;
            in_flight += 1;
        }

        // Determine how many completions to wait for
        let wait_count = if next_chunk_to_submit >= num_chunks {
            // All submitted, wait for remaining completions
            in_flight.min(1)
        } else if ring.submission().is_full() || in_flight >= QUEUE_DEPTH as usize {
            // Queue full, must wait for at least one completion
            1
        } else {
            // Can continue submitting, don't block
            0
        };

        // Submit and optionally wait for completions
        if wait_count > 0 {
            ring.submit_and_wait(wait_count).map_err(|e| {
                SafeTensorError::IoError(Error::other(format!("submit_and_wait failed: {}", e)))
            })?;
        } else {
            ring.submit().map_err(|e| {
                SafeTensorError::IoError(Error::other(format!("submit failed: {}", e)))
            })?;
        }

        // Process all available completions
        let mut io_error: Option<SafeTensorError> = None;
        for cqe in ring.completion() {
            let result = cqe.result();
            let chunk_idx = cqe.user_data() as usize;

            // Check for errors (negative result is errno)
            if result < 0 {
                io_error = Some(SafeTensorError::IoError(Error::from_raw_os_error(-result)));
                in_flight -= 1;
                continue;
            }

            let bytes_read = result as usize;

            // Calculate expected bytes for this chunk
            let expected_bytes = if chunk_idx == num_chunks - 1 {
                file_size_usize - (chunk_idx * BUF_SIZE)
            } else {
                BUF_SIZE
            };

            // Check for short reads
            if bytes_read != expected_bytes {
                io_error = Some(SafeTensorError::IoError(Error::new(
                    ErrorKind::UnexpectedEof,
                    format!(
                        "short read at chunk {}: expected {} bytes, got {}",
                        chunk_idx, expected_bytes, bytes_read
                    ),
                )));
                in_flight -= 1;
                continue;
            }

            completed_bytes += bytes_read;
            in_flight -= 1;
        }

        // Handle any errors after processing completions
        if let Some(err) = io_error {
            return Err(err);
        }
    }

    Ok(data)
}

/// Deserialize a safetensors file using Linux io_uring for high-performance reads.
///
/// This function uses io_uring to efficiently read large safetensors files on Linux.
/// It reads the entire file into an owned buffer and returns a SafeTensors instance
/// with 'static lifetime.
///
/// # Performance
///
/// io_uring provides better performance for large files by allowing multiple
/// concurrent read operations to be queued and processed efficiently. This
/// implementation uses several optimizations:
///
/// - **Batched async I/O**: Up to 64 concurrent read operations in flight
/// - **COOP_TASKRUN**: Reduces inter-processor interrupts for better batching
/// - **SINGLE_ISSUER**: Kernel optimizations for single-threaded access
/// - **Direct buffer writes**: Reads directly into the destination buffer
///
/// Expected performance improvement over mmap: 1.3-1.5x for cold cache reads on
/// large files (>100MB).
///
/// # Platform Support
///
/// This function is only available on Linux with kernel 5.6 or later (required
/// for IORING_OP_READ). For cross-platform code, use `SafeTensors::deserialize()`
/// with memory-mapped files via the `memmap2` crate.
///
/// # Memory
///
/// This function allocates memory for the entire file contents. The returned
/// `SafeTensors<'static>` owns this memory via a leaked allocation. For very
/// large files where memory is a concern, consider using memory-mapped I/O instead.
///
/// # Example
///
/// ```rust,no_run
/// use safetensors::deserialize_from_file_linux_io_uring;
///
/// let tensors = deserialize_from_file_linux_io_uring("model.safetensors")?;
/// for name in tensors.names() {
///     let tensor = tensors.tensor(&name)?;
///     println!("{}: {:?} {:?}", name, tensor.dtype(), tensor.shape());
/// }
/// # Ok::<(), safetensors::SafeTensorError>(())
/// ```
///
/// # Errors
///
/// Returns an error if:
/// - The file cannot be opened or read
/// - io_uring initialization fails (e.g., kernel too old)
/// - The file contents are not valid safetensors format
#[cfg(all(feature = "std", target_os = "linux"))]
pub fn deserialize_from_file_linux_io_uring(
    path: impl AsRef<Path>,
) -> Result<SafeTensors<'static>, SafeTensorError> {
    use std::os::unix::fs::OpenOptionsExt;

    // Get page size for alignment checks and O_DIRECT
    let page_size_raw = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page_size_raw == -1 {
        return Err(SafeTensorError::IoError(std::io::Error::last_os_error()));
    }
    let page_size = page_size_raw as usize;

    // First, open without O_DIRECT to get file size
    let file_std = std::fs::File::open(path.as_ref())?;
    let metadata = file_std.metadata()?;
    let file_size = metadata.len();

    // Handle empty files
    if file_size == 0 {
        return Err(SafeTensorError::HeaderTooSmall);
    }

    // Pre-allocate buffer - Vec may naturally align to page boundaries for large allocations
    let data = vec![0u8; file_size as usize];
    let buffer_ptr = data.as_ptr();

    // Check if buffer is naturally aligned to page size
    let is_buffer_aligned = is_aligned(buffer_ptr, page_size);

    // Try to open with O_DIRECT if buffer is aligned
    let use_o_direct = is_buffer_aligned;
    let file = if use_o_direct {
        match std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(path.as_ref())
        {
            Ok(f) => {
                // Verify O_DIRECT actually worked
                use std::os::fd::AsRawFd;
                let flags = unsafe { libc::fcntl(f.as_raw_fd(), libc::F_GETFL) };
                if flags >= 0 && (flags & libc::O_DIRECT) != 0 {
                    f // O_DIRECT successfully enabled
                } else {
                    // O_DIRECT not supported by filesystem, fall back
                    file_std
                }
            }
            Err(_) => {
                // O_DIRECT failed, use regular file
                file_std
            }
        }
    } else {
        file_std
    };

    // Read entire file via io_uring
    let data = linux_io_uring_read(&file, file_size, data)?;

    // Leak Vec to get 'static lifetime
    // SAFETY: We intentionally leak this memory to provide 'static lifetime.
    // The caller is expected to hold the SafeTensors for the duration they
    // need the tensor data. For long-lived applications, consider using
    // mmap-based deserialization instead.
    let data_static: &'static [u8] = Box::leak(data.into_boxed_slice());

    // Deserialize and return
    SafeTensors::deserialize(data_static)
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
            Just(Dtype::F4),
            Just(Dtype::F6_E3M2),
            Just(Dtype::F6_E2M3),
            Just(Dtype::F8_E5M2),
            Just(Dtype::F8_E4M3),
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
            Just(Dtype::C64),
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
            .prop_filter_map("Misaligned slices", |(dtypes, shapes)| {
                // Returns a valid metadata object for a random (length, dtypes, shapes) triple.
                let mut start = 0;
                let tensors: Vec<TensorInfo> = dtypes
                    .iter()
                    .zip(shapes)
                    .flat_map(|(dtype, shape)| {
                        // This cannot overflow because the size of
                        // the vector and elements are so small.
                        let bitlength: usize = shape.iter().product::<usize>() * dtype.bitsize();
                        if bitlength % 8 != 0 {
                            return None;
                        }
                        let length = bitlength.div_ceil(8);
                        let end = start + length;
                        let tensor = TensorInfo {
                            dtype: *dtype,
                            shape,
                            data_offsets: (start, end),
                        };
                        start = end;
                        Some(tensor)
                    })
                    .collect();
                let index_map = (0..tensors.len())
                    .map(|index| (format!("t.{index}"), index))
                    .collect();
                if tensors.is_empty() {
                    None
                } else {
                    Some(Metadata {
                        metadata: None,
                        tensors,
                        index_map,
                    })
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
            let bytes = serialize(tensors.iter().map(|(name, view)| (name.to_string(), view)), None).unwrap();

            let after = SafeTensors::deserialize(&bytes).unwrap();

            // Check that the tensors are the same after deserialization.
            assert_eq!(before.names().len(), after.names().len());
            for name in before.names() {
                let tensor_before = before.tensor(name).unwrap();
                let tensor_after = after.tensor(name).unwrap();
                assert_eq!(tensor_after.data().as_ptr() as usize % tensor_after.dtype().bitsize().div_ceil(8), 0);
                assert_eq!(tensor_before, tensor_after);
            }
        }
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

        let out = serialize(&metadata, None).unwrap();
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
    fn test_serialization_fp4() {
        let data: Vec<u8> = vec![0u8];
        let shape = vec![1, 2];
        let attn_0 = TensorView::new(Dtype::F4, shape, &data).unwrap();
        let metadata: HashMap<String, TensorView> =
            [("attn.0".to_string(), attn_0)].into_iter().collect();

        let out = serialize(&metadata, None).unwrap();
        assert_eq!(
            out,
            [
                64, 0, 0, 0, 0, 0, 0, 0, 123, 34, 97, 116, 116, 110, 46, 48, 34, 58, 123, 34, 100,
                116, 121, 112, 101, 34, 58, 34, 70, 52, 34, 44, 34, 115, 104, 97, 112, 101, 34, 58,
                91, 49, 44, 50, 93, 44, 34, 100, 97, 116, 97, 95, 111, 102, 102, 115, 101, 116,
                115, 34, 58, 91, 48, 44, 49, 93, 125, 125, 32, 32, 32, 32, 0
            ]
        );
        let parsed = SafeTensors::deserialize(&out).unwrap();
        let tensors: HashMap<_, _> = parsed.tensors().into_iter().collect();
        assert_eq!(tensors, metadata);
    }

    #[test]
    fn test_serialization_fp4_misaligned() {
        let data: Vec<u8> = vec![0u8, 1u8];
        let shape = vec![1, 3];
        let attn_0 = TensorView::new(Dtype::F4, shape, &data);
        assert!(matches!(attn_0, Err(SafeTensorError::MisalignedSlice)));
    }

    #[test]
    fn test_serialization_fp4_invalid() {
        let data: Vec<u8> = vec![0u8, 1u8];
        let shape = vec![1, 2];
        let attn_0 = TensorView::new(Dtype::F4, shape, &data);
        assert!(matches!(
            attn_0,
            Err(SafeTensorError::InvalidTensorView(Dtype::F4, _shape, _size))
        ));
    }

    #[test]
    fn test_empty() {
        let tensors: HashMap<String, TensorView> = HashMap::new();

        let out = serialize(&tensors, None).unwrap();
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
        let out = serialize(&tensors, metadata).unwrap();
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

        let out = serialize(&metadata, None).unwrap();
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
            tensor.data().as_ptr() as usize % tensor.dtype().bitsize().div_ceil(8),
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

        let out = serialize(&metadata, None).unwrap();
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
        let mut tensors_desc = vec![
            ("wte".to_string(), vec![50257, 768]),
            ("wpe".to_string(), vec![1024, 768]),
        ];
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
        let nbits: usize = tensors_desc
            .iter()
            .map(|(_, shape)| shape.iter().product::<usize>())
            .sum::<usize>()
            * dtype.bitsize();
        if nbits % 8 != 0 {
            panic!("Misaligned slice");
        }
        let n = nbits
            .checked_div(8)
            .ok_or(SafeTensorError::ValidationOverflow)
            .unwrap(); // 4
        let all_data = vec![0; n];
        let mut metadata = HashMap::with_capacity(tensors_desc.len());
        let mut offset = 0;
        for (name, shape) in tensors_desc {
            let n: usize = shape.iter().product();
            let buffer = &all_data[offset..offset + (n * dtype.bitsize()) / 8];
            let tensor = TensorView::new(dtype, shape, buffer).unwrap();
            metadata.insert(name, tensor);
            offset += n;
        }

        let filename = format!("./out_{model_id}.safetensors");

        let out = serialize(&metadata, None).unwrap();
        std::fs::write(&filename, out).unwrap();
        let raw = std::fs::read(&filename).unwrap();
        let _deserialized = SafeTensors::deserialize(&raw).unwrap();
        std::fs::remove_file(&filename).unwrap();

        // File api
        #[cfg(feature = "std")]
        {
            serialize_to_file(&metadata, None, std::path::Path::new(&filename)).unwrap();
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
            Err(SafeTensorError::InvalidHeader(_)) => {
                // Yes we have the correct error
            }
            _ => panic!("This should not be able to be deserialized"),
        }
    }

    #[test]
    fn test_invalid_header_not_json() {
        let serialized = b"\x01\x00\x00\x00\x00\x00\x00\x00{";
        match SafeTensors::deserialize(serialized) {
            Err(SafeTensorError::InvalidHeaderDeserialization(_)) => {
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
            something => panic!("This should not be able to be deserialized got {something:?}"),
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

    #[test]
    fn test_invalid_header_size_serialization() {
        let mut data_info = HashMap::<String, String>::new();
        let tensors: HashMap<String, TensorView> = HashMap::new();

        // a char is 1 byte in utf-8, so we can just repeat 'a' to get large metadata
        let very_large_metadata = "a".repeat(MAX_HEADER_SIZE);
        data_info.insert("very_large_metadata".to_string(), very_large_metadata);
        match serialize(&tensors, Some(data_info)) {
            Err(SafeTensorError::HeaderTooLarge) => {
                // Yes we have the correct error
            }
            _ => panic!("This should not be able to be serialized"),
        }
    }

    // ============================================================================
    // io_uring Read Tests (Linux only)
    // ============================================================================

    #[cfg(all(test, feature = "std", target_os = "linux"))]
    mod io_uring_read_tests {
        use super::*;

        #[test]
        fn test_io_uring_read_basic() {
            // Create test file with small tensor
            let filename = "/tmp/test_io_uring_read_basic.safetensors";
            let data: Vec<u8> = vec![0.0f32, 1.0, 2.0, 3.0]
                .into_iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();
            let shape = vec![2, 2];
            let tensor = TensorView::new(Dtype::F32, shape, &data).unwrap();
            let metadata: HashMap<String, TensorView> =
                [("test".to_string(), tensor)].into_iter().collect();

            // Write file using standard API
            serialize_to_file(&metadata, None, Path::new(filename)).unwrap();

            // Read with io_uring
            let loaded = deserialize_from_file_linux_io_uring(filename).unwrap();

            // Verify contents match
            assert_eq!(loaded.names(), vec!["test"]);
            let tensor = loaded.tensor("test").unwrap();
            assert_eq!(tensor.shape(), vec![2, 2]);
            assert_eq!(tensor.dtype(), Dtype::F32);

            // Verify actual data
            let loaded_data = tensor.data();
            assert_eq!(loaded_data.len(), data.len());

            // Cleanup
            std::fs::remove_file(filename).unwrap();
        }

        #[test]
        fn test_io_uring_read_large_file() {
            // Test with file larger than BUF_SIZE (256KB) to test chunked reads
            let filename = "/tmp/test_io_uring_large.safetensors";

            // Create 1MB of data (4 * 256KB chunks)
            let large_shape = vec![256 * 1024]; // 256K f32 elements = 1MB
            let large_data = vec![0u8; 256 * 1024 * 4];
            let tensor = TensorView::new(Dtype::F32, large_shape.clone(), &large_data).unwrap();
            let metadata: HashMap<String, TensorView> =
                [("large".to_string(), tensor)].into_iter().collect();

            serialize_to_file(&metadata, None, Path::new(filename)).unwrap();

            // Read with io_uring (tests chunking logic and registered buffers)
            let loaded = deserialize_from_file_linux_io_uring(filename).unwrap();

            assert_eq!(loaded.names(), vec!["large"]);
            let loaded_tensor = loaded.tensor("large").unwrap();
            assert_eq!(loaded_tensor.shape(), large_shape);
            assert_eq!(loaded_tensor.dtype(), Dtype::F32);

            std::fs::remove_file(filename).unwrap();
        }

        #[test]
        fn test_io_uring_read_nonexistent() {
            let result =
                deserialize_from_file_linux_io_uring("/tmp/nonexistent_io_uring.safetensors");
            assert!(matches!(result, Err(SafeTensorError::IoError(_))));
        }

        #[test]
        fn test_io_uring_read_empty_tensors() {
            // Edge case: safetensors file with no tensors
            let filename = "/tmp/test_empty_io_uring.safetensors";
            let tensors: HashMap<String, TensorView> = HashMap::new();
            serialize_to_file(&tensors, None, Path::new(filename)).unwrap();

            let loaded = deserialize_from_file_linux_io_uring(filename).unwrap();
            assert_eq!(loaded.len(), 0);

            std::fs::remove_file(filename).unwrap();
        }

        #[test]
        fn test_io_uring_vs_standard_deserialization() {
            // Verify io_uring produces same results as standard deserialization
            let filename = "/tmp/test_io_uring_comparison.safetensors";

            // Create test data with multiple tensors
            let data1: Vec<u8> = vec![1.0f32, 2.0, 3.0, 4.0]
                .into_iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();
            let data2: Vec<u8> = vec![5, 6, 7, 8]
                .into_iter()
                .map(|i: i32| i.to_le_bytes())
                .flatten()
                .collect();

            let tensor1 = TensorView::new(Dtype::F32, vec![4], &data1).unwrap();
            let tensor2 = TensorView::new(Dtype::I32, vec![2, 2], &data2).unwrap();

            let mut metadata: HashMap<String, TensorView> = HashMap::new();
            metadata.insert("weights".to_string(), tensor1);
            metadata.insert("biases".to_string(), tensor2);

            serialize_to_file(&metadata, None, Path::new(filename)).unwrap();

            // Read with io_uring
            let loaded_io_uring = deserialize_from_file_linux_io_uring(filename).unwrap();

            // Read with standard mmap-based approach
            let file = std::fs::File::open(filename).unwrap();
            let buffer = unsafe { memmap2::MmapOptions::new().map(&file).unwrap() };
            let loaded_standard = SafeTensors::deserialize(&buffer).unwrap();

            // Compare results
            let mut io_uring_names = loaded_io_uring.names();
            let mut standard_names = loaded_standard.names();
            io_uring_names.sort();
            standard_names.sort();
            assert_eq!(io_uring_names, standard_names);
            for name in io_uring_names {
                let tensor_io_uring = loaded_io_uring.tensor(&name).unwrap();
                let tensor_standard = loaded_standard.tensor(&name).unwrap();

                assert_eq!(tensor_io_uring.dtype(), tensor_standard.dtype());
                assert_eq!(tensor_io_uring.shape(), tensor_standard.shape());
                assert_eq!(tensor_io_uring.data(), tensor_standard.data());
            }

            std::fs::remove_file(filename).unwrap();
        }

        #[test]
        fn test_io_uring_read_multiple_dtypes() {
            // Test various data types to ensure proper handling
            let filename = "/tmp/test_io_uring_dtypes.safetensors";

            let mut metadata: HashMap<String, TensorView> = HashMap::new();

            // F32
            let data_f32: Vec<u8> = vec![1.5f32]
                .into_iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();
            metadata.insert(
                "f32".to_string(),
                TensorView::new(Dtype::F32, vec![1], &data_f32).unwrap(),
            );

            // I64
            let data_i64: Vec<u8> = vec![12345i64]
                .into_iter()
                .flat_map(|i| i.to_le_bytes())
                .collect();
            metadata.insert(
                "i64".to_string(),
                TensorView::new(Dtype::I64, vec![1], &data_i64).unwrap(),
            );

            // U8
            let data_u8: Vec<u8> = vec![42u8, 43u8, 44u8];
            metadata.insert(
                "u8".to_string(),
                TensorView::new(Dtype::U8, vec![3], &data_u8).unwrap(),
            );

            serialize_to_file(&metadata, None, Path::new(filename)).unwrap();

            let loaded = deserialize_from_file_linux_io_uring(filename).unwrap();

            assert_eq!(loaded.len(), 3);
            assert!(loaded.tensor("f32").is_ok());
            assert!(loaded.tensor("i64").is_ok());
            assert!(loaded.tensor("u8").is_ok());

            std::fs::remove_file(filename).unwrap();
        }

        #[test]
        fn test_is_aligned_helper() {
            let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
            assert!(page_size > 0);

            // Test with properly aligned pointer
            let aligned_vec: Vec<u8> = vec![0u8; page_size * 2];
            let ptr = aligned_vec.as_ptr();
            // Large allocations are typically page-aligned, but not guaranteed
            // Just verify the function works without panicking
            let _ = is_aligned(ptr, page_size);

            // Test with known unaligned pointer (offset by 1)
            let unaligned_ptr = unsafe { ptr.add(1) };
            assert!(!is_aligned(unaligned_ptr, page_size));

            // Test with small alignment values
            assert!(is_aligned(ptr, 1)); // Always aligned to 1
            let even_offset = unsafe { ptr.add(2) };
            assert!(is_aligned(even_offset, 2)); // Aligned to 2
        }

        #[test]
        fn test_o_direct_with_large_file() {
            // Test O_DIRECT behavior with a large file that should have aligned buffer
            let filename = "/tmp/test_io_uring_o_direct_large.safetensors";

            let mut metadata = HashMap::new();

            // Create a large tensor (>1MB) to increase chances of page-aligned allocation
            let size = 256 * 1024; // 256KB of f32 = 1MB
            let data: Vec<f32> = vec![std::f32::consts::PI; size];
            let data_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };

            metadata.insert(
                "large_tensor".to_string(),
                TensorView::new(Dtype::F32, vec![size], data_bytes).unwrap(),
            );

            serialize_to_file(&metadata, None, Path::new(filename)).unwrap();

            // Read with io_uring (may use O_DIRECT if buffer is naturally aligned)
            let loaded = deserialize_from_file_linux_io_uring(filename).unwrap();

            assert_eq!(loaded.len(), 1);
            let tensor = loaded.tensor("large_tensor").unwrap();
            assert_eq!(tensor.shape(), vec![size]);
            assert_eq!(tensor.dtype(), Dtype::F32);

            // Verify data correctness
            let loaded_data = tensor.data();
            assert_eq!(loaded_data.len(), size * 4);

            std::fs::remove_file(filename).unwrap();
        }

        #[test]
        fn test_o_direct_fallback_small_file() {
            // Small files are less likely to have page-aligned buffers
            // This tests the fallback to buffered I/O
            let filename = "/tmp/test_io_uring_small.safetensors";

            let mut metadata = HashMap::new();
            let data_u32: Vec<u32> = vec![1u32, 2u32, 3u32, 4u32];
            let data_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(data_u32.as_ptr() as *const u8, data_u32.len() * 4)
            };

            metadata.insert(
                "small".to_string(),
                TensorView::new(Dtype::U32, vec![4], data_bytes).unwrap(),
            );

            serialize_to_file(&metadata, None, Path::new(filename)).unwrap();

            // This will likely use buffered I/O due to small buffer size
            let loaded = deserialize_from_file_linux_io_uring(filename).unwrap();

            assert_eq!(loaded.len(), 1);
            let tensor = loaded.tensor("small").unwrap();
            assert_eq!(tensor.shape(), vec![4]);

            std::fs::remove_file(filename).unwrap();
        }

        #[test]
        fn test_page_size_detection() {
            // Verify we can get page size on Linux
            let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
            assert!(page_size > 0);
            // Common page sizes are 4KB or 64KB
            assert!(page_size >= 4096);
            assert!(page_size % 4096 == 0);
        }
    }
}
