//! High-performance tensor loading with pure Rust implementation.
//!
//! This module provides efficient file loading capabilities for safetensors files,
//! supporting both CPU and GPU (CUDA) targets with optional cuFile acceleration.
//!
//! # Features
//!
//! - **Zero-copy GPU loading**: Load tensors directly to GPU memory via DMA
//! - **cuFile/GDS support**: Direct NVMe→GPU DMA on Linux
//! - **Multi-file support**: Load from multiple sharded files efficiently
//!
//! # Example
//!
//! ```no_run
//! use safetensors::loader::{FileLoader, Device};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Open a safetensors file
//! let loader = FileLoader::open("model.safetensors", Device::Cpu)?;
//!
//! // Fetch a byte range
//! let buffer = loader.fetch(0, 1024)?;
//! println!("Loaded {} bytes", buffer.len());
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::tensor::{Dtype, Metadata};

mod buffer;
mod mmap;

pub use buffer::{Buffer, BufferError};
use mmap::MmapSource;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(all(target_os = "linux", feature = "cufile"))]
mod cufile;

mod scatter;

#[cfg(feature = "cuda")]
pub use self::cuda::{
    cuda_alloc_buffer, cuda_enable_peer_access, cuda_free_host, cuda_host_alloc,
    cuda_memcpy_dtod_async, cuda_memcpy_htod_async, cuda_stream_create, cuda_stream_destroy,
    cuda_stream_sync, set_cuda_device,
};

/// Register a GPU buffer with cuFile for direct DMA (avoids bounce buffers).
#[cfg(all(target_os = "linux", feature = "cufile"))]
#[inline]
pub fn cufile_buf_register(ptr: *const u8, size: usize) -> bool {
    cufile::cufile_buf_register(ptr, size)
}

/// Deregister a GPU buffer from cuFile.
#[cfg(all(target_os = "linux", feature = "cufile"))]
#[inline]
pub fn cufile_buf_deregister(ptr: *const u8) {
    cufile::cufile_buf_deregister(ptr)
}

// Re-export scatter types needed by Python bindings.
pub use scatter::{clear_staging_cache, ShardLoadResult, TensorReady};

/// Target device for loading tensors.
///
/// Currently only CPU and CUDA are supported at the I/O layer. Non-CUDA
/// accelerators (MPS, NPU, XPU, XLA, MLU, MUSA, HPU) are handled in the
/// Python bindings by loading to CPU and then transferring via the
/// framework's `.to(device)` method — matching upstream safetensors behavior.
///
/// TODO: To add native I/O for a new device backend:
///   1. Add a variant here (e.g., `Xpu(usize)`)
///   2. Add a `Backend` variant and implement a new source (like `cufile.rs`)
///   3. Extend `FileLoader::fetch()` / `read_into()` with a dispatch arm
///   4. Extend `Buffer` with an allocation/free path for the device memory
///   5. Wire the new variant through `scatter.rs` for multi-device scatter support
///   6. Map the Python `Device` enum variant to the new `LoaderDevice` variant
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Device {
    /// Load to CPU memory (default).
    #[default]
    Cpu,
    /// Load directly to CUDA GPU memory.
    #[cfg(feature = "cuda")]
    Cuda(usize),
}

impl Device {
    /// Get the CUDA device index, if applicable.
    #[cfg(feature = "cuda")]
    #[inline]
    pub fn cuda_index(self) -> Option<usize> {
        match self {
            Device::Cpu => None,
            Device::Cuda(idx) => Some(idx),
        }
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            #[cfg(feature = "cuda")]
            Self::Cuda(idx) => write!(f, "cuda:{idx}"),
        }
    }
}

/// Loader backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Backend {
    /// Automatically select the best backend for the platform.
    #[default]
    Auto,
    /// Use mmap-based loading (cross-platform).
    Mmap,
    /// io_uring with O_DIRECT (Linux only). Used by the scatter path for
    /// multi-device bulk loading; no effect on single-tensor FileLoader operations.
    #[cfg(all(target_os = "linux", feature = "io_uring"))]
    IoUring,
    /// Use cuFile/GDS for direct NVMe->GPU DMA (Linux + CUDA only).
    #[cfg(all(target_os = "linux", feature = "cufile"))]
    CuFile,
}

/// Resolve a key against a prefix map using longest-prefix-wins semantics.
///
/// Tries exact match first, then walks up the module tree by stripping
/// `.`-separated segments from the right. Matches the transformers `device_map`
/// convention where `"model.layers.0"` matches all tensors under that prefix.
///
/// Returns `None` if no prefix matches (caller provides its own default).
pub fn resolve_prefix_map<'a, V>(
    map: &'a std::collections::HashMap<String, V>,
    key: &str,
) -> Option<&'a V> {
    if let Some(v) = map.get(key) {
        return Some(v);
    }
    let mut end = key.len();
    while let Some(pos) = key[..end].rfind('.') {
        if let Some(v) = map.get(&key[..pos]) {
            return Some(v);
        }
        end = pos;
    }
    // Try root key "" (transformers convention for catch-all)
    map.get("")
}

/// Device mapping for multi-device tensor loading.
///
/// `DeviceMap` determines which device each tensor should be loaded to.
/// This enables distributing large models across multiple GPUs.
///
/// The `Map` variant supports both exact tensor name matching and
/// prefix-based matching (longest-prefix-wins) via [`resolve_prefix_map()`].
///
/// # Example
///
/// ```
/// use safetensors::loader::{Device, DeviceMap};
/// use std::collections::HashMap;
///
/// // Single device (default behavior)
/// let map = DeviceMap::single(Device::Cpu);
///
/// // Map with exact names and/or module prefixes
/// let mut tensor_map = HashMap::new();
/// tensor_map.insert("lm_head.weight".to_string(), Device::Cpu);
/// # #[cfg(feature = "cuda")]
/// # {
/// # // Module prefix: matches all tensors under model.layers.0.*
/// # tensor_map.insert("model.layers.0".to_string(), Device::Cuda(0));
/// # }
/// let map = DeviceMap::from_map(tensor_map, Device::Cpu);
/// ```
#[derive(Debug, Clone)]
pub enum DeviceMap {
    /// All tensors go to a single device.
    Single(Device),
    /// Map tensor names to specific devices, with a default fallback.
    ///
    /// Supports both exact tensor name matching and prefix-based matching
    /// (longest-prefix-wins, transformers-style). See [`resolve_prefix_map()`].
    Map {
        /// Tensor name or module prefix to device mapping.
        map: std::collections::HashMap<String, Device>,
        /// Default device for tensors not in the map.
        default: Device,
    },
}

impl Default for DeviceMap {
    fn default() -> Self {
        DeviceMap::Single(Device::Cpu)
    }
}

impl DeviceMap {
    /// Create a device map that routes all tensors to a single device.
    #[inline]
    pub fn single(device: Device) -> Self {
        DeviceMap::Single(device)
    }

    /// Create a device map from an exact name mapping.
    ///
    /// Tensors not in the map will use the default device.
    pub fn from_map(map: std::collections::HashMap<String, Device>, default: Device) -> Self {
        DeviceMap::Map { map, default }
    }

    /// Resolve which device a tensor should be loaded to.
    pub fn resolve(&self, tensor_name: &str) -> Device {
        match self {
            DeviceMap::Single(device) => *device,
            DeviceMap::Map { map, default } => resolve_prefix_map(map, tensor_name)
                .copied()
                .unwrap_or(*default),
        }
    }

    /// Get the default device for this map.
    pub fn default_device(&self) -> Device {
        match self {
            DeviceMap::Single(device) => *device,
            DeviceMap::Map { default, .. } => *default,
        }
    }

    /// Get all unique devices used in this map.
    pub fn devices(&self) -> Vec<Device> {
        use std::collections::HashSet;
        let mut devices: HashSet<Device> = HashSet::new();

        match self {
            DeviceMap::Single(device) => {
                devices.insert(*device);
            }
            DeviceMap::Map { map, default } => {
                devices.insert(*default);
                for device in map.values() {
                    devices.insert(*device);
                }
            }
        }

        devices.into_iter().collect()
    }

    /// Check if this is a single-device map (no multi-GPU routing).
    #[inline]
    pub fn is_single(&self) -> bool {
        matches!(self, DeviceMap::Single(_))
    }
}

impl From<Device> for DeviceMap {
    fn from(device: Device) -> Self {
        DeviceMap::Single(device)
    }
}

/// Error type for loader operations.
#[derive(Debug)]
pub enum LoaderError {
    /// Failed to open the source file.
    OpenError(String),
    /// Failed to initialize the loader.
    InitError(String),
    /// Failed to fetch data.
    FetchError(String),
    /// Invalid range specified.
    InvalidRange,
}

impl std::fmt::Display for LoaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoaderError::OpenError(msg) => write!(f, "failed to open source: {msg}"),
            LoaderError::InitError(msg) => write!(f, "failed to initialize loader: {msg}"),
            LoaderError::FetchError(msg) => write!(f, "failed to fetch data: {msg}"),
            LoaderError::InvalidRange => write!(f, "invalid byte range"),
        }
    }
}

impl std::error::Error for LoaderError {}

impl From<BufferError> for LoaderError {
    fn from(e: BufferError) -> Self {
        LoaderError::FetchError(e.to_string())
    }
}

/// Result type for loader operations.
pub type Result<T> = std::result::Result<T, LoaderError>;

/// A high-performance loader for safetensors files.
///
/// The `FileLoader` provides efficient access to tensor data from safetensors files,
/// with support for CPU and GPU targets.
///
/// # Thread Safety
///
/// `FileLoader` is `Send + Sync`. All read operations are thread-safe:
/// - mmap reads are just pointer dereference + memcpy
/// - CUDA allocations (cudaMalloc) are thread-safe
/// - cuFile reads (cuFileRead) are thread-safe
///
/// # Example
///
/// ```no_run
/// use safetensors::loader::{FileLoader, Device};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let loader = FileLoader::open("model.safetensors", Device::Cpu)?;
///
/// // Fetch first 1KB
/// let data = loader.fetch(0, 1024)?;
/// # Ok(())
/// # }
/// ```
pub struct FileLoader {
    source: MmapSource,
    device: Device,
    #[allow(dead_code)]
    backend: Backend,
    file_size: usize,
    #[cfg(all(target_os = "linux", feature = "cufile"))]
    cufile_source: Option<cufile::CuFileSource>,
}

impl FileLoader {
    /// Open a safetensors file for loading.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the safetensors file
    /// * `device` - Target device (CPU or CUDA)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use safetensors::loader::{FileLoader, Device};
    ///
    /// let loader = FileLoader::open("model.safetensors", Device::Cpu)?;
    /// # Ok::<(), safetensors::loader::LoaderError>(())
    /// ```
    pub fn open<P: AsRef<Path>>(path: P, device: Device) -> Result<Self> {
        Self::with_backend(path, device, Backend::Auto)
    }

    /// Open a safetensors file with a specific backend.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the safetensors file
    /// * `device` - Target device (CPU or CUDA)
    /// * `backend` - Loader backend to use
    pub fn with_backend<P: AsRef<Path>>(path: P, device: Device, backend: Backend) -> Result<Self> {
        #[cfg(all(target_os = "linux", feature = "io_uring"))]
        if matches!(backend, Backend::IoUring) {
            return Err(LoaderError::InitError(
                "backend `io_uring` is not supported for single-tensor FileLoader operations; \
                 io_uring is used automatically through the scatter path for multi-device loading"
                    .into(),
            ));
        }

        #[cfg(all(target_os = "linux", feature = "cufile"))]
        if matches!(backend, Backend::CuFile) && matches!(device, Device::Cpu) {
            return Err(LoaderError::InitError(
                "backend `cufile` requires a CUDA device".into(),
            ));
        }

        let source =
            MmapSource::open(path.as_ref()).map_err(|e| LoaderError::OpenError(format!("{e}")))?;
        let file_size = source.size();

        #[cfg(feature = "cuda")]
        if let Device::Cuda(device_id) = device {
            cuda::set_cuda_device(device_id)?;
        }

        #[cfg(all(target_os = "linux", feature = "cufile"))]
        let cufile_source = {
            let resolved = resolve_backend(backend, device);
            if matches!(resolved, Backend::CuFile) {
                Some(
                    cufile::CuFileSource::open(source.raw_fd())
                        .map_err(|e| LoaderError::InitError(format!("cuFile init: {e}")))?,
                )
            } else {
                None
            }
        };

        Ok(Self {
            source,
            device,
            backend,
            file_size,
            #[cfg(all(target_os = "linux", feature = "cufile"))]
            cufile_source,
        })
    }

    /// Fetch a range of bytes from the file.
    ///
    /// Returns a `Buffer` containing the data. For CPU targets, this is
    /// host memory. For CUDA targets, this is device memory.
    ///
    /// # Arguments
    ///
    /// * `start` - Start offset in bytes
    /// * `end` - End offset in bytes (exclusive)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use safetensors::loader::{FileLoader, Device};
    /// # let loader = FileLoader::open("model.safetensors", Device::Cpu)?;
    /// // Fetch bytes 1000-2000
    /// let buffer = loader.fetch(1000, 2000)?;
    /// assert_eq!(buffer.len(), 1000);
    /// # Ok::<(), safetensors::loader::LoaderError>(())
    /// ```
    pub fn fetch(&self, start: usize, end: usize) -> Result<Buffer> {
        if start > end {
            return Err(LoaderError::InvalidRange);
        }
        if start == end {
            return Ok(Buffer::empty(self.device));
        }
        let size = end - start;

        match self.device {
            Device::Cpu => {
                let buf = Buffer::alloc_cpu(size)?;
                unsafe {
                    libc::madvise(
                        self.source.content_ptr().add(start) as *mut libc::c_void,
                        size,
                        libc::MADV_WILLNEED | libc::MADV_SEQUENTIAL,
                    );
                    std::ptr::copy_nonoverlapping(
                        self.source.content_ptr().add(start),
                        buf.as_ptr() as *mut u8,
                        size,
                    );
                }
                Ok(buf)
            }
            #[cfg(feature = "cuda")]
            Device::Cuda(device_id) => {
                cuda::set_cuda_device(device_id)?;

                #[cfg(all(target_os = "linux", feature = "cufile"))]
                if let Some(ref cufile_src) = self.cufile_source {
                    let ptr = cuda::cuda_malloc(size)?;
                    match cufile_src.read(ptr, size, start, 0) {
                        Ok(_) => {
                            return Ok(unsafe { Buffer::from_raw_cuda(ptr, size, self.device) });
                        }
                        Err(e) => {
                            cuda::cuda_free(ptr);
                            return Err(e);
                        }
                    }
                }

                // Fallback: mmap + cudaMemcpy
                let ptr = cuda::cuda_malloc(size)?;
                match cuda::cuda_memcpy_htod(
                    ptr,
                    unsafe { self.source.content_ptr().add(start) },
                    size,
                ) {
                    Ok(()) => Ok(unsafe { Buffer::from_raw_cuda(ptr, size, self.device) }),
                    Err(e) => {
                        cuda::cuda_free(ptr);
                        Err(e)
                    }
                }
            }
        }
    }

    /// Fetch a zero-copy view of a byte range from the file.
    ///
    /// This returns a `Buffer` that points directly into the mmap'd file
    /// without any memory allocation or copying. The buffer can safely
    /// outlive the `FileLoader` due to Arc-based reference counting.
    ///
    /// **Note**: This only works for CPU targets.
    /// For CUDA targets, use `fetch()` instead.
    ///
    /// # Arguments
    ///
    /// * `start` - Start offset in bytes
    /// * `end` - End offset in bytes (exclusive)
    pub fn fetch_view(&self, start: usize, end: usize) -> Result<Buffer> {
        if start > end {
            return Err(LoaderError::InvalidRange);
        }
        if start == end {
            return Ok(Buffer::empty(Device::Cpu));
        }

        if self.device != Device::Cpu {
            return Err(LoaderError::FetchError(
                "fetch_view only supported for CPU device".into(),
            ));
        }

        let size = end - start;
        let ptr = unsafe { self.source.content_ptr().add(start) };
        let mmap = self.source.mmap_arc();

        Ok(unsafe { Buffer::from_mmap_view(ptr, size, mmap) })
    }

    /// Fetch a range of bytes and copy to a Vec.
    ///
    /// This is a convenience method that fetches data and copies it to
    /// owned memory. For CPU targets, this copies directly from mmap
    /// to a Vec (single copy, no intermediate buffer).
    pub fn fetch_to_vec(&self, start: usize, end: usize) -> Result<Vec<u8>> {
        if start > end {
            return Err(LoaderError::InvalidRange);
        }
        if start == end {
            return Ok(Vec::new());
        }
        match self.device {
            Device::Cpu => {
                let size = end - start;
                let src = unsafe {
                    std::slice::from_raw_parts(self.source.content_ptr().add(start), size)
                };
                Ok(src.to_vec())
            }
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => {
                let buffer = self.fetch(start, end)?;
                buffer.to_vec().map_err(Into::into)
            }
        }
    }

    /// Allocate a buffer for a byte range without doing any I/O.
    ///
    /// Use `read_into()` to later populate the buffer with data.
    /// This allows separating allocation (sequential) from I/O (parallel).
    pub fn alloc_buffer(&self, start: usize, end: usize) -> Result<Buffer> {
        if start > end {
            return Err(LoaderError::InvalidRange);
        }
        if start == end {
            return Ok(Buffer::empty(self.device));
        }
        let size = end - start;

        match self.device {
            Device::Cpu => Buffer::alloc_cpu(size),
            #[cfg(feature = "cuda")]
            Device::Cuda(device_id) => {
                cuda::set_cuda_device(device_id)?;
                let ptr = cuda::cuda_malloc(size)?;
                Ok(unsafe { Buffer::from_raw_cuda(ptr, size, self.device) })
            }
        }
    }

    /// Read data into a pre-allocated buffer.
    ///
    /// The buffer must have been allocated via `alloc_buffer()` or equivalent.
    pub fn read_into(&self, buffer: &Buffer, file_offset: usize) -> Result<usize> {
        if buffer.is_empty() {
            return Ok(0);
        }

        match buffer.device() {
            Device::Cpu => {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        self.source.content_ptr().add(file_offset),
                        buffer.as_ptr() as *mut u8,
                        buffer.len(),
                    );
                }
                Ok(buffer.len())
            }
            #[cfg(feature = "cuda")]
            Device::Cuda(_device_id) => {
                // Check if cuFile backend is available
                #[cfg(all(target_os = "linux", feature = "cufile"))]
                if let Some(ref cufile_src) = self.cufile_source {
                    return cufile_src.read(
                        buffer.as_ptr() as *mut u8,
                        buffer.len(),
                        file_offset,
                        0,
                    );
                }

                // Fallback: cudaMemcpy from mmap
                cuda::cuda_memcpy_htod(
                    buffer.as_ptr() as *mut u8,
                    unsafe { self.source.content_ptr().add(file_offset) },
                    buffer.len(),
                )?;
                Ok(buffer.len())
            }
        }
    }

    /// Get the target device for this loader.
    #[inline]
    pub fn device(&self) -> Device {
        self.device
    }

    /// Get the size of the source file in bytes.
    #[inline]
    pub fn file_size(&self) -> usize {
        self.file_size
    }

    /// Set the active CUDA device for subsequent fetch operations.
    #[cfg(feature = "cuda")]
    #[inline]
    pub fn set_cuda_device(&self, device_index: usize) -> Result<()> {
        cuda::set_cuda_device(device_index)
    }

    /// Get the mmap content pointer.
    #[allow(dead_code)]
    pub(crate) fn source_content_ptr(&self) -> *const u8 {
        self.source.content_ptr()
    }
}

/// Resolve Backend::Auto to a concrete backend.
///
/// For single-tensor FileLoader operations (fetch/read_into), the only accelerated
/// backend is cuFile. io_uring is used exclusively through the scatter path
/// for multi-device bulk loading.
#[allow(dead_code, unused_variables)] // used inside cfg(cufile) blocks
fn resolve_backend(backend: Backend, device: Device) -> Backend {
    match backend {
        Backend::Auto => {
            #[cfg(all(target_os = "linux", feature = "cufile"))]
            if matches!(device, Device::Cuda(_)) {
                return Backend::CuFile;
            }
            Backend::Mmap
        }
        other => other,
    }
}

/// Information about a tensor to be loaded.
#[derive(Debug, Clone)]
pub struct TensorLoadInfo {
    /// Name of the tensor.
    pub name: String,
    /// Start offset in the file.
    pub start: usize,
    /// End offset in the file (exclusive).
    pub end: usize,
}

impl TensorLoadInfo {
    /// Create new tensor load info.
    pub fn new(name: impl Into<String>, start: usize, end: usize) -> Self {
        Self {
            name: name.into(),
            start,
            end,
        }
    }

    /// Size of the tensor in bytes.
    #[inline]
    pub fn size(&self) -> usize {
        self.end.saturating_sub(self.start)
    }
}

impl FileLoader {
    /// Fetch multiple tensors in a single batched operation.
    ///
    /// For mmap backends, this is equivalent to calling `fetch` in a loop.
    /// Returns `(name, Buffer)` pairs in the same order as input.
    pub fn fetch_batch(&self, tensors: &[TensorLoadInfo]) -> Result<Vec<(String, Buffer)>> {
        let mut results = Vec::with_capacity(tensors.len());
        for info in tensors {
            let buf = self.fetch(info.start, info.end)?;
            results.push((info.name.clone(), buf));
        }
        Ok(results)
    }
}

// ─── Unified Loader ──────────────────────────────────────────────────────

/// Metadata for a single tensor within a model.
#[derive(Debug, Clone)]
pub struct TensorMeta {
    /// Data type of the tensor elements.
    pub dtype: Dtype,
    /// Shape of the tensor.
    pub shape: Vec<usize>,
    /// Original byte range within the shard's data section.
    pub data_offsets: (usize, usize),
    /// Byte range to actually load (may differ for TP colwise slicing).
    pub load_offsets: (usize, usize),
}

/// Reference to tensor data from the loader, avoiding unnecessary copies.
pub enum BufferRef {
    /// Exact-sized buffer for this tensor (io_uring scatter or fetched).
    Whole(Arc<Buffer>),
    /// Slice into a larger preloaded buffer (per-device or bulk preload).
    Slice {
        /// The parent buffer.
        buffer: Arc<Buffer>,
        /// Byte offset into the parent buffer.
        offset: usize,
        /// Number of bytes.
        len: usize,
    },
    /// On-demand fetch from mmap or CUDA loader.
    Fetched(Buffer),
}

impl BufferRef {
    /// Get the length of the referenced data.
    pub fn len(&self) -> usize {
        match self {
            BufferRef::Whole(buf) => buf.len(),
            BufferRef::Slice { len, .. } => *len,
            BufferRef::Fetched(buf) => buf.len(),
        }
    }

    /// Check if the referenced data is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Per-shard state within a Loader.
#[allow(dead_code)] // fields used once Python bindings switch to Loader
struct ShardState {
    loader: Arc<FileLoader>,
    data_offset: usize,
    path: PathBuf,
    preloaded: Option<Arc<Buffer>>,
    per_device: Option<HashMap<usize, Arc<Buffer>>>,
    per_tensor: Option<HashMap<(usize, usize), Arc<Buffer>>>,
}

/// A unified loader for safetensors models (single or sharded, any backend).
///
/// Replaces the separate `FileLoader` + `scatter_load` API with a single type
/// that handles all loading strategies internally. Use [`LoaderBuilder`] to
/// construct.
pub struct Loader {
    shards: Vec<ShardState>,
    tensor_index: HashMap<String, (usize, TensorMeta)>,
    device_map: DeviceMap,
    backend: Backend,
}

/// Builder for constructing a [`Loader`].
pub struct LoaderBuilder {
    paths: Vec<PathBuf>,
    shard_info: Vec<(usize, Metadata)>,
    device_map: DeviceMap,
    backend: Backend,
    byte_range_overrides: HashMap<String, (usize, usize)>,
}

impl LoaderBuilder {
    /// Create a builder for a single safetensors file.
    pub fn single(path: PathBuf, data_offset: usize, metadata: Metadata) -> Self {
        Self {
            paths: vec![path],
            shard_info: vec![(data_offset, metadata)],
            device_map: DeviceMap::default(),
            backend: Backend::Auto,
            byte_range_overrides: HashMap::new(),
        }
    }

    /// Create a builder for multiple sharded safetensors files.
    pub fn sharded(paths: Vec<PathBuf>, shard_info: Vec<(usize, Metadata)>) -> Self {
        Self {
            paths,
            shard_info,
            device_map: DeviceMap::default(),
            backend: Backend::Auto,
            byte_range_overrides: HashMap::new(),
        }
    }

    /// Set the device map for tensor routing.
    pub fn device_map(mut self, map: DeviceMap) -> Self {
        self.device_map = map;
        self
    }

    /// Set the I/O backend.
    pub fn backend(mut self, backend: Backend) -> Self {
        self.backend = backend;
        self
    }

    /// Set per-tensor byte range overrides (for TP colwise slicing).
    pub fn byte_range_overrides(mut self, overrides: HashMap<String, (usize, usize)>) -> Self {
        self.byte_range_overrides = overrides;
        self
    }

    /// Build the Loader. Performs all I/O (scatter, preload, mmap open).
    pub fn build(self) -> Result<Loader> {
        // Build tensor_index from shard metadata
        let mut tensor_index: HashMap<String, (usize, TensorMeta)> = HashMap::new();
        for (shard_idx, (_, metadata)) in self.shard_info.iter().enumerate() {
            for (name, info) in metadata.tensors() {
                let load_offsets = self
                    .byte_range_overrides
                    .get(&name)
                    .copied()
                    .unwrap_or(info.data_offsets);
                tensor_index.insert(
                    name,
                    (
                        shard_idx,
                        TensorMeta {
                            dtype: info.dtype,
                            shape: info.shape.clone(),
                            data_offsets: info.data_offsets,
                            load_offsets,
                        },
                    ),
                );
            }
        }

        // Determine multi vs single device and CUDA presence
        let devices = self.device_map.devices();
        let is_single = self.device_map.is_single() || devices.len() <= 1;

        #[cfg(feature = "cuda")]
        let is_cuda = devices.iter().any(|d| matches!(d, Device::Cuda(_)));
        #[cfg(not(feature = "cuda"))]
        let is_cuda = false;

        // Validate backend
        self.validate_backend(is_single, is_cuda)?;

        // Compute dev_ranges for multi-device CUDA
        let shard_dev_ranges: Vec<HashMap<usize, Vec<(usize, usize)>>> = if !is_single && is_cuda {
            self.compute_dev_ranges(&tensor_index)
        } else {
            vec![HashMap::new(); self.paths.len()]
        };

        // Build scatter specs and config
        let shard_offsets: Vec<usize> = self.shard_info.iter().map(|(off, _)| *off).collect();
        let specs: Vec<scatter::ShardSpec> = self
            .paths
            .iter()
            .zip(shard_offsets.iter())
            .zip(shard_dev_ranges)
            .map(|((path, &offset), dev_ranges)| scatter::ShardSpec {
                path: path.clone(),
                offset,
                dev_ranges,
            })
            .collect();

        // Determine loader device: use CUDA if any tensor targets CUDA.
        // The scatter paths need a CUDA device to dispatch correctly;
        // individual tensors still route to their specific GPU.
        #[cfg(feature = "cuda")]
        let loader_device = {
            devices
                .iter()
                .find_map(|d| d.cuda_index())
                .map(Device::Cuda)
                .unwrap_or(Device::Cpu)
        };
        #[cfg(not(feature = "cuda"))]
        let loader_device = Device::Cpu;

        let scatter_config = scatter::ScatterConfig {
            backend: self.backend,
            device: loader_device,
            is_single,
        };

        // Dispatch loading
        let shard_results =
            scatter::scatter_load(&specs, &self.paths, &shard_offsets, &scatter_config)?;

        // Convert to ShardState
        let shards: Vec<ShardState> = shard_results
            .into_iter()
            .enumerate()
            .map(|(i, result)| ShardState {
                loader: result.loader,
                data_offset: shard_offsets[i],
                path: self.paths[i].clone(),
                preloaded: result.preloaded,
                per_device: result.per_device,
                per_tensor: result.per_tensor,
            })
            .collect();

        Ok(Loader {
            shards,
            tensor_index,
            device_map: self.device_map,
            backend: self.backend,
        })
    }

    fn validate_backend(&self, is_single: bool, is_cuda: bool) -> Result<()> {
        // Suppress unused warnings when cfg features are disabled.
        let _ = (is_single, is_cuda);
        match self.backend {
            Backend::Auto | Backend::Mmap => Ok(()),

            #[cfg(all(target_os = "linux", feature = "io_uring"))]
            Backend::IoUring => {
                if is_single {
                    return Err(LoaderError::InitError(
                        "backend `io_uring` requires multi-device CUDA loading".into(),
                    ));
                }
                if !is_cuda {
                    return Err(LoaderError::InitError(
                        "backend `io_uring` requires CUDA devices".into(),
                    ));
                }
                Ok(())
            }

            #[cfg(all(target_os = "linux", feature = "cufile"))]
            Backend::CuFile => {
                if !is_cuda {
                    return Err(LoaderError::InitError(
                        "backend `cufile` requires CUDA devices".into(),
                    ));
                }
                Ok(())
            }
        }
    }

    fn compute_dev_ranges(
        &self,
        tensor_index: &HashMap<String, (usize, TensorMeta)>,
    ) -> Vec<HashMap<usize, Vec<(usize, usize)>>> {
        let mut shard_dev_ranges: Vec<HashMap<usize, Vec<(usize, usize)>>> =
            vec![HashMap::new(); self.paths.len()];

        for (_name, (_shard_idx, _meta)) in tensor_index {
            #[cfg(feature = "cuda")]
            if let Device::Cuda(dev_idx) = self.device_map.resolve(_name) {
                shard_dev_ranges[*_shard_idx]
                    .entry(dev_idx)
                    .or_default()
                    .push(_meta.load_offsets);
            }
        }

        for ranges_map in &mut shard_dev_ranges {
            for ranges in ranges_map.values_mut() {
                ranges.sort_by_key(|&(start, _)| start);
            }
        }

        shard_dev_ranges
    }
}

impl Loader {
    /// Get a buffer for a tensor.
    ///
    /// Handles the per_tensor → per_device → preloaded → fetch cascade.
    pub fn get_buffer(&self, name: &str) -> Result<BufferRef> {
        let (shard_idx, meta) = self
            .tensor_index
            .get(name)
            .ok_or_else(|| LoaderError::FetchError(format!("tensor not found: {name}")))?;

        let shard = &self.shards[*shard_idx];
        let load_start = meta.load_offsets.0;
        let load_len = meta.load_offsets.1 - meta.load_offsets.0;
        let file_start = load_start + shard.data_offset;
        let file_end = file_start + load_len;

        #[cfg(feature = "cuda")]
        {
            let device = self.device_map.resolve(name);
            if let Device::Cuda(target_idx) = device {
                // Fast path 1: per-tensor buffer (io_uring scatter)
                if let Some(ref per_tensor) = shard.per_tensor {
                    if let Some(buffer) = per_tensor.get(&(load_start, target_idx)) {
                        return Ok(BufferRef::Whole(buffer.clone()));
                    }
                }
                // Fast path 2: per-device buffer (cuFile P2P scatter)
                if let Some(ref per_device) = shard.per_device {
                    if let Some(buffer) = per_device.get(&target_idx) {
                        return Ok(BufferRef::Slice {
                            buffer: buffer.clone(),
                            offset: load_start,
                            len: load_len,
                        });
                    }
                }
                // Fast path 3: single-device preloaded buffer
                if let Some(ref preloaded) = shard.preloaded {
                    return Ok(BufferRef::Slice {
                        buffer: preloaded.clone(),
                        offset: load_start,
                        len: load_len,
                    });
                }
            }
        }

        // Fallback: on-demand fetch from mmap (CPU) or CUDA loader
        let buffer = shard.loader.fetch(file_start, file_end)?;
        Ok(BufferRef::Fetched(buffer))
    }

    /// Get tensor metadata (dtype, shape, offsets).
    pub fn tensor_meta(&self, name: &str) -> Option<&TensorMeta> {
        self.tensor_index.get(name).map(|(_, meta)| meta)
    }

    /// Iterate tensor names (sorted by file offset within each shard).
    pub fn tensor_names(&self) -> Vec<&str> {
        let mut names: Vec<(&str, usize, usize)> = self
            .tensor_index
            .iter()
            .map(|(name, (si, meta))| (name.as_str(), *si, meta.load_offsets.0))
            .collect();
        names.sort_by_key(|&(_, si, off)| (si, off));
        names.into_iter().map(|(name, _, _)| name).collect()
    }

    /// Number of tensors.
    pub fn len(&self) -> usize {
        self.tensor_index.len()
    }

    /// Check if the loader has no tensors.
    pub fn is_empty(&self) -> bool {
        self.tensor_index.is_empty()
    }

    /// Get the device a tensor is mapped to.
    pub fn tensor_device(&self, name: &str) -> Option<Device> {
        if self.tensor_index.contains_key(name) {
            Some(self.device_map.resolve(name))
        } else {
            None
        }
    }

    /// Get a reference to the device map.
    pub fn device_map(&self) -> &DeviceMap {
        &self.device_map
    }

    /// Get the shard loader for a tensor (for direct fetch operations).
    pub fn shard_loader(&self, name: &str) -> Option<&Arc<FileLoader>> {
        self.tensor_index
            .get(name)
            .map(|(si, _)| &self.shards[*si].loader)
    }

    /// Get the data offset for the shard containing a tensor.
    pub fn shard_offset(&self, name: &str) -> Option<usize> {
        self.tensor_index
            .get(name)
            .map(|(si, _)| self.shards[*si].data_offset)
    }

    /// Get the shard index for a tensor.
    pub fn shard_index(&self, name: &str) -> Option<usize> {
        self.tensor_index.get(name).map(|(si, _)| *si)
    }

    /// Zero-copy view into mmap (CPU only).
    pub fn fetch_view(&self, name: &str) -> Result<Buffer> {
        let (shard_idx, meta) = self
            .tensor_index
            .get(name)
            .ok_or_else(|| LoaderError::FetchError(format!("tensor not found: {name}")))?;
        let shard = &self.shards[*shard_idx];
        let start = meta.load_offsets.0 + shard.data_offset;
        let end = meta.load_offsets.1 + shard.data_offset;
        shard.loader.fetch_view(start, end)
    }

    /// Fetch tensor data to Vec.
    pub fn fetch_to_vec(&self, name: &str) -> Result<Vec<u8>> {
        let (shard_idx, meta) = self
            .tensor_index
            .get(name)
            .ok_or_else(|| LoaderError::FetchError(format!("tensor not found: {name}")))?;
        let shard = &self.shards[*shard_idx];
        let start = meta.load_offsets.0 + shard.data_offset;
        let end = meta.load_offsets.1 + shard.data_offset;
        shard.loader.fetch_to_vec(start, end)
    }

    /// Start streaming iteration over tensors via io_uring scatter.
    ///
    /// Returns `Some((receiver, handle))` if the loader is configured for
    /// multi-device CUDA with io_uring. Returns `None` for other backends.
    ///
    /// The receiver yields `TensorReady` results as each tensor completes its
    /// GPU transfer. The handle can be joined to get the `ShardLoadResult`s.
    #[cfg(all(target_os = "linux", feature = "io_uring", feature = "cuda"))]
    pub fn iter_start(
        &self,
        prefetch_count: usize,
    ) -> Option<(
        std::sync::mpsc::Receiver<std::result::Result<scatter::TensorReady, String>>,
        std::thread::JoinHandle<Result<Vec<scatter::ShardLoadResult>>>,
    )> {
        // Only supported for multi-device CUDA with io_uring or Auto backend.
        // Requires at least 2 distinct CUDA devices in the device map.
        let is_iouring = matches!(self.backend, Backend::IoUring | Backend::Auto);
        let cuda_device_count = self
            .device_map
            .devices()
            .iter()
            .filter(|d| d.cuda_index().is_some())
            .count();
        if !is_iouring || cuda_device_count < 2 {
            return None;
        }

        // Reconstruct ShardSpec from stored state
        let specs: Vec<scatter::ShardSpec> = self
            .shards
            .iter()
            .enumerate()
            .map(|(shard_idx, shard)| {
                let mut dev_ranges: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
                for (name, (si, meta)) in &self.tensor_index {
                    if *si != shard_idx {
                        continue;
                    }
                    if let Device::Cuda(dev_idx) = self.device_map.resolve(name) {
                        dev_ranges
                            .entry(dev_idx)
                            .or_default()
                            .push(meta.load_offsets);
                    }
                }
                for ranges in dev_ranges.values_mut() {
                    ranges.sort_by_key(|&(start, _)| start);
                }
                scatter::ShardSpec {
                    path: shard.path.clone(),
                    offset: shard.data_offset,
                    dev_ranges,
                }
            })
            .collect();

        let paths: Vec<PathBuf> = self.shards.iter().map(|s| s.path.clone()).collect();
        let offsets: Vec<usize> = self.shards.iter().map(|s| s.data_offset).collect();

        let loader_device = self
            .device_map
            .devices()
            .into_iter()
            .find_map(|d| d.cuda_index().map(Device::Cuda))
            .unwrap_or(Device::Cpu);

        let config = scatter::ScatterConfig {
            backend: self.backend,
            device: loader_device,
            is_single: false,
        };

        Some(scatter::scatter_iouring_start(
            specs,
            paths,
            offsets,
            config,
            prefetch_count,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn create_test_file(content: &[u8]) -> tempfile::NamedTempFile {
        let mut file = tempfile::NamedTempFile::new().expect("Failed to create temp file");
        file.write_all(content).expect("Failed to write");
        file.flush().expect("Failed to flush");
        file
    }

    #[test]
    fn test_device_default() {
        assert_eq!(Device::default(), Device::Cpu);
    }

    #[test]
    fn test_backend_default() {
        assert_eq!(Backend::default(), Backend::Auto);
    }

    #[test]
    fn test_loader_open_and_fetch() {
        let content = b"Hello, safetensors loader!";
        let temp_file = create_test_file(content);

        let loader = FileLoader::open(temp_file.path(), Device::Cpu).expect("Failed to open");
        assert_eq!(loader.device(), Device::Cpu);
        assert_eq!(loader.file_size(), content.len());

        let buffer = loader.fetch(0, content.len()).expect("Failed to fetch");
        assert_eq!(buffer.len(), content.len());
        assert_eq!(buffer.as_slice().unwrap(), content);
    }

    #[test]
    fn test_loader_partial_fetch() {
        let content = b"0123456789ABCDEFGHIJ";
        let temp_file = create_test_file(content);

        let loader = FileLoader::open(temp_file.path(), Device::Cpu).expect("Failed to open");

        let buffer = loader.fetch(5, 15).expect("Failed to fetch");
        assert_eq!(buffer.len(), 10);
        assert_eq!(buffer.as_slice().unwrap(), b"56789ABCDE");
    }

    #[test]
    fn test_loader_empty_fetch() {
        let content = b"Some content";
        let temp_file = create_test_file(content);

        let loader = FileLoader::open(temp_file.path(), Device::Cpu).expect("Failed to open");

        let buffer = loader.fetch(5, 5).expect("Failed to fetch empty range");
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_loader_invalid_range() {
        let content = b"Some content";
        let temp_file = create_test_file(content);

        let loader = FileLoader::open(temp_file.path(), Device::Cpu).expect("Failed to open");

        let result = loader.fetch(10, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_loader_fetch_to_vec() {
        let content = b"Vector conversion test";
        let temp_file = create_test_file(content);

        let loader = FileLoader::open(temp_file.path(), Device::Cpu).expect("Failed to open");

        let vec = loader
            .fetch_to_vec(0, content.len())
            .expect("Failed to fetch");
        assert_eq!(vec, content.to_vec());
    }

    #[test]
    fn test_tensor_load_info() {
        let info = TensorLoadInfo::new("test", 100, 500);
        assert_eq!(info.name, "test");
        assert_eq!(info.start, 100);
        assert_eq!(info.end, 500);
        assert_eq!(info.size(), 400);
    }

    #[test]
    fn test_loader_with_backend_mmap() {
        let content = b"Hello from explicit mmap backend!";
        let temp_file = create_test_file(content);

        let loader = FileLoader::with_backend(temp_file.path(), Device::Cpu, Backend::Mmap)
            .expect("Failed to open with Mmap");
        assert_eq!(loader.device(), Device::Cpu);
        assert_eq!(loader.file_size(), content.len());

        let buffer = loader.fetch(0, content.len()).expect("Failed to fetch");
        assert_eq!(buffer.as_slice().unwrap(), content);
    }

    #[test]
    fn test_loader_fetch_view_basic() {
        let content = b"Zero-copy mmap view test content!";
        let temp_file = create_test_file(content);

        let loader = FileLoader::open(temp_file.path(), Device::Cpu).expect("Failed to open");

        let view = loader
            .fetch_view(0, content.len())
            .expect("fetch_view failed");
        assert_eq!(view.len(), content.len());
        assert_eq!(view.as_slice().unwrap(), content);
    }

    #[test]
    fn test_loader_fetch_view_partial() {
        let content = b"0123456789ABCDEFGHIJ";
        let temp_file = create_test_file(content);

        let loader = FileLoader::open(temp_file.path(), Device::Cpu).expect("Failed to open");

        let view = loader.fetch_view(5, 15).expect("fetch_view failed");
        assert_eq!(view.len(), 10);
        assert_eq!(view.as_slice().unwrap(), b"56789ABCDE");
    }

    #[test]
    fn test_loader_fetch_view_empty() {
        let content = b"Some content";
        let temp_file = create_test_file(content);

        let loader = FileLoader::open(temp_file.path(), Device::Cpu).expect("Failed to open");

        let view = loader
            .fetch_view(5, 5)
            .expect("fetch_view empty should work");
        assert!(view.is_empty());
    }

    #[test]
    fn test_loader_fetch_view_invalid_range() {
        let content = b"Some content";
        let temp_file = create_test_file(content);

        let loader = FileLoader::open(temp_file.path(), Device::Cpu).expect("Failed to open");

        let result = loader.fetch_view(10, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_loader_alloc_buffer_and_read_into() {
        let content = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let temp_file = create_test_file(content);

        let loader = FileLoader::open(temp_file.path(), Device::Cpu).expect("Failed to open");

        let buffer = loader.alloc_buffer(5, 15).expect("alloc_buffer failed");
        assert_eq!(buffer.len(), 10);

        let bytes_read = loader.read_into(&buffer, 5).expect("read_into failed");
        assert_eq!(bytes_read, 10);
        assert_eq!(buffer.as_slice().unwrap(), b"FGHIJKLMNO");
    }

    #[test]
    fn test_loader_alloc_buffer_empty() {
        let content = b"Some content";
        let temp_file = create_test_file(content);

        let loader = FileLoader::open(temp_file.path(), Device::Cpu).expect("Failed to open");

        let buffer = loader
            .alloc_buffer(5, 5)
            .expect("alloc_buffer empty should work");
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_loader_alloc_buffer_invalid_range() {
        let content = b"Some content";
        let temp_file = create_test_file(content);

        let loader = FileLoader::open(temp_file.path(), Device::Cpu).expect("Failed to open");

        let result = loader.alloc_buffer(10, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_loader_fetch_batch_basic() {
        let content = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij";
        let temp_file = create_test_file(content);

        let loader = FileLoader::open(temp_file.path(), Device::Cpu).expect("Failed to open");

        let tensors = vec![
            TensorLoadInfo::new("a", 0, 10),
            TensorLoadInfo::new("b", 10, 20),
            TensorLoadInfo::new("c", 20, 36),
        ];

        let results = loader.fetch_batch(&tensors).expect("fetch_batch failed");
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, "a");
        assert_eq!(results[0].1.as_slice().unwrap(), b"0123456789");
        assert_eq!(results[1].0, "b");
        assert_eq!(results[1].1.as_slice().unwrap(), b"ABCDEFGHIJ");
        assert_eq!(results[2].0, "c");
        assert_eq!(results[2].1.as_slice().unwrap(), b"KLMNOPQRSTUVWXYZ");
    }

    #[test]
    fn test_loader_fetch_batch_empty() {
        let content = b"Some content";
        let temp_file = create_test_file(content);

        let loader = FileLoader::open(temp_file.path(), Device::Cpu).expect("Failed to open");

        let results = loader
            .fetch_batch(&[])
            .expect("fetch_batch empty should work");
        assert!(results.is_empty());
    }

    #[test]
    fn test_loader_fetch_batch_single() {
        let content = b"Single tensor content here!";
        let temp_file = create_test_file(content);

        let loader = FileLoader::open(temp_file.path(), Device::Cpu).expect("Failed to open");

        let tensors = vec![TensorLoadInfo::new("only", 0, content.len())];

        let results = loader.fetch_batch(&tensors).expect("fetch_batch failed");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "only");
        assert_eq!(results[0].1.as_slice().unwrap(), content);
    }

    #[test]
    fn test_loader_concurrent_fetch() {
        let content: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        let temp_file = create_test_file(&content);
        let path = temp_file.path().to_path_buf();

        let loader = std::sync::Arc::new(FileLoader::open(&path, Device::Cpu).expect("Failed to open"));

        let handles: Vec<_> = (0..4)
            .map(|i| {
                let loader = loader.clone();
                std::thread::spawn(move || {
                    let start = i * 256;
                    let end = start + 256;
                    let buf = loader.fetch(start, end).expect("Concurrent fetch failed");
                    let expected: Vec<u8> = (start..end).map(|j| (j % 256) as u8).collect();
                    assert_eq!(buf.as_slice().unwrap(), &expected[..]);
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread panicked");
        }
    }

    // ── Integration tests with real safetensors data ─────────────────────────

    fn create_safetensors_file(
        n_tensors: usize,
        shape: &[usize],
    ) -> (tempfile::NamedTempFile, Vec<(String, usize, usize)>) {
        use crate::tensor::{serialize, Dtype, TensorView};
        use std::collections::HashMap;

        let dtype = Dtype::F32;
        let tensor_size: usize = shape.iter().product::<usize>() * 4;
        let data: Vec<u8> = (0..tensor_size).map(|i| (i % 251) as u8).collect();

        let mut metadata: HashMap<String, TensorView> = HashMap::new();
        for i in 0..n_tensors {
            let tensor = TensorView::new(dtype, shape.to_vec(), &data).unwrap();
            metadata.insert(format!("layer{i}.weight"), tensor);
        }

        let serialized = serialize(&metadata, None).unwrap();

        let st = crate::SafeTensors::deserialize(&serialized).unwrap();

        let mut info_with_offsets: Vec<(String, usize, usize)> = Vec::new();
        for (name, tv) in st.tensors() {
            let data_ptr = tv.data().as_ptr() as usize;
            let base_ptr = serialized.as_ptr() as usize;
            let file_start = data_ptr - base_ptr;
            let file_end = file_start + tv.data().len();
            info_with_offsets.push((name, file_start, file_end));
        }
        info_with_offsets.sort_by_key(|(_, start, _)| *start);

        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&serialized).unwrap();
        file.flush().unwrap();

        (file, info_with_offsets)
    }

    #[test]
    fn test_integration_safetensors_fetch() {
        let (file, tensors) = create_safetensors_file(5, &[100, 100]);
        let loader = FileLoader::open(file.path(), Device::Cpu).expect("Failed to open");

        for (name, start, end) in &tensors {
            let buf = loader
                .fetch(*start, *end)
                .expect(&format!("fetch {} failed", name));
            assert_eq!(buf.len(), end - start);
            let slice = buf.as_slice().unwrap();
            for (i, &byte) in slice.iter().enumerate() {
                assert_eq!(
                    byte,
                    (i % 251) as u8,
                    "Mismatch at byte {} of tensor {}",
                    i,
                    name
                );
            }
        }
    }

    #[test]
    fn test_integration_safetensors_fetch_view() {
        let (file, tensors) = create_safetensors_file(3, &[50, 50]);
        let loader = FileLoader::open(file.path(), Device::Cpu).expect("Failed to open");

        for (name, start, end) in &tensors {
            let view = loader
                .fetch_view(*start, *end)
                .expect(&format!("fetch_view {} failed", name));
            assert_eq!(view.len(), end - start);
            let slice = view.as_slice().unwrap();
            for (i, &byte) in slice.iter().enumerate() {
                assert_eq!(
                    byte,
                    (i % 251) as u8,
                    "Mismatch at byte {} of tensor {}",
                    i,
                    name
                );
            }
        }
    }

    #[test]
    fn test_integration_safetensors_fetch_batch() {
        let (file, tensors) = create_safetensors_file(10, &[50, 50]);
        let loader = FileLoader::open(file.path(), Device::Cpu).expect("Failed to open");

        let infos: Vec<TensorLoadInfo> = tensors
            .iter()
            .map(|(name, start, end)| TensorLoadInfo::new(name.clone(), *start, *end))
            .collect();

        let results = loader.fetch_batch(&infos).expect("fetch_batch failed");
        assert_eq!(results.len(), 10);

        for (result, info) in results.iter().zip(infos.iter()) {
            assert_eq!(result.0, info.name);
            assert_eq!(result.1.len(), info.size());
            let slice = result.1.as_slice().unwrap();
            for (i, &byte) in slice.iter().enumerate() {
                assert_eq!(
                    byte,
                    (i % 251) as u8,
                    "Mismatch at byte {} of tensor {}",
                    i,
                    info.name
                );
            }
        }
    }

    #[test]
    fn test_integration_alloc_then_read() {
        let (file, tensors) = create_safetensors_file(3, &[100, 100]);
        let loader = FileLoader::open(file.path(), Device::Cpu).expect("Failed to open");

        for (name, start, end) in &tensors {
            let buf = loader
                .alloc_buffer(*start, *end)
                .expect(&format!("alloc {} failed", name));
            assert_eq!(buf.len(), end - start);

            let bytes_read = loader
                .read_into(&buf, *start)
                .expect(&format!("read_into {} failed", name));
            assert_eq!(bytes_read, end - start);

            let slice = buf.as_slice().unwrap();
            for (i, &byte) in slice.iter().enumerate() {
                assert_eq!(
                    byte,
                    (i % 251) as u8,
                    "Mismatch at byte {} of tensor {}",
                    i,
                    name
                );
            }
        }
    }

    #[test]
    fn test_device_map_single() {
        let dm = DeviceMap::Single(Device::Cpu);
        assert_eq!(dm.resolve("anything"), Device::Cpu);
        assert_eq!(dm.default_device(), Device::Cpu);
        assert!(dm.is_single());
    }

    #[test]
    fn test_device_map_exact() {
        let mut map = std::collections::HashMap::new();
        map.insert("tensor_a".to_string(), Device::Cpu);
        let dm = DeviceMap::from_map(map, Device::Cpu);

        assert_eq!(dm.resolve("tensor_a"), Device::Cpu);
        assert_eq!(dm.resolve("unknown"), Device::Cpu);
        assert!(!dm.is_single());
    }

    #[test]
    fn test_device_map_prefix_fallback() {
        let mut map = std::collections::HashMap::new();
        map.insert("model.layers.0".to_string(), Device::Cpu);
        let dm = DeviceMap::from_map(map, Device::Cpu);

        // Prefix match: "model.layers.0" matches "model.layers.0.weight"
        assert_eq!(dm.resolve("model.layers.0.weight"), Device::Cpu);
        // No prefix match falls to default
        assert_eq!(dm.resolve("model.layers.1.weight"), Device::Cpu);
        assert!(!dm.is_single());
    }

    #[test]
    fn test_resolve_prefix_map() {
        let mut map = std::collections::HashMap::new();
        map.insert("model.layers.0".to_string(), 0);
        map.insert("model.layers.0.self_attn".to_string(), 1);
        map.insert("model.layers.1".to_string(), 2);
        map.insert("".to_string(), 99);

        // Exact match wins over prefix
        assert_eq!(
            resolve_prefix_map(&map, "model.layers.0.self_attn"),
            Some(&1)
        );

        // Longest prefix wins
        assert_eq!(
            resolve_prefix_map(&map, "model.layers.0.self_attn.q_proj.weight"),
            Some(&1)
        );

        // Shorter prefix match
        assert_eq!(
            resolve_prefix_map(&map, "model.layers.0.mlp.weight"),
            Some(&0)
        );

        // Different layer
        assert_eq!(resolve_prefix_map(&map, "model.layers.1.weight"), Some(&2));

        // Root key "" as catch-all
        assert_eq!(resolve_prefix_map(&map, "lm_head.weight"), Some(&99));

        // No match at all (no "" key)
        let mut no_root: std::collections::HashMap<String, i32> = std::collections::HashMap::new();
        no_root.insert("model.layers.0".to_string(), 0);
        assert_eq!(resolve_prefix_map(&no_root, "lm_head.weight"), None);
    }

    #[test]
    fn test_loader_error_display() {
        assert!(LoaderError::OpenError("test".into())
            .to_string()
            .contains("test"));
        assert!(LoaderError::InitError("init".into())
            .to_string()
            .contains("init"));
        assert!(LoaderError::FetchError("fetch".into())
            .to_string()
            .contains("fetch"));
        assert!(LoaderError::InvalidRange.to_string().contains("invalid"));
    }

    #[test]
    fn test_tensor_load_info_zero_size() {
        let info = TensorLoadInfo::new("empty", 100, 100);
        assert_eq!(info.size(), 0);
    }

    #[test]
    fn test_tensor_load_info_saturating_sub() {
        let info = TensorLoadInfo::new("bad", 500, 100);
        assert_eq!(info.size(), 0);
    }

    #[test]
    fn test_fetch_view_outlives_loader() {
        let content = b"Data that should remain valid after loader is dropped.";
        let temp_file = create_test_file(content);

        let view = {
            let loader = FileLoader::open(temp_file.path(), Device::Cpu).expect("Failed to open");
            loader
                .fetch_view(0, content.len())
                .expect("Failed to get view")
        };

        assert_eq!(view.len(), content.len());
        assert_eq!(view.as_slice().unwrap(), content);
    }

    #[test]
    fn test_multiple_views_same_file() {
        let content = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let temp_file = create_test_file(content);

        let loader = FileLoader::open(temp_file.path(), Device::Cpu).expect("Failed to open");

        let view1 = loader.fetch_view(0, 10).expect("view1");
        let view2 = loader.fetch_view(10, 20).expect("view2");
        let view3 = loader.fetch_view(20, 30).expect("view3");

        assert_eq!(view1.as_slice().unwrap(), b"0123456789");
        assert_eq!(view2.as_slice().unwrap(), b"ABCDEFGHIJ");
        assert_eq!(view3.as_slice().unwrap(), b"KLMNOPQRST");
    }
}
