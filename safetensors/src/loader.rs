//! High-performance tensor loading using hmll.
//!
//! This module provides efficient file loading capabilities for safetensors files,
//! supporting both CPU and GPU (CUDA) targets with optional io_uring acceleration.
//!
//! # Features
//!
//! - **Zero-copy GPU loading**: Load tensors directly to GPU memory via DMA
//! - **io_uring support**: Async I/O on Linux for improved throughput
//! - **Multi-file support**: Load from multiple sharded files efficiently
//!
//! # Example
//!
//! ```no_run
//! use safetensors::loader::{Loader, Device};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Open a safetensors file
//! let loader = Loader::open("model.safetensors", Device::Cpu)?;
//!
//! // Fetch a byte range
//! let buffer = loader.fetch(0, 1024)?;
//! println!("Loaded {} bytes", buffer.len());
//! # Ok(())
//! # }
//! ```

use std::path::Path;
use std::sync::Mutex;

/// Re-export hmll types that users might need
pub use hmll::Buffer;

/// Target device for loading tensors.
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
    /// Convert to hmll Device.
    #[inline]
    fn to_hmll(self) -> hmll::Device {
        match self {
            Device::Cpu => hmll::Device::Cpu,
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => hmll::Device::Cuda,
        }
    }

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
    /// Use io_uring for async I/O (Linux only).
    #[cfg(all(target_os = "linux", feature = "io_uring"))]
    IoUring,
}

impl Backend {
    /// Convert to hmll LoaderKind.
    #[inline]
    fn to_hmll(self) -> hmll::LoaderKind {
        match self {
            Backend::Auto => hmll::LoaderKind::Auto,
            Backend::Mmap => hmll::LoaderKind::Mmap,
            #[cfg(all(target_os = "linux", feature = "io_uring"))]
            Backend::IoUring => hmll::LoaderKind::IoUring,
        }
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

impl From<hmll::Error> for LoaderError {
    fn from(err: hmll::Error) -> Self {
        LoaderError::FetchError(format!("{err}"))
    }
}

/// Result type for loader operations.
pub type Result<T> = std::result::Result<T, LoaderError>;

/// A high-performance loader for safetensors files.
///
/// The `Loader` provides efficient access to tensor data from safetensors files,
/// with support for CPU and GPU targets.
///
/// # Thread Safety
///
/// `Loader` is `Send + Sync`, allowing it to be shared across threads.
/// Internal synchronization ensures safe concurrent access.
///
/// # Example
///
/// ```no_run
/// use safetensors::loader::{Loader, Device};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let loader = Loader::open("model.safetensors", Device::Cpu)?;
///
/// // Fetch first 1KB
/// let data = loader.fetch(0, 1024)?;
/// # Ok(())
/// # }
/// ```
pub struct Loader {
    /// Boxed source to ensure stable memory address.
    /// Must be kept alive for the lifetime of the loader.
    #[allow(dead_code)]
    source: Box<hmll::Source>,
    /// The actual loader, behind a Mutex for interior mutability.
    inner: Mutex<hmll::WeightLoader<'static>>,
    /// Target device.
    device: Device,
    /// Cached file size (avoids locking mutex).
    file_size: usize,
}

impl Loader {
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
    /// use safetensors::loader::{Loader, Device};
    ///
    /// let loader = Loader::open("model.safetensors", Device::Cpu)?;
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
        let source = Box::new(
            hmll::Source::open(path.as_ref())
                .map_err(|e| LoaderError::OpenError(format!("{e}")))?,
        );
        let file_size = source.size();

        // SAFETY: The source is boxed and will live as long as Loader.
        // We transmute the lifetime to 'static because the source won't be
        // moved or dropped while the loader exists.
        let sources_slice: &[hmll::Source] = std::slice::from_ref(source.as_ref());
        let static_sources: &'static [hmll::Source] = unsafe { std::mem::transmute(sources_slice) };

        let loader = hmll::WeightLoader::new(static_sources, device.to_hmll(), backend.to_hmll())
            .map_err(|e| LoaderError::InitError(format!("{e}")))?;

        Ok(Self {
            source,
            inner: Mutex::new(loader),
            device,
            file_size,
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
    /// # use safetensors::loader::{Loader, Device};
    /// # let loader = Loader::open("model.safetensors", Device::Cpu)?;
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
            // Return empty buffer for zero-length range
            return Ok(Buffer::empty(self.device.to_hmll()));
        }

        let mut guard = self.inner.lock().unwrap();
        let buffer = guard.fetch(start..end, 0)?;
        Ok(buffer)
    }

    /// Fetch a zero-copy view of a byte range from the file.
    ///
    /// This returns a `Buffer` that points directly into the mmap'd file
    /// without any memory allocation or copying. The buffer is only valid
    /// as long as this `Loader` remains valid.
    ///
    /// **Note**: This only works for CPU targets with the mmap backend.
    /// For CUDA targets or other backends, use `fetch()` instead.
    ///
    /// # Arguments
    ///
    /// * `start` - Start offset in bytes
    /// * `end` - End offset in bytes (exclusive)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use safetensors::loader::{Loader, Device};
    /// # let loader = Loader::open("model.safetensors", Device::Cpu)?;
    /// // Get a zero-copy view into the file
    /// let view = loader.fetch_view(1000, 2000)?;
    /// assert_eq!(view.len(), 1000);
    /// // view points directly into mmap'd memory - no allocation or copy!
    /// # Ok::<(), safetensors::loader::LoaderError>(())
    /// ```
    pub fn fetch_view(&self, start: usize, end: usize) -> Result<Buffer> {
        if start > end {
            return Err(LoaderError::InvalidRange);
        }
        if start == end {
            // Return empty buffer for zero-length range
            return Ok(Buffer::empty(self.device.to_hmll()));
        }

        // Only works for CPU device
        if self.device != Device::Cpu {
            return Err(LoaderError::FetchError(
                "fetch_view only supported for CPU device".into(),
            ));
        }

        let mut guard = self.inner.lock().unwrap();
        let buffer = guard.fetch_view(start..end, 0)?;
        Ok(buffer)
    }

    /// Fetch a range of bytes and copy to a Vec.
    ///
    /// This is a convenience method that fetches data and copies it to
    /// owned memory. For CPU targets, this involves a memory copy.
    /// For CUDA targets, this involves a device-to-host transfer.
    ///
    /// # Arguments
    ///
    /// * `start` - Start offset in bytes
    /// * `end` - End offset in bytes (exclusive)
    pub fn fetch_to_vec(&self, start: usize, end: usize) -> Result<Vec<u8>> {
        let buffer = self.fetch(start, end)?;
        Ok(buffer.to_vec()?)
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
}

// SAFETY: Loader is Send because:
// - source is boxed and immutable after creation
// - inner is behind a Mutex which provides synchronization
unsafe impl Send for Loader {}

// SAFETY: Loader is Sync because:
// - All mutable access goes through the Mutex
// - The Mutex provides proper synchronization
unsafe impl Sync for Loader {}

/// Configuration for the prefetch iterator.
#[derive(Debug, Clone)]
pub struct PrefetchConfig {
    /// Number of tensors to prefetch ahead (default: 4).
    pub prefetch_count: usize,
}

impl Default for PrefetchConfig {
    fn default() -> Self {
        Self { prefetch_count: 4 }
    }
}

impl PrefetchConfig {
    /// Create a new prefetch configuration.
    pub fn new(prefetch_count: usize) -> Self {
        Self {
            prefetch_count: prefetch_count.max(1),
        }
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

/// An iterator that prefetches tensors ahead of consumption.
///
/// This iterator yields `(name, buffer)` pairs while prefetching subsequent
/// tensors in the background. This allows overlapping I/O with user processing
/// (e.g., quantization).
///
/// For CUDA devices, this uses async GPU allocation and memcpy with per-slot
/// streams and events, enabling true overlap between I/O and compute.
///
/// For CPU devices, this uses zero-copy mmap views (already optimal).
///
/// # Example
///
/// ```no_run
/// use safetensors::loader::{Loader, Device, PrefetchConfig, TensorLoadInfo};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let loader = Loader::open("model.safetensors", Device::Cpu)?;
///
/// // Define tensors to load (normally from safetensors metadata)
/// let tensors = vec![
///     TensorLoadInfo::new("layer1.weight", 0, 1024),
///     TensorLoadInfo::new("layer1.bias", 1024, 1536),
///     // ... more tensors
/// ];
///
/// // Create prefetch iterator
/// let config = PrefetchConfig::new(4);
/// for result in loader.iter_prefetch(&tensors, config) {
///     let (name, buffer) = result?;
///     // Process tensor (e.g., quantize) while next tensors load
///     println!("Loaded {}: {} bytes", name, buffer.len());
/// }
/// # Ok(())
/// # }
/// ```
pub struct PrefetchIterator<'a> {
    loader: &'a Loader,
    tensors: Vec<TensorLoadInfo>,
    current_index: usize,
    #[cfg(feature = "cuda")]
    next_to_schedule: usize,
    #[cfg(feature = "cuda")]
    mmap_ptr: Option<*const std::ffi::c_void>,
    #[cfg(feature = "cuda")]
    prefetch_ctx: Option<hmll::PrefetchContext>,
}

impl<'a> PrefetchIterator<'a> {
    /// Create a new prefetch iterator.
    pub fn new(loader: &'a Loader, tensors: Vec<TensorLoadInfo>, config: PrefetchConfig) -> Self {
        #[cfg(feature = "cuda")]
        let num_slots = config.prefetch_count.min(tensors.len()).max(1);
        #[cfg(not(feature = "cuda"))]
        let _ = config; // Suppress unused warning

        #[cfg(feature = "cuda")]
        let mmap_ptr = {
            let guard = loader.inner.lock().unwrap();
            guard.source_content_ptr(0)
        };

        #[cfg(feature = "cuda")]
        let prefetch_ctx = match loader.device {
            Device::Cuda(device_id) => {
                hmll::PrefetchContext::new(num_slots, hmll::Device::Cuda, device_id as i32).ok()
            }
            Device::Cpu => None,
        };

        let mut iter = Self {
            loader,
            tensors,
            current_index: 0,
            #[cfg(feature = "cuda")]
            next_to_schedule: 0,
            #[cfg(feature = "cuda")]
            mmap_ptr,
            #[cfg(feature = "cuda")]
            prefetch_ctx,
        };

        iter.fill_slots();
        iter
    }

    fn fill_slots(&mut self) {
        #[cfg(feature = "cuda")]
        if let Some(ref mut ctx) = self.prefetch_ctx {
            while self.next_to_schedule < self.tensors.len() {
                if ctx.find_available_slot().is_none() {
                    break;
                }

                let tensor_index = self.next_to_schedule;
                let info = &self.tensors[tensor_index];

                if let Some(mmap_base) = self.mmap_ptr {
                    let src_ptr =
                        unsafe { (mmap_base as *const u8).add(info.start) as *const std::ffi::c_void };

                    if ctx.start_load(src_ptr, info.size(), tensor_index).is_ok() {
                        self.next_to_schedule += 1;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
        }
        // CPU: mmap views are created on-demand, no prefetch needed
    }

    /// Take the next tensor from the iterator.
    fn take_next(&mut self) -> Option<Result<(String, Buffer)>> {
        if self.current_index >= self.tensors.len() {
            return None;
        }

        let tensor_index = self.current_index;
        let info = self.tensors[tensor_index].clone();
        let name = info.name.clone();

        let buffer = self.get_tensor_buffer(tensor_index, &info);

        self.current_index += 1;

        // Refill slots with next tensors
        self.fill_slots();

        Some(buffer.map(|b| (name, b)))
    }

    #[allow(unused_variables)]
    fn get_tensor_buffer(&mut self, tensor_index: usize, info: &TensorLoadInfo) -> Result<Buffer> {
        #[cfg(feature = "cuda")]
        if let Some(ref mut ctx) = self.prefetch_ctx {
            if let Some(slot) = ctx.find_tensor(tensor_index) {
                // Tensor is in a slot, take the buffer (waits if still loading)
                return ctx.take_buffer(slot).map_err(|e| LoaderError::FetchError(e.to_string()));
            }

            // Tensor not prefetched, do synchronous load
            // This can happen if prefetch slots were exhausted
            if let Some(mmap_base) = self.mmap_ptr {
                let src_ptr =
                    unsafe { (mmap_base as *const u8).add(info.start) as *const std::ffi::c_void };
                let size = info.size();

                // Start load and immediately wait
                if let Ok(slot) = ctx.start_load(src_ptr, size, tensor_index) {
                    return ctx.take_buffer(slot).map_err(|e| LoaderError::FetchError(e.to_string()));
                }
            }

            // Fallback to regular fetch
            return self.loader.fetch(info.start, info.end);
        }

        // CPU path: use zero-copy mmap view
        self.loader.fetch_view(info.start, info.end)
    }
}

impl<'a> Iterator for PrefetchIterator<'a> {
    type Item = Result<(String, Buffer)>;

    fn next(&mut self) -> Option<Self::Item> {
        self.take_next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.tensors.len().saturating_sub(self.current_index);
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for PrefetchIterator<'a> {}

/// An owned prefetch iterator that holds an Arc to the loader.
///
/// This variant is useful when you need to store the iterator in a struct
/// that outlives the loader reference (e.g., Python bindings).
///
/// For most Rust use cases, prefer `PrefetchIterator` which uses a reference.
pub struct OwnedPrefetchIterator {
    loader: std::sync::Arc<Loader>,
    tensors: Vec<TensorLoadInfo>,
    current_index: usize,
    #[cfg(feature = "cuda")]
    next_to_schedule: usize,
    #[cfg(feature = "cuda")]
    mmap_ptr: Option<*const std::ffi::c_void>,
    #[cfg(feature = "cuda")]
    prefetch_ctx: Option<hmll::PrefetchContext>,
}

// SAFETY: OwnedPrefetchIterator can be moved between threads.
// - `mmap_ptr` is a read-only pointer into memory kept alive by `Arc<Loader>`
// - `PrefetchContext` is Send (see hmll::PrefetchContext safety comment)
// - All methods require `&mut self`, ensuring exclusive access
// - No `Sync` impl means the iterator cannot be shared across threads
unsafe impl Send for OwnedPrefetchIterator {}

impl OwnedPrefetchIterator {
    /// Create a new owned prefetch iterator.
    pub fn new(
        loader: std::sync::Arc<Loader>,
        tensors: Vec<TensorLoadInfo>,
        config: PrefetchConfig,
    ) -> Self {
        #[cfg(feature = "cuda")]
        let num_slots = config.prefetch_count.min(tensors.len()).max(1);
        #[cfg(not(feature = "cuda"))]
        let _ = config;

        #[cfg(feature = "cuda")]
        let mmap_ptr = {
            let guard = loader.inner.lock().unwrap();
            guard.source_content_ptr(0)
        };

        #[cfg(feature = "cuda")]
        let prefetch_ctx = match loader.device {
            Device::Cuda(device_id) => {
                hmll::PrefetchContext::new(num_slots, hmll::Device::Cuda, device_id as i32).ok()
            }
            Device::Cpu => None,
        };

        let mut iter = Self {
            loader,
            tensors,
            current_index: 0,
            #[cfg(feature = "cuda")]
            next_to_schedule: 0,
            #[cfg(feature = "cuda")]
            mmap_ptr,
            #[cfg(feature = "cuda")]
            prefetch_ctx,
        };

        iter.fill_slots();
        iter
    }

    fn fill_slots(&mut self) {
        #[cfg(feature = "cuda")]
        if let Some(ref mut ctx) = self.prefetch_ctx {
            while self.next_to_schedule < self.tensors.len() {
                if ctx.find_available_slot().is_none() {
                    break;
                }

                let tensor_index = self.next_to_schedule;
                let info = &self.tensors[tensor_index];

                if let Some(mmap_base) = self.mmap_ptr {
                    let src_ptr =
                        unsafe { (mmap_base as *const u8).add(info.start) as *const std::ffi::c_void };

                    if ctx.start_load(src_ptr, info.size(), tensor_index).is_ok() {
                        self.next_to_schedule += 1;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
        }
        // CPU: mmap views are created on-demand
    }

    #[allow(unused_variables)]
    fn get_tensor_buffer(&mut self, tensor_index: usize, info: &TensorLoadInfo) -> Result<Buffer> {
        #[cfg(feature = "cuda")]
        if let Some(ref mut ctx) = self.prefetch_ctx {
            if let Some(slot) = ctx.find_tensor(tensor_index) {
                return ctx.take_buffer(slot).map_err(|e| LoaderError::FetchError(e.to_string()));
            }

            // Fallback: tensor not in prefetch queue, load synchronously
            if let Some(mmap_base) = self.mmap_ptr {
                let src_ptr =
                    unsafe { (mmap_base as *const u8).add(info.start) as *const std::ffi::c_void };

                if let Ok(slot) = ctx.start_load(src_ptr, info.size(), tensor_index) {
                    return ctx.take_buffer(slot).map_err(|e| LoaderError::FetchError(e.to_string()));
                }
            }

            // Fallback to regular fetch
            return self.loader.fetch(info.start, info.end);
        }

        // CPU path: use zero-copy mmap view
        self.loader.fetch_view(info.start, info.end)
    }

    /// Get the number of remaining tensors.
    pub fn remaining(&self) -> usize {
        self.tensors.len().saturating_sub(self.current_index)
    }

    /// Get the total number of tensors.
    pub fn total(&self) -> usize {
        self.tensors.len()
    }
}

impl Iterator for OwnedPrefetchIterator {
    type Item = Result<(String, Buffer)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.tensors.len() {
            return None;
        }

        let tensor_index = self.current_index;
        let info = self.tensors[tensor_index].clone();
        let name = info.name.clone();

        let buffer = self.get_tensor_buffer(tensor_index, &info);

        self.current_index += 1;

        // Refill slots with next tensors
        self.fill_slots();

        Some(buffer.map(|b| (name, b)))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.remaining();
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for OwnedPrefetchIterator {}

impl Loader {
    /// Create a prefetch iterator over the given tensors.
    ///
    /// This iterator loads tensors sequentially while prefetching ahead,
    /// allowing user-side processing to overlap with I/O.
    ///
    /// # Arguments
    ///
    /// * `tensors` - Tensor metadata (name, start offset, end offset)
    /// * `config` - Prefetch configuration
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use safetensors::loader::{Loader, Device, PrefetchConfig, TensorLoadInfo};
    /// # let loader = Loader::open("model.safetensors", Device::Cpu).unwrap();
    /// let tensors = vec![
    ///     TensorLoadInfo::new("weight", 0, 4096),
    ///     TensorLoadInfo::new("bias", 4096, 4608),
    /// ];
    ///
    /// for result in loader.iter_prefetch(&tensors, PrefetchConfig::default()) {
    ///     let (name, buffer) = result.unwrap();
    ///     println!("Loaded {}", name);
    /// }
    /// ```
    pub fn iter_prefetch(
        &self,
        tensors: &[TensorLoadInfo],
        config: PrefetchConfig,
    ) -> PrefetchIterator<'_> {
        PrefetchIterator::new(self, tensors.to_vec(), config)
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

        let loader = Loader::open(temp_file.path(), Device::Cpu).expect("Failed to open");
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

        let loader = Loader::open(temp_file.path(), Device::Cpu).expect("Failed to open");

        let buffer = loader.fetch(5, 15).expect("Failed to fetch");
        assert_eq!(buffer.len(), 10);
        assert_eq!(buffer.as_slice().unwrap(), b"56789ABCDE");
    }

    #[test]
    fn test_loader_empty_fetch() {
        let content = b"Some content";
        let temp_file = create_test_file(content);

        let loader = Loader::open(temp_file.path(), Device::Cpu).expect("Failed to open");

        let buffer = loader.fetch(5, 5).expect("Failed to fetch empty range");
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_loader_invalid_range() {
        let content = b"Some content";
        let temp_file = create_test_file(content);

        let loader = Loader::open(temp_file.path(), Device::Cpu).expect("Failed to open");

        let result = loader.fetch(10, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_loader_fetch_to_vec() {
        let content = b"Vector conversion test";
        let temp_file = create_test_file(content);

        let loader = Loader::open(temp_file.path(), Device::Cpu).expect("Failed to open");

        let vec = loader
            .fetch_to_vec(0, content.len())
            .expect("Failed to fetch");
        assert_eq!(vec, content.to_vec());
    }

    #[test]
    fn test_prefetch_iterator_basic() {
        let content = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let temp_file = create_test_file(content);

        let loader = Loader::open(temp_file.path(), Device::Cpu).expect("Failed to open");

        let tensors = vec![
            TensorLoadInfo::new("first", 0, 10),
            TensorLoadInfo::new("second", 10, 20),
            TensorLoadInfo::new("third", 20, 30),
        ];

        let results: Vec<_> = loader
            .iter_prefetch(&tensors, PrefetchConfig::default())
            .collect();

        assert_eq!(results.len(), 3);

        let (name, buf) = results[0].as_ref().unwrap();
        assert_eq!(name, "first");
        assert_eq!(buf.as_slice().unwrap(), b"0123456789");

        let (name, buf) = results[1].as_ref().unwrap();
        assert_eq!(name, "second");
        assert_eq!(buf.as_slice().unwrap(), b"ABCDEFGHIJ");

        let (name, buf) = results[2].as_ref().unwrap();
        assert_eq!(name, "third");
        assert_eq!(buf.as_slice().unwrap(), b"KLMNOPQRST");
    }

    #[test]
    fn test_prefetch_iterator_single_tensor() {
        let content = b"Single tensor content";
        let temp_file = create_test_file(content);

        let loader = Loader::open(temp_file.path(), Device::Cpu).expect("Failed to open");

        let tensors = vec![TensorLoadInfo::new("only", 0, content.len())];

        let results: Vec<_> = loader
            .iter_prefetch(&tensors, PrefetchConfig::new(4))
            .collect();

        assert_eq!(results.len(), 1);
        let (name, buf) = results[0].as_ref().unwrap();
        assert_eq!(name, "only");
        assert_eq!(buf.as_slice().unwrap(), content);
    }

    #[test]
    fn test_prefetch_iterator_empty() {
        let content = b"Some content";
        let temp_file = create_test_file(content);

        let loader = Loader::open(temp_file.path(), Device::Cpu).expect("Failed to open");

        let tensors: Vec<TensorLoadInfo> = vec![];

        let results: Vec<_> = loader
            .iter_prefetch(&tensors, PrefetchConfig::default())
            .collect();

        assert!(results.is_empty());
    }

    #[test]
    fn test_prefetch_iterator_large_prefetch_count() {
        let content = b"AB";
        let temp_file = create_test_file(content);

        let loader = Loader::open(temp_file.path(), Device::Cpu).expect("Failed to open");

        // More prefetch slots than tensors
        let tensors = vec![
            TensorLoadInfo::new("a", 0, 1),
            TensorLoadInfo::new("b", 1, 2),
        ];

        let results: Vec<_> = loader
            .iter_prefetch(&tensors, PrefetchConfig::new(100))
            .collect();

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_prefetch_iterator_size_hint() {
        let content = b"0123456789";
        let temp_file = create_test_file(content);

        let loader = Loader::open(temp_file.path(), Device::Cpu).expect("Failed to open");

        let tensors = vec![
            TensorLoadInfo::new("a", 0, 3),
            TensorLoadInfo::new("b", 3, 6),
            TensorLoadInfo::new("c", 6, 10),
        ];

        let mut iter = loader.iter_prefetch(&tensors, PrefetchConfig::default());

        assert_eq!(iter.len(), 3);
        assert_eq!(iter.size_hint(), (3, Some(3)));

        iter.next();
        assert_eq!(iter.len(), 2);
        assert_eq!(iter.size_hint(), (2, Some(2)));

        iter.next();
        iter.next();
        assert_eq!(iter.len(), 0);
        assert!(iter.next().is_none());
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
    fn test_prefetch_config() {
        let config = PrefetchConfig::default();
        assert_eq!(config.prefetch_count, 4);

        let config = PrefetchConfig::new(8);
        assert_eq!(config.prefetch_count, 8);

        // Should clamp to 1
        let config = PrefetchConfig::new(0);
        assert_eq!(config.prefetch_count, 1);
    }
}
