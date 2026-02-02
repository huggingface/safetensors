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
    #[cfg(feature = "io_uring")]
    IoUring,
}

impl Backend {
    /// Convert to hmll LoaderKind.
    #[inline]
    fn to_hmll(self) -> hmll::LoaderKind {
        match self {
            Backend::Auto => hmll::LoaderKind::Auto,
            Backend::Mmap => hmll::LoaderKind::Mmap,
            #[cfg(feature = "io_uring")]
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
    /// I/O error.
    IoError(std::io::Error),
}

impl std::fmt::Display for LoaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoaderError::OpenError(msg) => write!(f, "failed to open source: {msg}"),
            LoaderError::InitError(msg) => write!(f, "failed to initialize loader: {msg}"),
            LoaderError::FetchError(msg) => write!(f, "failed to fetch data: {msg}"),
            LoaderError::InvalidRange => write!(f, "invalid byte range"),
            LoaderError::IoError(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for LoaderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LoaderError::IoError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<hmll::Error> for LoaderError {
    fn from(err: hmll::Error) -> Self {
        LoaderError::FetchError(format!("{err}"))
    }
}

impl From<std::io::Error> for LoaderError {
    fn from(err: std::io::Error) -> Self {
        LoaderError::IoError(err)
    }
}

/// Result type for loader operations.
pub type Result<T> = std::result::Result<T, LoaderError>;

/// Internal wrapper for WeightLoader with 'static lifetime.
struct LoaderInner {
    loader: hmll::WeightLoader<'static>,
}

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
    inner: Mutex<LoaderInner>,
    /// Target device.
    device: Device,
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

        // SAFETY: The source is boxed and will live as long as Loader.
        // We transmute the lifetime to 'static because the source won't be
        // moved or dropped while the loader exists.
        let sources_slice: &[hmll::Source] = std::slice::from_ref(source.as_ref());
        let static_sources: &'static [hmll::Source] = unsafe { std::mem::transmute(sources_slice) };

        let loader = hmll::WeightLoader::new(static_sources, device.to_hmll(), backend.to_hmll())
            .map_err(|e| LoaderError::InitError(format!("{e}")))?;

        Ok(Self {
            source,
            inner: Mutex::new(LoaderInner { loader }),
            device,
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
        let buffer = guard.loader.fetch(start..end, 0)?;
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
        let buffer = guard.loader.fetch_view(start..end, 0)?;
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
        Ok(buffer.to_vec())
    }

    /// Get the target device for this loader.
    #[inline]
    pub fn device(&self) -> Device {
        self.device
    }

    /// Get the size of the source file in bytes.
    pub fn file_size(&self) -> usize {
        let guard = self.inner.lock().unwrap();
        guard
            .loader
            .source_info(0)
            .map(|info| info.size)
            .unwrap_or(0)
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

/// Builder for creating loaders with custom configuration.
///
/// # Example
///
/// ```no_run
/// use safetensors::loader::{LoaderBuilder, Device, Backend};
///
/// let loader = LoaderBuilder::new()
///     .device(Device::Cpu)
///     .backend(Backend::Mmap)
///     .open("model.safetensors")?;
/// # Ok::<(), safetensors::loader::LoaderError>(())
/// ```
#[derive(Debug, Clone, Default)]
pub struct LoaderBuilder {
    device: Device,
    backend: Backend,
}

impl LoaderBuilder {
    /// Create a new loader builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the target device.
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Set the loader backend.
    pub fn backend(mut self, backend: Backend) -> Self {
        self.backend = backend;
        self
    }

    /// Open a file with the configured settings.
    pub fn open<P: AsRef<Path>>(self, path: P) -> Result<Loader> {
        Loader::with_backend(path, self.device, self.backend)
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
    fn test_loader_builder() {
        let content = b"Builder test";
        let temp_file = create_test_file(content);

        let loader = LoaderBuilder::new()
            .device(Device::Cpu)
            .backend(Backend::Mmap)
            .open(temp_file.path())
            .expect("Failed to open with builder");

        assert_eq!(loader.device(), Device::Cpu);
    }
}
