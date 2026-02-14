//! MmapSource: memory-mapped file wrapper using memmap2.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// A memory-mapped source file.
///
/// Wraps a `memmap2::Mmap` with Arc-based lifetime management so that
/// zero-copy view buffers can safely outlive the loader.
pub(crate) struct MmapSource {
    mmap: Arc<memmap2::Mmap>,
    #[allow(dead_code)]
    file: File,
    size: usize,
    #[allow(dead_code)]
    path: PathBuf,
}

impl MmapSource {
    /// Open and memory-map a file.
    pub fn open<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)?;
        let size = file.metadata()?.len() as usize;
        // SAFETY: The file is opened read-only and the mmap is immutable.
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        Ok(Self {
            mmap: Arc::new(mmap),
            file,
            size,
            path: path.to_path_buf(),
        })
    }

    /// Get a pointer to the mmap'd content.
    #[inline]
    pub fn content_ptr(&self) -> *const u8 {
        self.mmap.as_ptr()
    }

    /// Clone the Arc to the mmap (for creating view buffers that outlive the loader).
    #[inline]
    pub fn mmap_arc(&self) -> Arc<memmap2::Mmap> {
        self.mmap.clone()
    }

    /// Get the file size in bytes.
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the raw file descriptor (Linux only, for cuFile/io_uring).
    #[cfg(target_os = "linux")]
    #[allow(dead_code)] // used by cufile::CuFileSource when cufile feature is enabled
    #[inline]
    pub fn raw_fd(&self) -> std::os::unix::io::RawFd {
        use std::os::unix::io::AsRawFd;
        self.file.as_raw_fd()
    }
}
