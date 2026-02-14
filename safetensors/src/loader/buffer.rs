//! Buffer type for tensor data with automatic memory management.

use super::Device;
use std::sync::Arc;

/// Error returned when trying to read GPU buffer memory from CPU.
#[derive(Debug)]
pub struct BufferError;

impl std::fmt::Display for BufferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cannot access GPU buffer from CPU")
    }
}

impl std::error::Error for BufferError {}

/// Ownership semantics for buffer memory.
enum BufferKind {
    /// Empty buffer — nothing to free.
    Empty,
    /// Owned CPU memory allocated via libc::mmap (MAP_PRIVATE | MAP_ANONYMOUS).
    /// Drop → libc::munmap with the stored allocation size.
    OwnedCpu { mmap_size: usize },
    /// Owned CUDA device memory allocated via cudaMalloc.
    /// Drop → cudaFree.
    #[cfg(feature = "cuda")]
    OwnedCuda,
    /// Zero-copy view into mmap'd file memory.
    /// The Arc keeps the mmap alive while this buffer exists.
    MmapView(#[allow(dead_code)] Arc<memmap2::Mmap>),
}

/// A buffer containing tensor data, either on CPU or GPU.
///
/// Buffers come in several flavors:
/// - **Empty**: Zero-length buffer with no memory.
/// - **OwnedCpu**: CPU memory allocated via anonymous mmap (freed on drop).
/// - **OwnedCuda**: CUDA device memory allocated via cudaMalloc (freed on drop).
/// - **MmapView**: Zero-copy pointer into mmap'd file memory, kept alive via Arc.
pub struct Buffer {
    ptr: *mut u8,
    size: usize,
    device: Device,
    kind: BufferKind,
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let owned = match &self.kind {
            BufferKind::OwnedCpu { .. } => true,
            #[cfg(feature = "cuda")]
            BufferKind::OwnedCuda => true,
            _ => false,
        };
        f.debug_struct("Buffer")
            .field("size", &self.size)
            .field("ptr", &self.ptr)
            .field("device", &self.device)
            .field("owned", &owned)
            .finish()
    }
}

// SAFETY: Buffer is Send because:
// - OwnedCpu: exclusively owned memory, no aliasing
// - OwnedCuda: CUDA device memory, no aliasing (cudaFree is thread-safe)
// - MmapView: immutable mmap, Arc<Mmap> is Send+Sync
// - Empty: no memory
unsafe impl Send for Buffer {}

// SAFETY: Buffer is Sync because:
// - All variants provide immutable access only (as_slice returns &[u8])
// - No interior mutability
unsafe impl Sync for Buffer {}

impl Buffer {
    /// Create an empty buffer for the given device.
    #[inline]
    pub fn empty(device: Device) -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            size: 0,
            device,
            kind: BufferKind::Empty,
        }
    }

    /// Allocate a CPU buffer via anonymous mmap.
    ///
    /// Uses MAP_HUGETLB on Linux (with fallback to MADV_HUGEPAGE)
    /// for optimal TLB performance with large tensor buffers.
    pub(crate) fn alloc_cpu(size: usize) -> super::Result<Self> {
        if size == 0 {
            return Ok(Self::empty(Device::Cpu));
        }
        let mmap_size = page_align(size);
        let ptr = alloc_anonymous_mmap(mmap_size)?;
        Ok(Self {
            ptr,
            size,
            device: Device::Cpu,
            kind: BufferKind::OwnedCpu { mmap_size },
        })
    }

    /// Create an owned CUDA buffer from a raw pointer.
    ///
    /// # Safety
    ///
    /// The pointer must have been allocated via cudaMalloc with at least `size` bytes.
    /// The caller transfers ownership — the buffer will call cudaFree on drop.
    #[cfg(feature = "cuda")]
    #[inline]
    pub(crate) unsafe fn from_raw_cuda(ptr: *mut u8, size: usize, device: Device) -> Self {
        Self {
            ptr,
            size,
            device,
            kind: BufferKind::OwnedCuda,
        }
    }

    /// Create a zero-copy view into mmap'd file memory.
    ///
    /// # Safety
    ///
    /// The pointer must point into the mmap'd region. The Arc keeps the mmap alive.
    #[inline]
    pub(crate) unsafe fn from_mmap_view(
        ptr: *const u8,
        size: usize,
        mmap: Arc<memmap2::Mmap>,
    ) -> Self {
        Self {
            ptr: ptr as *mut u8,
            size,
            device: Device::Cpu,
            kind: BufferKind::MmapView(mmap),
        }
    }

    /// Get the buffer as a byte slice (CPU only).
    ///
    /// Returns `None` for CUDA buffers (device memory cannot be read directly).
    #[inline]
    pub fn as_slice(&self) -> Option<&[u8]> {
        match self.device {
            Device::Cpu => {
                if self.ptr.is_null() || self.size == 0 {
                    Some(&[])
                } else {
                    unsafe { Some(std::slice::from_raw_parts(self.ptr, self.size)) }
                }
            }
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => None,
        }
    }

    /// Get the size of the buffer in bytes.
    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.size
    }

    /// Check if the buffer is empty.
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Get the device where the buffer is located.
    #[inline(always)]
    pub fn device(&self) -> Device {
        self.device
    }

    /// Get a raw pointer to the buffer data.
    #[inline(always)]
    pub const fn as_ptr(&self) -> *const u8 {
        self.ptr as *const u8
    }

    /// Convert to a Vec (copies data). Only works for CPU buffers.
    #[inline]
    pub fn to_vec(&self) -> std::result::Result<Vec<u8>, BufferError> {
        self.as_slice().map(|s| s.to_vec()).ok_or(BufferError)
    }

}

impl Drop for Buffer {
    fn drop(&mut self) {
        match &self.kind {
            BufferKind::Empty | BufferKind::MmapView(_) => {}
            BufferKind::OwnedCpu { mmap_size } => {
                if !self.ptr.is_null() {
                    unsafe {
                        libc::munmap(self.ptr as *mut libc::c_void, *mmap_size);
                    }
                }
            }
            #[cfg(feature = "cuda")]
            BufferKind::OwnedCuda => {
                if !self.ptr.is_null() {
                    super::cuda::cuda_free(self.ptr);
                }
            }
        }
    }
}

/// Round up to page boundary (4096).
#[inline]
fn page_align(size: usize) -> usize {
    (size + 4095) & !4095
}

/// Allocate anonymous mmap'd memory.
///
/// On Linux, tries MAP_HUGETLB first for 2MB huge pages (better TLB performance),
/// falling back to regular pages with MADV_HUGEPAGE hint.
fn alloc_anonymous_mmap(mmap_size: usize) -> super::Result<*mut u8> {
    // Try huge pages first (Linux only)
    #[cfg(target_os = "linux")]
    {
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                mmap_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_HUGETLB,
                -1,
                0,
            )
        };
        if ptr != libc::MAP_FAILED {
            return Ok(ptr as *mut u8);
        }
    }

    // Regular mmap (works on all platforms)
    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            mmap_size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };

    if ptr == libc::MAP_FAILED {
        return Err(super::LoaderError::FetchError(format!(
            "mmap allocation failed for {} bytes: {}",
            mmap_size,
            std::io::Error::last_os_error()
        )));
    }

    // Hint: use transparent huge pages if available
    #[cfg(target_os = "linux")]
    unsafe {
        libc::madvise(ptr, mmap_size, libc::MADV_HUGEPAGE);
    }

    Ok(ptr as *mut u8)
}
