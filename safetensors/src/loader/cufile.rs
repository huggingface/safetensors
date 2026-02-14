//! cuFile/GDS FFI for direct NVMe→GPU DMA.
//!
//! Provides direct FFI to the cuFile library for GPU Direct Storage,
//! enabling NVMe-to-GPU DMA transfers that bypass CPU entirely.

use std::ffi::{c_int, c_uint, c_void};

use super::{LoaderError, Result};

/// cuFile error type (returned by value from C API).
#[repr(C)]
struct CUfileError {
    err: c_int,
    cu_err: c_int,
}

extern "C" {
    fn cuFileDriverOpen() -> CUfileError;
    fn cuFileDriverClose();
    fn cuFileHandleRegister(fh: *mut *mut c_void, descr: *mut c_void) -> CUfileError;
    fn cuFileHandleDeregister(fh: *mut c_void);
    fn cuFileRead(
        fh: *mut c_void,
        buf: *mut c_void,
        size: usize,
        file_offset: i64,
        buf_offset: i64,
    ) -> isize;
    fn cuFileBufRegister(devPtr: *const c_void, size: usize, flags: c_uint) -> CUfileError;
    fn cuFileBufDeregister(devPtr: *const c_void) -> CUfileError;
}

/// CU_FILE_HANDLE_TYPE_OPAQUE_FD
const CU_FILE_HANDLE_TYPE_OPAQUE_FD: c_int = 1;

/// A cuFile source wrapping a registered file handle.
///
/// Manages the cuFile driver lifecycle and file handle registration.
pub(crate) struct CuFileSource {
    handle: *mut c_void,
}

// SAFETY: cuFile handles can be used from any thread.
// cuFileRead is documented as thread-safe.
unsafe impl Send for CuFileSource {}
unsafe impl Sync for CuFileSource {}

impl CuFileSource {
    /// Open a cuFile source from a raw file descriptor.
    ///
    /// Initializes the cuFile driver (if not already) and registers the fd.
    pub fn open(fd: c_int) -> Result<Self> {
        // Open cuFile driver (can be called multiple times, ref-counted)
        let err = unsafe { cuFileDriverOpen() };
        if err.err != 0 {
            return Err(LoaderError::InitError(format!(
                "cuFileDriverOpen failed: err={}, cu_err={}",
                err.err, err.cu_err
            )));
        }

        // Build CUfileDescr_t on the stack.
        // Layout: { CUfileFileHandleType type; union { int fd; void* handle; }; ... }
        // Using a large zero-initialized buffer to accommodate different cuFile versions.
        let mut descr = [0u8; 512];

        // Set type = CU_FILE_HANDLE_TYPE_OPAQUE_FD at offset 0
        unsafe {
            *(descr.as_mut_ptr() as *mut c_int) = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        }

        // Set fd in the union.
        // The union is after `type` (4 bytes). On x86_64 with pointer alignment,
        // the union starts at offset 8 (4 bytes padding after the int type field).
        unsafe {
            *(descr.as_mut_ptr().add(8) as *mut c_int) = fd;
        }

        let mut handle: *mut c_void = std::ptr::null_mut();
        let err = unsafe { cuFileHandleRegister(&mut handle, descr.as_mut_ptr() as *mut c_void) };
        if err.err != 0 {
            unsafe { cuFileDriverClose() };
            return Err(LoaderError::InitError(format!(
                "cuFileHandleRegister failed: err={}, cu_err={}",
                err.err, err.cu_err
            )));
        }

        Ok(Self { handle })
    }

    /// Read from NVMe directly to GPU memory.
    ///
    /// For best performance, the GPU buffer should be registered with
    /// `cufile_buf_register` first (otherwise cuFile uses internal bounce buffers).
    pub fn read(
        &self,
        gpu_ptr: *mut u8,
        size: usize,
        file_offset: usize,
        buf_offset: usize,
    ) -> Result<usize> {
        let bytes = unsafe {
            cuFileRead(
                self.handle,
                gpu_ptr as *mut c_void,
                size,
                file_offset as i64,
                buf_offset as i64,
            )
        };
        if bytes < 0 {
            return Err(LoaderError::FetchError(format!(
                "cuFileRead failed: {bytes}"
            )));
        }
        Ok(bytes as usize)
    }
}

impl Drop for CuFileSource {
    fn drop(&mut self) {
        unsafe {
            cuFileHandleDeregister(self.handle);
            cuFileDriverClose();
        }
    }
}

/// Register a GPU buffer with cuFile for direct DMA (avoids bounce buffers).
///
/// Call before cuFileRead for optimal performance. Must be followed by
/// `cufile_buf_deregister` before freeing the buffer.
#[inline]
pub fn cufile_buf_register(ptr: *const u8, size: usize) -> bool {
    let err = unsafe { cuFileBufRegister(ptr as *const c_void, size, 0) };
    err.err == 0
}

/// Deregister a GPU buffer from cuFile.
#[inline]
pub fn cufile_buf_deregister(ptr: *const u8) {
    unsafe {
        cuFileBufDeregister(ptr as *const c_void);
    }
}
