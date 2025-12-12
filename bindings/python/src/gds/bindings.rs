//! FFI bindings to NVIDIA cuFile library (libcufile.so)
//!
//! These are low-level bindings to the cuFile API for GPU Direct Storage.
//! See: https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html

#![allow(non_camel_case_types)]
#![allow(dead_code)]

use std::os::raw::{c_int, c_longlong, c_void};

// Use libc's size_t instead of unstable std::ffi::c_size_t
type c_size_t = libc::size_t;

/// cuFile error structure
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CUfileError_t {
    /// cuFile specific error code
    pub err: c_int,
    /// CUDA runtime error code
    pub cu_err: c_int,
}

/// Opaque handle to a cuFile descriptor
pub type CUfileHandle_t = *mut c_void;

/// Union for file descriptor or handle
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUfileDescr_handle {
    /// POSIX file descriptor
    pub fd: c_int,
    /// Opaque handle
    pub handle: *mut c_void,
}

/// File descriptor for cuFile operations
#[repr(C)]
pub struct CUfileDescr_t {
    /// Type of descriptor (1 = file descriptor, 2 = handle)
    pub type_: c_int,
    /// File descriptor or handle
    pub handle: CUfileDescr_handle,
    /// File system operations (reserved, set to null)
    pub fs_ops: *mut c_void,
}

impl CUfileDescr_t {
    /// Create a new descriptor from a file descriptor
    pub fn from_fd(fd: c_int) -> Self {
        CUfileDescr_t {
            type_: 1, // CU_FILE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
            handle: CUfileDescr_handle { fd },
            fs_ops: std::ptr::null_mut(),
        }
    }
}

// FFI declarations for cuFile library
#[link(name = "cufile")]
extern "C" {
    /// Initialize the cuFile driver
    pub fn cuFileDriverOpen() -> CUfileError_t;

    /// Shutdown the cuFile driver
    pub fn cuFileDriverClose() -> CUfileError_t;

    /// Register a file handle with cuFile
    pub fn cuFileHandleRegister(
        handle: *mut CUfileHandle_t,
        descr: *const CUfileDescr_t,
    ) -> CUfileError_t;

    /// Deregister a file handle
    pub fn cuFileHandleDeregister(handle: CUfileHandle_t) -> CUfileError_t;

    /// Register a GPU buffer for use with cuFile
    pub fn cuFileBufRegister(
        buf: *const c_void,
        size: c_size_t,
        flags: c_int,
    ) -> CUfileError_t;

    /// Deregister a GPU buffer
    pub fn cuFileBufDeregister(buf: *const c_void) -> CUfileError_t;

    /// Read from file directly to GPU memory
    /// Returns number of bytes read on success, negative on error
    pub fn cuFileRead(
        handle: CUfileHandle_t,
        buf: *mut c_void,
        size: c_size_t,
        file_offset: c_longlong,
        dev_offset: c_longlong,
    ) -> isize;

    /// Write from GPU memory directly to file
    /// Returns number of bytes written on success, negative on error
    pub fn cuFileWrite(
        handle: CUfileHandle_t,
        buf: *const c_void,
        size: c_size_t,
        file_offset: c_longlong,
        dev_offset: c_longlong,
    ) -> isize;
}

/// Helper function to check if a cuFile error occurred
pub fn check_cufile_error(err: CUfileError_t, operation: &str) -> Result<(), crate::gds::error::GdsError> {
    if err.err != 0 {
        eprintln!(
            "cuFile operation '{}' failed: err={}, cu_err={}",
            operation, err.err, err.cu_err
        );
        Err(crate::gds::error::GdsError::CuFileError(err.err, err.cu_err))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_descriptor_from_fd() {
        let descr = CUfileDescr_t::from_fd(42);
        assert_eq!(descr.type_, 1);
        unsafe {
            assert_eq!(descr.handle.fd, 42);
        }
    }
}
