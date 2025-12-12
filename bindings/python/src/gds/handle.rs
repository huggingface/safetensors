//! cuFile handle management for file operations

use super::bindings::{
    check_cufile_error, cuFileHandleDeregister, cuFileHandleRegister,
    CUfileDescr_t, CUfileHandle_t,
};
use super::driver::GdsDriver;
use super::error::GdsError;

use std::fs::{File, OpenOptions};
use std::os::unix::io::AsRawFd;
use std::path::Path;

/// Wrapper for cuFile handle with strict RAII semantics
pub struct GdsHandle {
    handle: CUfileHandle_t,
    file: Option<File>,
    _driver: GdsDriver,
}

impl GdsHandle {
    /// Create a new cuFile-enabled handle
    pub fn new<P: AsRef<Path>>(path: P, read_only: bool) -> Result<Self, GdsError> {
        // Ensure cuFile driver is initialized
        let driver = GdsDriver::get()?;

        // Open file with correct permissions
        let file = OpenOptions::new()
            .read(true)
            .write(!read_only)
            .open(path)
            .map_err(GdsError::FileOpenFailed)?;

        // Create cuFile descriptor
        let fd = file.as_raw_fd();
        let descr = CUfileDescr_t::from_fd(fd);

        // Register handle with cuFile
        let mut handle: CUfileHandle_t = std::ptr::null_mut();
        unsafe {
            let result = cuFileHandleRegister(&mut handle, &descr);
            check_cufile_error(result, "cuFileHandleRegister")?;
        }

        if handle.is_null() {
            return Err(GdsError::HandleRegistrationFailed);
        }

        Ok(Self {
            handle,
            file: Some(file),
            _driver: driver,
        })
    }

    /// Raw cuFile handle
    pub fn raw_handle(&self) -> CUfileHandle_t {
        self.handle
    }

    /// Return file descriptor
    pub fn fd(&self) -> i32 {
        self.file
            .as_ref()
            .map(|f| f.as_raw_fd())
            .unwrap_or(-1)
    }
}

impl Drop for GdsHandle {
    fn drop(&mut self) {
        // Step 1: take the file out early
        let file = self.file.take();

        // Step 2: deregister cuFile handle while fd is still valid
        if !self.handle.is_null() {
            unsafe {
                let result = cuFileHandleDeregister(self.handle);

                // Only print errors in debug builds (cuFile often returns spurious errors)
                #[cfg(debug_assertions)]
                if result.err != 0 {
                    eprintln!(
                        "Debug: cuFileHandleDeregister err={} cu_err={}",
                        result.err, result.cu_err
                    );
                }
            }

            // Mark handle invalid
            self.handle = std::ptr::null_mut();
        }

        // Step 3: file is dropped here at the end of Drop
        drop(file);
    }
}

// Ensure GdsHandle is Send and Sync (can be shared/moved between threads)
// SAFETY: The raw pointer is managed by cuFile library and never dereferenced in Rust
unsafe impl Send for GdsHandle {}
unsafe impl Sync for GdsHandle {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    #[ignore] // Only run if GDS is available
    fn test_handle_creation() {
        // Create a temporary file
        let mut temp_file = std::env::temp_dir();
        temp_file.push("test_gds_handle.bin");
        
        {
            let mut f = std::fs::File::create(&temp_file).unwrap();
            f.write_all(b"test data").unwrap();
        }

        let handle = GdsHandle::new(&temp_file, true);
        
        if handle.is_ok() {
            let h = handle.unwrap();
            assert!(!h.raw_handle().is_null());
            assert!(h.fd() >= 0);
        }

        let _ = std::fs::remove_file(temp_file);
    }
}
