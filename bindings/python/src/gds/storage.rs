//! GDS storage implementation for safetensors

use super::bindings::cuFileRead;
use super::error::GdsError;
use super::handle::GdsHandle;
use std::path::PathBuf;

/// GPU Direct Storage backend for safetensors
pub struct GdsStorage {
    handle: GdsHandle,
    path: PathBuf,
    file_size: usize,
}

impl GdsStorage {
    /// Create a new GDS storage instance
    ///
    /// # Arguments
    /// * `path` - Path to the safetensors file
    pub fn new(path: PathBuf) -> Result<Self, GdsError> {
        // Get file size
        let metadata = std::fs::metadata(&path).map_err(GdsError::FileOpenFailed)?;
        let file_size = metadata.len() as usize;

        // Open file with GDS
        let handle = GdsHandle::new(&path, true)?;

        Ok(GdsStorage {
            handle,
            path,
            file_size,
        })
    }

    /// Get the file path
    pub fn path(&self) -> &PathBuf {
        &self.path
    }

    /// Get the file size
    pub fn size(&self) -> usize {
        self.file_size
    }

    /// Read data directly to GPU memory
    ///
    /// # Arguments
    /// * `gpu_ptr` - Pointer to GPU memory (must be CUDA device memory)
    /// * `size` - Number of bytes to read
    /// * `file_offset` - Offset in the file to read from
    ///
    /// # Safety
    /// The caller must ensure that:
    /// - `gpu_ptr` points to valid CUDA device memory
    /// - The GPU buffer has at least `size` bytes available
    /// - The file has at least `file_offset + size` bytes
    pub unsafe fn read_to_device(
        &self,
        gpu_ptr: *mut std::ffi::c_void,
        size: usize,
        file_offset: usize,
    ) -> Result<usize, GdsError> {
        if gpu_ptr.is_null() {
            return Err(GdsError::InvalidFileDescriptor);
        }

        // Check bounds
        if file_offset + size > self.file_size {
            return Err(GdsError::ReadFailed(-1));
        }

        // Perform GDS read
        let bytes_read = cuFileRead(
            self.handle.raw_handle(),
            gpu_ptr,
            size,
            file_offset as i64,
            0, // dev_offset: offset within the GPU buffer (0 = start)
        );

        if bytes_read < 0 {
            Err(GdsError::ReadFailed(bytes_read))
        } else if bytes_read as usize != size {
            eprintln!(
                "Warning: cuFileRead returned {} bytes, expected {}",
                bytes_read, size
            );
            Ok(bytes_read as usize)
        } else {
            Ok(bytes_read as usize)
        }
    }
}

// Ensure GdsStorage is Send and Sync (can be shared/moved between threads)
// SAFETY: The handle is thread-safe as cuFile manages the internal state
unsafe impl Send for GdsStorage {}
unsafe impl Sync for GdsStorage {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    #[ignore] // Only run if GDS is available
    fn test_storage_creation() {
        let mut temp_file = std::env::temp_dir();
        temp_file.push("test_gds_storage.bin");
        
        {
            let mut f = std::fs::File::create(&temp_file).unwrap();
            f.write_all(&vec![0u8; 1024]).unwrap();
        }

        let storage = GdsStorage::new(temp_file.clone());
        
        if storage.is_ok() {
            let s = storage.unwrap();
            assert_eq!(s.size(), 1024);
            assert_eq!(s.path(), &temp_file);
        }

        let _ = std::fs::remove_file(temp_file);
    }
}
