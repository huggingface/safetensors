//! Error types for GDS operations

use std::fmt;

/// Errors that can occur during GDS operations
#[derive(Debug)]
pub enum GdsError {
    /// Failed to initialize the cuFile driver
    DriverInitFailed,
    /// cuFile API returned an error
    CuFileError(i32, i32),
    /// Failed to register file handle with cuFile
    HandleRegistrationFailed,
    /// Failed to register buffer with cuFile
    BufferRegistrationFailed,
    /// cuFileRead operation failed
    ReadFailed(isize),
    /// cuFileWrite operation failed
    WriteFailed(isize),
    /// GDS is only supported on CUDA devices
    InvalidDevice,
    /// Failed to open file
    FileOpenFailed(std::io::Error),
    /// Invalid file descriptor
    InvalidFileDescriptor,
    /// Buffer alignment requirements not met
    AlignmentError(String),
}

impl fmt::Display for GdsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GdsError::DriverInitFailed => {
                write!(f, "Failed to initialize cuFile driver. Is libcufile.so available?")
            }
            GdsError::CuFileError(err, cu_err) => {
                write!(
                    f,
                    "cuFile error: err={}, cuda_err={} (check NVIDIA GDS documentation)",
                    err, cu_err
                )
            }
            GdsError::HandleRegistrationFailed => {
                write!(f, "Failed to register file handle with cuFile")
            }
            GdsError::BufferRegistrationFailed => {
                write!(f, "Failed to register GPU buffer with cuFile")
            }
            GdsError::ReadFailed(ret) => {
                write!(f, "cuFileRead failed with return code: {}", ret)
            }
            GdsError::WriteFailed(ret) => {
                write!(f, "cuFileWrite failed with return code: {}", ret)
            }
            GdsError::InvalidDevice => {
                write!(f, "GDS is only supported on CUDA devices")
            }
            GdsError::FileOpenFailed(e) => {
                write!(f, "Failed to open file for GDS: {}", e)
            }
            GdsError::InvalidFileDescriptor => {
                write!(f, "Invalid file descriptor")
            }
            GdsError::AlignmentError(msg) => {
                write!(f, "Buffer alignment error: {}", msg)
            }
        }
    }
}

impl std::error::Error for GdsError {}

/// Convert GdsError to PyErr for Python integration
impl From<GdsError> for pyo3::PyErr {
    fn from(err: GdsError) -> Self {
        pyo3::exceptions::PyRuntimeError::new_err(err.to_string())
    }
}
