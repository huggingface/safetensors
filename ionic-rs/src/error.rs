//! Errors produced by the ionic-rs engine.

#[derive(Debug)]
pub enum Error {
    UnknownTensor(String),
    /// CUDA driver call failed; carries the driver's `cuResult` code and the
    /// symbol that produced it.
    Cuda { symbol: &'static str, code: i32 },
    /// `libcuda.so.1` not loadable at runtime.
    CudaUnavailable(String),
    IoUring { op: &'static str, errno: i32 },
    /// Failure parsing sysfs for NUMA topology. Non-fatal; callers fall back
    /// to not pinning.
    NumaProbe(String),
    Other(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::UnknownTensor(name) => write!(f, "unknown tensor: {name}"),
            Error::Cuda { symbol, code } => write!(f, "CUDA {symbol} failed with code {code}"),
            Error::CudaUnavailable(msg) => write!(f, "CUDA driver unavailable: {msg}"),
            Error::IoUring { op, errno } => write!(f, "io_uring {op} failed with errno {errno}"),
            Error::NumaProbe(msg) => write!(f, "NUMA probe failed: {msg}"),
            Error::Other(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
