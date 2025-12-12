//! GPU Direct Storage (GDS) module for NVIDIA cuFile API
//! 
//! This module provides zero-copy direct storage access between NVMe SSDs
//! and GPU memory, bypassing the CPU for utilizing high-throughput storage.

#[cfg(all(target_os = "linux", feature = "cuda-gds"))]
pub mod bindings;
#[cfg(all(target_os = "linux", feature = "cuda-gds"))]
pub mod driver;
#[cfg(all(target_os = "linux", feature = "cuda-gds"))]
pub mod error;
#[cfg(all(target_os = "linux", feature = "cuda-gds"))]
pub mod handle;
#[cfg(all(target_os = "linux", feature = "cuda-gds"))]
pub mod storage;

#[cfg(all(target_os = "linux", feature = "cuda-gds"))]
pub use driver::GdsDriver;
#[cfg(all(target_os = "linux", feature = "cuda-gds"))]
pub use error::GdsError;
#[cfg(all(target_os = "linux", feature = "cuda-gds"))]
pub use handle::GdsHandle;
#[cfg(all(target_os = "linux", feature = "cuda-gds"))]
pub use storage::GdsStorage;

// Suppress unused warnings for conditional compilation
#[allow(unused)]
use driver::GdsDriver as _GdsDriver;
#[allow(unused)]
use error::GdsError as _GdsError;
#[allow(unused)]
use handle::GdsHandle as _GdsHandle;
