//! GPU Direct Storage (GDS) module for NVIDIA cuFile API
//! 
//! This module provides zero-copy direct storage access between NVMe SSDs
//! and GPU memory, bypassing the CPU for utilizing high-throughput storage.

#[cfg(target_os = "linux")]
pub mod bindings;
#[cfg(target_os = "linux")]
pub mod driver;
#[cfg(target_os = "linux")]
pub mod error;
#[cfg(target_os = "linux")]
pub mod handle;
#[cfg(target_os = "linux")]
pub mod storage;

#[cfg(target_os = "linux")]
pub use driver::GdsDriver;
#[cfg(target_os = "linux")]
pub use error::GdsError;
#[cfg(target_os = "linux")]
pub use handle::GdsHandle;
#[cfg(target_os = "linux")]
pub use storage::GdsStorage;

// Suppress unused warnings for conditional compilation
#[allow(unused)]
use driver::GdsDriver as _GdsDriver;
#[allow(unused)]
use error::GdsError as _GdsError;
#[allow(unused)]
use handle::GdsHandle as _GdsHandle;
