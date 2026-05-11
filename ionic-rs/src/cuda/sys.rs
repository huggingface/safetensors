//! Raw `libcuda.so.1` FFI.
//!
//! We link against the driver API (not the runtime API) for two reasons:
//! it's the stable ABI shipped by every install, and it supports
//! primary-context attach (`cuDevicePrimaryCtxRetain`), letting us co-exist
//! with PyTorch's context.
//!
//! Symbols are resolved at first use; absent driver yields
//! `Error::CudaUnavailable` on the first CUDA call.

use std::ffi::{c_char, c_int, c_uint, c_void};

use libloading::Library;

use crate::error::{Error, Result};

// ── C types ─────────────────────────────────────────────────────────────

pub type CUresult = c_int;
pub type CUdevice = c_int;
pub type CUcontext = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUevent = *mut c_void;
/// 64-bit device pointer regardless of host word size.
pub type CUdeviceptr = u64;

pub const CUDA_SUCCESS: CUresult = 0;
/// Returned by `cuEventQuery` when captured work is still outstanding.
pub const CUDA_ERROR_NOT_READY: CUresult = 600;

/// Skip implicit serialization with the legacy default stream.
pub const CU_STREAM_NON_BLOCKING: c_uint = 0x1;

/// Cheaper to record (we only use events as happens-before fences).
pub const CU_EVENT_DISABLE_TIMING: c_uint = 0x2;

/// Pinning is visible to all CUDA contexts in the process.
pub const CU_MEMHOSTALLOC_PORTABLE: c_uint = 0x1;

/// Keeping `_lib` alive is load-bearing: dropping it unloads the .so and
/// invalidates every function pointer.
#[allow(non_snake_case)]
pub struct CudaLib {
    _lib: Library,
    pub cuInit: unsafe extern "C" fn(c_uint) -> CUresult,
    pub cuDeviceGetCount: unsafe extern "C" fn(*mut c_int) -> CUresult,
    pub cuDeviceGet: unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult,
    pub cuDeviceGetName: unsafe extern "C" fn(*mut c_char, c_int, CUdevice) -> CUresult,
    pub cuDeviceGetPCIBusId: unsafe extern "C" fn(*mut c_char, c_int, CUdevice) -> CUresult,
    pub cuDevicePrimaryCtxRetain:
        unsafe extern "C" fn(*mut CUcontext, CUdevice) -> CUresult,
    pub cuDevicePrimaryCtxRelease: unsafe extern "C" fn(CUdevice) -> CUresult,
    pub cuCtxPushCurrent: unsafe extern "C" fn(CUcontext) -> CUresult,
    pub cuCtxPopCurrent: unsafe extern "C" fn(*mut CUcontext) -> CUresult,
    pub cuStreamCreate: unsafe extern "C" fn(*mut CUstream, c_uint) -> CUresult,
    pub cuStreamDestroy: unsafe extern "C" fn(CUstream) -> CUresult,
    pub cuStreamSynchronize: unsafe extern "C" fn(CUstream) -> CUresult,
    pub cuEventCreate: unsafe extern "C" fn(*mut CUevent, c_uint) -> CUresult,
    pub cuEventDestroy: unsafe extern "C" fn(CUevent) -> CUresult,
    pub cuEventRecord: unsafe extern "C" fn(CUevent, CUstream) -> CUresult,
    pub cuEventSynchronize: unsafe extern "C" fn(CUevent) -> CUresult,
    pub cuEventQuery: unsafe extern "C" fn(CUevent) -> CUresult,
    pub cuMemAlloc: unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult,
    pub cuMemFree: unsafe extern "C" fn(CUdeviceptr) -> CUresult,
    pub cuMemHostAlloc: unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> CUresult,
    pub cuMemFreeHost: unsafe extern "C" fn(*mut c_void) -> CUresult,
    pub cuMemcpyHtoDAsync:
        unsafe extern "C" fn(CUdeviceptr, *const c_void, usize, CUstream) -> CUresult,
    /// Test-only synchronous D2H readback.
    pub cuMemcpyDtoH: unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult,
    pub cuGetErrorString: unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult,
}

macro_rules! load_fn {
    ($lib:expr, $name:literal) => {{
        let sym: libloading::Symbol<_> = unsafe {
            $lib.get(concat!($name, "\0").as_bytes())
        }
        .map_err(|e| {
            Error::CudaUnavailable(format!("resolving {}: {e}", $name))
        })?;
        *sym
    }};
}

impl CudaLib {
    /// Open `libcuda.so.1` and resolve every symbol. Returns
    /// `CudaUnavailable` if the library or any symbol is missing. Does *not*
    /// call `cuInit`.
    pub fn load() -> Result<Self> {
        // SAFETY: loading the system's libcuda.so.1.
        let lib = unsafe { Library::new("libcuda.so.1") }
            .map_err(|e| Error::CudaUnavailable(format!("dlopen libcuda.so.1: {e}")))?;

        Ok(Self {
            cuInit: load_fn!(&lib, "cuInit"),
            cuDeviceGetCount: load_fn!(&lib, "cuDeviceGetCount"),
            cuDeviceGet: load_fn!(&lib, "cuDeviceGet"),
            cuDeviceGetName: load_fn!(&lib, "cuDeviceGetName"),
            cuDeviceGetPCIBusId: load_fn!(&lib, "cuDeviceGetPCIBusId"),
            cuDevicePrimaryCtxRetain: load_fn!(&lib, "cuDevicePrimaryCtxRetain"),
            cuDevicePrimaryCtxRelease: load_fn!(&lib, "cuDevicePrimaryCtxRelease"),
            // Versioned suffix `_v2` matches the ABI the driver has exported
            // since CUDA 4.x. The unsuffixed symbols are deprecated shims.
            cuCtxPushCurrent: load_fn!(&lib, "cuCtxPushCurrent_v2"),
            cuCtxPopCurrent: load_fn!(&lib, "cuCtxPopCurrent_v2"),
            cuStreamCreate: load_fn!(&lib, "cuStreamCreate"),
            cuStreamDestroy: load_fn!(&lib, "cuStreamDestroy_v2"),
            cuStreamSynchronize: load_fn!(&lib, "cuStreamSynchronize"),
            cuEventCreate: load_fn!(&lib, "cuEventCreate"),
            cuEventDestroy: load_fn!(&lib, "cuEventDestroy_v2"),
            cuEventRecord: load_fn!(&lib, "cuEventRecord"),
            cuEventSynchronize: load_fn!(&lib, "cuEventSynchronize"),
            cuEventQuery: load_fn!(&lib, "cuEventQuery"),
            cuMemAlloc: load_fn!(&lib, "cuMemAlloc_v2"),
            cuMemFree: load_fn!(&lib, "cuMemFree_v2"),
            cuMemHostAlloc: load_fn!(&lib, "cuMemHostAlloc"),
            cuMemFreeHost: load_fn!(&lib, "cuMemFreeHost"),
            cuMemcpyHtoDAsync: load_fn!(&lib, "cuMemcpyHtoDAsync_v2"),
            cuMemcpyDtoH: load_fn!(&lib, "cuMemcpyDtoH_v2"),
            cuGetErrorString: load_fn!(&lib, "cuGetErrorString"),
            _lib: lib,
        })
    }
}

// SAFETY: Raw function pointers are trivially `Send + Sync`. The CUDA driver
// API is documented as thread-safe for the symbols we use here (device
// queries, context push/pop, stream/event ops, memory alloc/free/copy). The
// `libloading::Library` itself is `Send + Sync` on unix.
unsafe impl Send for CudaLib {}
unsafe impl Sync for CudaLib {}
