//! Safe Rust wrappers over the CUDA driver API.
//!
//! We attach to the device's primary context (the one PyTorch retains) via
//! `CuContext::primary_retain(dev)`; creating our own with `cuCtxCreate` would
//! split ownership. Scope context-current calls with `ctx.with_current(...)`.

pub mod sys;

use std::ffi::{c_char, CStr};
use std::sync::OnceLock;

use sys::{CudaLib, CUcontext, CUdevice, CUdeviceptr, CUevent, CUresult, CUstream, CUDA_SUCCESS};

use crate::error::{Error, Result};

fn check(rc: CUresult, symbol: &'static str) -> Result<()> {
    if rc == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(Error::Cuda {
            symbol,
            code: rc as i32,
        })
    }
}

pub fn lib() -> Result<&'static CudaLib> {
    // OnceLock<Result<_, String>> rather than the crate Result alias: a flat
    // String is cheap to clone on each failure path.
    static CUDA_LIB: OnceLock<std::result::Result<CudaLib, String>> = OnceLock::new();
    let res = CUDA_LIB.get_or_init(|| {
        CudaLib::load()
            .and_then(|lib| {
                let rc = unsafe { (lib.cuInit)(0) };
                check(rc, "cuInit").map(|_| lib)
            })
            .map_err(|e| e.to_string())
    });
    res.as_ref()
        .map_err(|s| Error::CudaUnavailable(s.clone()))
}

#[derive(Clone, Copy, Debug)]
pub struct CuDevice(CUdevice);

impl CuDevice {
    pub fn count() -> Result<i32> {
        let cuda = lib()?;
        let mut n: i32 = 0;
        check(unsafe { (cuda.cuDeviceGetCount)(&mut n) }, "cuDeviceGetCount")?;
        Ok(n)
    }

    pub fn get(ordinal: i32) -> Result<Self> {
        let cuda = lib()?;
        let mut dev: CUdevice = 0;
        check(
            unsafe { (cuda.cuDeviceGet)(&mut dev, ordinal) },
            "cuDeviceGet",
        )?;
        Ok(Self(dev))
    }

    pub fn name(self) -> Result<String> {
        let cuda = lib()?;
        let mut buf = [0 as c_char; 256];
        check(
            unsafe { (cuda.cuDeviceGetName)(buf.as_mut_ptr(), buf.len() as i32, self.0) },
            "cuDeviceGetName",
        )?;
        // SAFETY: driver writes a NUL-terminated string of length <= buf.len().
        let cstr = unsafe { CStr::from_ptr(buf.as_ptr()) };
        Ok(cstr.to_string_lossy().into_owned())
    }

    /// Lowercase `DDDD:BB:DD.F` (the driver returns uppercase; sysfs paths
    /// are case-sensitive and lowercase).
    pub fn pci_bus_id(self) -> Result<String> {
        let cuda = lib()?;
        let mut buf = [0 as c_char; 32];
        check(
            unsafe { (cuda.cuDeviceGetPCIBusId)(buf.as_mut_ptr(), buf.len() as i32, self.0) },
            "cuDeviceGetPCIBusId",
        )?;
        let cstr = unsafe { CStr::from_ptr(buf.as_ptr()) };
        Ok(cstr.to_string_lossy().to_lowercase())
    }

    pub fn as_raw(self) -> CUdevice {
        self.0
    }
}

/// RAII handle for a retained primary context. `Drop` releases the retention;
/// PyTorch's refcount on the same context is unaffected.
pub struct CuContext {
    device: CuDevice,
    ctx: CUcontext,
}

// SAFETY: a primary context can be made current on multiple threads
// independently. Sharing `&CuContext` so each thread calls `with_current`
// is the documented use pattern.
unsafe impl Send for CuContext {}
unsafe impl Sync for CuContext {}

impl CuContext {
    pub fn primary_retain(device: CuDevice) -> Result<Self> {
        let cuda = lib()?;
        let mut ctx: CUcontext = std::ptr::null_mut();
        check(
            unsafe { (cuda.cuDevicePrimaryCtxRetain)(&mut ctx, device.as_raw()) },
            "cuDevicePrimaryCtxRetain",
        )?;
        Ok(Self { device, ctx })
    }

    /// Run `f` with this context current. The pop runs even on panic.
    pub fn with_current<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce() -> Result<R>,
    {
        let cuda = lib()?;
        check(
            unsafe { (cuda.cuCtxPushCurrent)(self.ctx) },
            "cuCtxPushCurrent_v2",
        )?;
        struct PopGuard;
        impl Drop for PopGuard {
            fn drop(&mut self) {
                if let Ok(cuda) = lib() {
                    let mut _out: CUcontext = std::ptr::null_mut();
                    unsafe { (cuda.cuCtxPopCurrent)(&mut _out) };
                }
            }
        }
        let _guard = PopGuard;
        f()
    }

    pub fn device(&self) -> CuDevice {
        self.device
    }
}

impl Drop for CuContext {
    fn drop(&mut self) {
        if let Ok(cuda) = lib() {
            // Release (not destroy): decrement the driver's refcount; PyTorch's
            // retention keeps the context alive.
            unsafe { (cuda.cuDevicePrimaryCtxRelease)(self.device.as_raw()) };
        }
    }
}

pub struct CuStream {
    stream: CUstream,
}

// SAFETY: driver-documented thread-safe stream ops; the handle is opaque.
unsafe impl Send for CuStream {}
unsafe impl Sync for CuStream {}

impl CuStream {
    /// Non-blocking so it doesn't implicitly serialize with the legacy
    /// default stream (required for overlap with torch's compute streams).
    pub fn new() -> Result<Self> {
        let cuda = lib()?;
        let mut s: CUstream = std::ptr::null_mut();
        check(
            unsafe { (cuda.cuStreamCreate)(&mut s, sys::CU_STREAM_NON_BLOCKING) },
            "cuStreamCreate",
        )?;
        Ok(Self { stream: s })
    }

    pub fn synchronize(&self) -> Result<()> {
        let cuda = lib()?;
        check(
            unsafe { (cuda.cuStreamSynchronize)(self.stream) },
            "cuStreamSynchronize",
        )
    }

    pub fn as_raw(&self) -> CUstream {
        self.stream
    }
}

impl Drop for CuStream {
    fn drop(&mut self) {
        if let Ok(cuda) = lib() {
            unsafe { (cuda.cuStreamDestroy)(self.stream) };
        }
    }
}

pub struct CuEvent {
    event: CUevent,
}

// SAFETY: event ops are driver-documented as thread-safe.
unsafe impl Send for CuEvent {}
unsafe impl Sync for CuEvent {}

impl CuEvent {
    /// Timing-disabled; we only use these as happens-before fences.
    pub fn new() -> Result<Self> {
        let cuda = lib()?;
        let mut e: CUevent = std::ptr::null_mut();
        check(
            unsafe { (cuda.cuEventCreate)(&mut e, sys::CU_EVENT_DISABLE_TIMING) },
            "cuEventCreate",
        )?;
        Ok(Self { event: e })
    }

    pub fn record(&self, stream: &CuStream) -> Result<()> {
        let cuda = lib()?;
        check(
            unsafe { (cuda.cuEventRecord)(self.event, stream.as_raw()) },
            "cuEventRecord",
        )
    }

    pub fn synchronize(&self) -> Result<()> {
        let cuda = lib()?;
        check(
            unsafe { (cuda.cuEventSynchronize)(self.event) },
            "cuEventSynchronize",
        )
    }

    /// Non-blocking: `Ok(true)` if complete, `Ok(false)` if still in flight.
    pub fn query(&self) -> Result<bool> {
        let cuda = lib()?;
        let rc = unsafe { (cuda.cuEventQuery)(self.event) };
        match rc {
            sys::CUDA_SUCCESS => Ok(true),
            sys::CUDA_ERROR_NOT_READY => Ok(false),
            _ => Err(Error::Cuda {
                symbol: "cuEventQuery",
                code: rc as i32,
            }),
        }
    }
}

impl Drop for CuEvent {
    fn drop(&mut self) {
        if let Ok(cuda) = lib() {
            unsafe { (cuda.cuEventDestroy)(self.event) };
        }
    }
}

/// Owned device allocation. `Drop` calls `cuMemFree_v2`.
pub struct DeviceBuf {
    ptr: CUdeviceptr,
    bytes: usize,
}

impl DeviceBuf {
    pub fn alloc(bytes: usize) -> Result<Self> {
        let cuda = lib()?;
        let mut ptr: CUdeviceptr = 0;
        check(
            unsafe { (cuda.cuMemAlloc)(&mut ptr, bytes) },
            "cuMemAlloc_v2",
        )?;
        Ok(Self { ptr, bytes })
    }

    pub fn as_device_ptr(&self) -> CUdeviceptr {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.bytes
    }
}

impl Drop for DeviceBuf {
    fn drop(&mut self) {
        if let Ok(cuda) = lib() {
            unsafe { (cuda.cuMemFree)(self.ptr) };
        }
    }
}

/// Page-locked host allocation. Pageable host memory silently stages through
/// the driver's bounce buffer and serializes; pinned is required for full
/// PCIe bandwidth.
pub struct PinnedBuf {
    ptr: *mut u8,
    bytes: usize,
}

// SAFETY: `malloc`-style memory with a page-lock attribute; thread-safe
// modulo ordinary races.
unsafe impl Send for PinnedBuf {}
unsafe impl Sync for PinnedBuf {}

impl PinnedBuf {
    pub fn alloc(bytes: usize) -> Result<Self> {
        let cuda = lib()?;
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        check(
            unsafe {
                (cuda.cuMemHostAlloc)(&mut ptr, bytes, sys::CU_MEMHOSTALLOC_PORTABLE)
            },
            "cuMemHostAlloc",
        )?;
        Ok(Self {
            ptr: ptr as *mut u8,
            bytes,
        })
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr as *const u8
    }

    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Callers must not mutate the slice while a `cuMemcpyHtoDAsync` from it
    /// is in flight.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: we own the allocation of size `bytes`.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.bytes) }
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: we own the allocation of size `bytes`.
        unsafe { std::slice::from_raw_parts(self.ptr, self.bytes) }
    }

    pub fn len(&self) -> usize {
        self.bytes
    }
}

impl Drop for PinnedBuf {
    fn drop(&mut self) {
        if let Ok(cuda) = lib() {
            unsafe { (cuda.cuMemFreeHost)(self.ptr as *mut std::ffi::c_void) };
        }
    }
}

/// `src` must remain valid until the copy completes (use `PinnedBuf`).
/// Context must be current.
pub fn memcpy_h2d_async(
    dst: CUdeviceptr,
    src: *const u8,
    bytes: usize,
    stream: &CuStream,
) -> Result<()> {
    let cuda = lib()?;
    check(
        unsafe {
            (cuda.cuMemcpyHtoDAsync)(dst, src as *const std::ffi::c_void, bytes, stream.as_raw())
        },
        "cuMemcpyHtoDAsync_v2",
    )
}

/// Test-only D2H readback. Context must be current.
///
/// # Safety
///
/// `dst` must be writable for at least `bytes`; `src` must reach at least
/// `bytes`.
pub unsafe fn memcpy_d2h(dst: *mut u8, src: CUdeviceptr, bytes: usize) -> Result<()> {
    let cuda = lib()?;
    check(
        unsafe { (cuda.cuMemcpyDtoH)(dst as *mut std::ffi::c_void, src, bytes) },
        "cuMemcpyDtoH_v2",
    )
}
