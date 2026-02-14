//! Direct CUDA runtime FFI helpers.
//!
//! Provides safe wrappers around CUDA runtime functions for device management,
//! memory allocation, streams, events, and memory copies.

use std::ffi::{c_int, c_uint, c_void};

type CudaResult<T> = super::Result<T>;

extern "C" {
    fn cudaSetDevice(device: c_int) -> c_int;
    fn cudaGetDevice(device: *mut c_int) -> c_int;
    fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> c_int;
    fn cudaFree(devPtr: *mut c_void) -> c_int;
    fn cudaHostAlloc(pHost: *mut *mut c_void, size: usize, flags: c_uint) -> c_int;
    fn cudaFreeHost(ptr: *mut c_void) -> c_int;
    fn cudaStreamCreateWithFlags(pStream: *mut *mut c_void, flags: c_uint) -> c_int;
    fn cudaStreamSynchronize(stream: *mut c_void) -> c_int;
    fn cudaStreamDestroy(stream: *mut c_void) -> c_int;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: c_int) -> c_int;
    fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: c_int,
        stream: *mut c_void,
    ) -> c_int;
    fn cudaEventCreateWithFlags(event: *mut *mut c_void, flags: c_uint) -> c_int;
    fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> c_int;
    fn cudaEventQuery(event: *mut c_void) -> c_int;
    fn cudaEventSynchronize(event: *mut c_void) -> c_int;
    fn cudaEventDestroy(event: *mut c_void) -> c_int;
    fn cudaDeviceGetDefaultMemPool(pool: *mut *mut c_void, device: c_int) -> c_int;
    fn cudaMallocFromPoolAsync(
        ptr: *mut *mut c_void,
        size: usize,
        pool: *mut c_void,
        stream: *mut c_void,
    ) -> c_int;
    fn cudaFreeAsync(devPtr: *mut c_void, stream: *mut c_void) -> c_int;
    fn cudaMallocAsync(devPtr: *mut *mut c_void, size: usize, stream: *mut c_void) -> c_int;
    fn cudaDeviceEnablePeerAccess(peerDevice: c_int, flags: c_uint) -> c_int;
}

const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
const CUDA_MEMCPY_DEVICE_TO_DEVICE: c_int = 3;
const CUDA_STREAM_NON_BLOCKING: c_uint = 1;
const CUDA_HOST_ALLOC_PORTABLE: c_uint = 1;
const CUDA_EVENT_DISABLE_TIMING: c_uint = 2;
/// cudaErrorPeerAccessAlreadyEnabled — not a real error.
const CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED: c_int = 704;

/// Set the current CUDA device for subsequent operations.
pub fn set_cuda_device(device_id: usize) -> CudaResult<()> {
    let ret = unsafe { cudaSetDevice(device_id as c_int) };
    if ret != 0 {
        return Err(super::LoaderError::FetchError(format!(
            "cudaSetDevice({device_id}) failed: {ret}"
        )));
    }
    Ok(())
}

/// Get the current CUDA device index.
fn get_current_device() -> CudaResult<usize> {
    let mut device: c_int = 0;
    let ret = unsafe { cudaGetDevice(&mut device) };
    if ret != 0 {
        return Err(super::LoaderError::FetchError(format!(
            "cudaGetDevice failed: {ret}"
        )));
    }
    Ok(device as usize)
}

/// Enable peer access between two CUDA devices (for D2D copies).
pub fn cuda_enable_peer_access(peer_device: usize) -> super::Result<()> {
    let ret = unsafe { cudaDeviceEnablePeerAccess(peer_device as c_int, 0) };
    if ret != 0 && ret != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED {
        return Err(super::LoaderError::FetchError(format!(
            "cudaDeviceEnablePeerAccess({peer_device}) failed: {ret}"
        )));
    }
    Ok(())
}

/// Allocate GPU device memory.
pub(crate) fn cuda_malloc(size: usize) -> CudaResult<*mut u8> {
    let mut ptr: *mut c_void = std::ptr::null_mut();
    let ret = unsafe { cudaMalloc(&mut ptr, size) };
    if ret != 0 || ptr.is_null() {
        return Err(super::LoaderError::FetchError(format!(
            "cudaMalloc({size}) failed: {ret}"
        )));
    }
    Ok(ptr as *mut u8)
}

/// Allocate GPU device memory asynchronously on the given stream.
///
/// Uses the CUDA memory pool allocator, which is faster than `cudaMalloc`
/// for frequent allocations by reusing previously freed memory.
/// The returned pointer is compatible with `cuda_free()`.
#[allow(dead_code)]
pub(crate) fn cuda_malloc_async(size: usize, stream: *mut c_void) -> CudaResult<*mut u8> {
    let mut ptr: *mut c_void = std::ptr::null_mut();
    let ret = unsafe { cudaMallocAsync(&mut ptr, size, stream) };
    if ret != 0 || ptr.is_null() {
        return Err(super::LoaderError::FetchError(format!(
            "cudaMallocAsync({size}) failed: {ret}"
        )));
    }
    Ok(ptr as *mut u8)
}

/// Free GPU device memory.
pub(crate) fn cuda_free(ptr: *mut u8) {
    if !ptr.is_null() {
        unsafe {
            cudaFree(ptr as *mut c_void);
        }
    }
}

/// Allocate GPU memory and wrap as an owned Buffer.
///
/// The buffer stores the current CUDA device index.
pub fn cuda_alloc_buffer(size: usize) -> CudaResult<super::Buffer> {
    let device_id = get_current_device()?;
    let ptr = cuda_malloc(size)?;
    Ok(unsafe { super::Buffer::from_raw_cuda(ptr, size, super::Device::Cuda(device_id)) })
}

/// Allocate page-locked (pinned) host memory for fast CPU↔GPU transfers.
///
/// Must be freed with `cuda_free_host`.
pub fn cuda_host_alloc(size: usize) -> CudaResult<*mut u8> {
    let mut ptr: *mut c_void = std::ptr::null_mut();
    let ret = unsafe { cudaHostAlloc(&mut ptr, size, CUDA_HOST_ALLOC_PORTABLE) };
    if ret != 0 || ptr.is_null() {
        return Err(super::LoaderError::FetchError(format!(
            "cudaHostAlloc({size}) failed: {ret}"
        )));
    }
    Ok(ptr as *mut u8)
}

/// Free page-locked host memory.
pub fn cuda_free_host(ptr: *mut u8) {
    if !ptr.is_null() {
        unsafe {
            cudaFreeHost(ptr as *mut c_void);
        }
    }
}

/// Create a non-blocking CUDA stream.
pub fn cuda_stream_create() -> CudaResult<*mut c_void> {
    let mut stream: *mut c_void = std::ptr::null_mut();
    let ret = unsafe { cudaStreamCreateWithFlags(&mut stream, CUDA_STREAM_NON_BLOCKING) };
    if ret != 0 {
        return Err(super::LoaderError::FetchError(format!(
            "cudaStreamCreate failed: {ret}"
        )));
    }
    Ok(stream)
}

/// Synchronize a CUDA stream (wait for all operations to complete).
///
/// # Safety
/// `stream` must be a valid CUDA stream handle returned by `cuda_stream_create`.
pub unsafe fn cuda_stream_sync(stream: *mut c_void) -> CudaResult<()> {
    let ret = unsafe { cudaStreamSynchronize(stream) };
    if ret != 0 {
        return Err(super::LoaderError::FetchError(format!(
            "cudaStreamSynchronize failed: {ret}"
        )));
    }
    Ok(())
}

/// Destroy a CUDA stream.
///
/// # Safety
/// `stream` must be a valid CUDA stream handle. Must not be used after this call.
pub unsafe fn cuda_stream_destroy(stream: *mut c_void) {
    unsafe {
        cudaStreamDestroy(stream);
    }
}

/// Synchronous memcpy from host (CPU) to device (GPU).
pub(crate) fn cuda_memcpy_htod(dst: *mut u8, src: *const u8, size: usize) -> CudaResult<()> {
    let ret = unsafe {
        cudaMemcpy(
            dst as *mut c_void,
            src as *const c_void,
            size,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    };
    if ret != 0 {
        return Err(super::LoaderError::FetchError(format!(
            "cudaMemcpy H2D failed: {ret}"
        )));
    }
    Ok(())
}

/// Async memcpy from host (CPU) to device (GPU) on the given stream.
///
/// # Safety
/// `dst` must be a valid CUDA device pointer, `src` a valid host pointer,
/// and `stream` a valid CUDA stream handle.
pub unsafe fn cuda_memcpy_htod_async(
    dst: *mut u8,
    src: *const u8,
    size: usize,
    stream: *mut c_void,
) -> CudaResult<()> {
    let ret = unsafe {
        cudaMemcpyAsync(
            dst as *mut c_void,
            src as *const c_void,
            size,
            CUDA_MEMCPY_HOST_TO_DEVICE,
            stream,
        )
    };
    if ret != 0 {
        return Err(super::LoaderError::FetchError(format!(
            "cudaMemcpyAsync H2D failed: {ret}"
        )));
    }
    Ok(())
}

/// Async device-to-device memcpy (GPU→GPU, including cross-device P2P).
///
/// # Safety
/// `dst` and `src` must be valid CUDA device pointers,
/// and `stream` a valid CUDA stream handle.
pub unsafe fn cuda_memcpy_dtod_async(
    dst: *mut u8,
    src: *const u8,
    size: usize,
    stream: *mut c_void,
) -> super::Result<()> {
    let ret = unsafe {
        cudaMemcpyAsync(
            dst as *mut c_void,
            src as *const c_void,
            size,
            CUDA_MEMCPY_DEVICE_TO_DEVICE,
            stream,
        )
    };
    if ret != 0 {
        return Err(super::LoaderError::FetchError(format!(
            "cudaMemcpyAsync D2D failed: {ret}"
        )));
    }
    Ok(())
}

pub(crate) fn cuda_event_create() -> CudaResult<*mut c_void> {
    let mut event: *mut c_void = std::ptr::null_mut();
    let ret = unsafe { cudaEventCreateWithFlags(&mut event, CUDA_EVENT_DISABLE_TIMING) };
    if ret != 0 {
        return Err(super::LoaderError::FetchError(format!(
            "cudaEventCreate failed: {ret}"
        )));
    }
    Ok(event)
}

pub(crate) fn cuda_event_record(event: *mut c_void, stream: *mut c_void) -> CudaResult<()> {
    let ret = unsafe { cudaEventRecord(event, stream) };
    if ret != 0 {
        return Err(super::LoaderError::FetchError(format!(
            "cudaEventRecord failed: {ret}"
        )));
    }
    Ok(())
}

/// Non-blocking check if event has completed. Returns true if done.
pub(crate) fn cuda_event_query(event: *mut c_void) -> bool {
    let ret = unsafe { cudaEventQuery(event) };
    ret == 0 // 0 = cudaSuccess = completed
}

pub(crate) fn cuda_event_synchronize(event: *mut c_void) -> CudaResult<()> {
    let ret = unsafe { cudaEventSynchronize(event) };
    if ret != 0 {
        return Err(super::LoaderError::FetchError(format!(
            "cudaEventSynchronize failed: {ret}"
        )));
    }
    Ok(())
}

pub(crate) fn cuda_event_destroy(event: *mut c_void) {
    unsafe {
        cudaEventDestroy(event);
    }
}

pub(crate) fn cuda_get_default_mem_pool(device_id: usize) -> CudaResult<*mut c_void> {
    let mut pool: *mut c_void = std::ptr::null_mut();
    let ret = unsafe { cudaDeviceGetDefaultMemPool(&mut pool, device_id as c_int) };
    if ret != 0 {
        return Err(super::LoaderError::FetchError(format!(
            "cudaDeviceGetDefaultMemPool({device_id}) failed: {ret}"
        )));
    }
    Ok(pool)
}

pub(crate) fn cuda_malloc_from_pool_async(
    size: usize,
    pool: *mut c_void,
    stream: *mut c_void,
) -> CudaResult<*mut u8> {
    let mut ptr: *mut c_void = std::ptr::null_mut();
    let ret = unsafe { cudaMallocFromPoolAsync(&mut ptr, size, pool, stream) };
    if ret != 0 || ptr.is_null() {
        return Err(super::LoaderError::FetchError(format!(
            "cudaMallocFromPoolAsync({size}) failed: {ret}"
        )));
    }
    Ok(ptr as *mut u8)
}

pub(crate) fn cuda_free_async(ptr: *mut u8, stream: *mut c_void) {
    if !ptr.is_null() {
        unsafe {
            cudaFreeAsync(ptr as *mut c_void, stream);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test 1: cudaMallocAsync with NULL stream + synchronous cudaMemcpy.
    /// This is what previously failed with error 719.
    #[test]
    fn test_malloc_async_null_stream_sync_memcpy() {
        set_cuda_device(0).unwrap();

        let size = 64 * 1024; // 64KB
        let ptr = match cuda_malloc_async(size, std::ptr::null_mut()) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("cudaMallocAsync(NULL stream) not supported: {e}");
                return;
            }
        };
        eprintln!("cudaMallocAsync(NULL stream) succeeded, ptr: {ptr:?}");

        // Try synchronous memcpy (this previously failed with error 719)
        let ret = cuda_memcpy_htod(ptr, [42u8; 64].as_ptr(), 64);
        eprintln!("sync cudaMemcpy H2D result: {ret:?}");
        match ret {
            Ok(()) => eprintln!("NULL stream + sync memcpy: WORKS"),
            Err(e) => eprintln!("CONFIRMED: NULL stream + sync memcpy FAILS: {e}"),
        }
        cuda_free(ptr);
    }

    /// Test 2: cudaMallocAsync with explicit stream + cudaMemcpyAsync.
    /// This is the approach we want to validate for fetch_batch.
    #[test]
    fn test_malloc_async_explicit_stream_async_memcpy() {
        set_cuda_device(0).unwrap();

        let stream = cuda_stream_create().unwrap();
        let size = 1024 * 1024; // 1MB

        let ptr = cuda_malloc_async(size, stream).unwrap();
        eprintln!("cudaMallocAsync(explicit stream) succeeded, ptr: {ptr:?}");

        let src = vec![0xABu8; size];
        unsafe {
            cuda_memcpy_htod_async(ptr, src.as_ptr(), size, stream).unwrap();
            cuda_stream_sync(stream).unwrap();
        }
        eprintln!("Explicit stream + async memcpy: OK");

        cuda_free(ptr);
        unsafe { cuda_stream_destroy(stream) };
    }

    /// Test 3: cudaMallocAsync(stream) + sync stream + then sync cudaMemcpy.
    /// Tests if the pointer works with sync APIs after stream is drained.
    #[test]
    fn test_malloc_async_stream_sync_then_sync_memcpy() {
        set_cuda_device(0).unwrap();

        let stream = cuda_stream_create().unwrap();
        let size = 1024 * 1024; // 1MB

        let ptr = cuda_malloc_async(size, stream).unwrap();

        // Sync stream to ensure allocation is complete
        unsafe { cuda_stream_sync(stream).unwrap() };

        // Now try synchronous memcpy
        let ret = cuda_memcpy_htod(ptr, vec![7u8; size].as_ptr(), size);
        eprintln!("stream alloc + sync + sync memcpy: {ret:?}");
        ret.unwrap();

        cuda_free(ptr);
        unsafe { cuda_stream_destroy(stream) };
    }

    /// Test 4: Measure alloc overhead — cudaMalloc vs cudaMallocAsync pool reuse.
    #[test]
    fn test_alloc_overhead_comparison() {
        set_cuda_device(0).unwrap();

        let size = 2 * 1024 * 1024; // 2MB
        let iterations = 100;
        let src = vec![0u8; size];

        // Warm up
        let _ = cuda_malloc(size).map(|p| {
            cuda_free(p);
        });

        // Time cudaMalloc + cudaMemcpy (sync) + cudaFree
        let t0 = std::time::Instant::now();
        for _ in 0..iterations {
            let ptr = cuda_malloc(size).unwrap();
            cuda_memcpy_htod(ptr, src.as_ptr(), size).unwrap();
            cuda_free(ptr);
        }
        let malloc_time = t0.elapsed();

        // Time cudaMallocAsync + cudaMemcpyAsync + stream sync + cudaFree
        let stream = cuda_stream_create().unwrap();
        // Warm up async pool
        let p = cuda_malloc_async(size, stream).unwrap();
        unsafe { cuda_stream_sync(stream).unwrap() };
        cuda_free(p);

        let t0 = std::time::Instant::now();
        for _ in 0..iterations {
            let ptr = cuda_malloc_async(size, stream).unwrap();
            unsafe {
                cuda_memcpy_htod_async(ptr, src.as_ptr(), size, stream).unwrap();
                cuda_stream_sync(stream).unwrap();
            }
            cuda_free(ptr);
        }
        let async_time = t0.elapsed();

        unsafe { cuda_stream_destroy(stream) };

        let speedup = malloc_time.as_secs_f64() / async_time.as_secs_f64();
        eprintln!("{iterations}x 2MB alloc+H2D+free:");
        eprintln!(
            "  cudaMalloc (sync):  {:?} ({:.0} us/iter)",
            malloc_time,
            malloc_time.as_micros() as f64 / iterations as f64
        );
        eprintln!(
            "  cudaMallocAsync:    {:?} ({:.0} us/iter)",
            async_time,
            async_time.as_micros() as f64 / iterations as f64
        );
        eprintln!("  speedup: {speedup:.2}x");
    }
}
