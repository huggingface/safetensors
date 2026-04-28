//! DLPack v0 producer + Python capsule wrapping.

use std::ffi::{c_char, c_void};

use pyo3::prelude::*;

use safetensors::Dtype;

#[cfg(target_os = "linux")]
use ionic_rs::cuda::DeviceBuf;

/// Subset of `DLDeviceType` we actually emit
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DLDeviceType {
    /// `kDLCPU` — host memory.
    Cpu = 1,
    /// `kDLCUDA` — device memory (regular CUDA, not host or unified).
    Cuda = 2,
}

/// Type code of a `DLDataType`. Only the variants we currently emit.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DLDataTypeCode {
    Int = 0,
    UInt = 1,
    Float = 2,
    /// Opaque handle used for dtypes the framework doesn't model
    /// natively (e.g. our F4, F6, F8 variants beyond what DLPack
    /// standardizes); consumers typically reinterpret the bytes.
    OpaqueHandle = 3,
    Bfloat = 4,
    Complex = 5,
    Bool = 6,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DLDataType {
    pub code: DLDataTypeCode,
    pub bits: u8,
    pub lanes: u16,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DLDevice {
    pub device_type: DLDeviceType,
    pub device_id: i32,
}

#[repr(C)]
#[derive(Debug)]
pub struct DLTensor {
    pub data: *mut c_void,
    pub device: DLDevice,
    pub ndim: i32,
    pub dtype: DLDataType,
    pub shape: *mut i64,
    /// `null` signals "row-major C-contiguous" per the DLPack spec; we
    /// rely on this to avoid heap-allocating a strides array for every
    /// tensor we hand off.
    pub strides: *mut i64,
    pub byte_offset: u64,
}

/// DLPack v0 managed tensor. `dl_tensor` carries the data; `manager_ctx`
/// is opaque to consumers and points back to producer-side state; the
/// consumer calls `deleter(self)` exactly once when done.
#[repr(C)]
pub struct DLManagedTensor {
    pub dl_tensor: DLTensor,
    pub manager_ctx: *mut c_void,
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

/// Capsule name (v0 spec). Consumers rename to `"used_dltensor"` after
/// taking ownership.
const CAPSULE_NAME: &[u8] = b"dltensor\0";

/// Buffers that can be handed off through DLPack. Callers must guarantee
/// the data pointer stays valid until the implementer's `Drop` fires
/// (which is when the consumer invokes the deleter). `Send` is required
/// because the deleter may run on any thread; `'static` because the
/// buffer is stored opaquely in `manager_ctx`.
pub(crate) trait AsDevicePtr: Send + 'static {
    fn as_device_ptr(&self) -> *mut c_void;
}

#[cfg(target_os = "linux")]
impl AsDevicePtr for DeviceBuf {
    fn as_device_ptr(&self) -> *mut c_void {
        // CUdeviceptr is u64 on every supported platform; casting to a
        // pointer here is the standard DLPack `data` representation.
        DeviceBuf::as_device_ptr(self) as *mut c_void
    }
}

/// Owns the producer-side state behind a [`DLManagedTensor`]. The
/// deleter takes ownership via `Box::from_raw` and drops it, which
/// frees the device allocation (via `B::Drop`) and the heap-allocated
/// shape array.
struct ManagedCtx<B: AsDevicePtr> {
    /// Owns the device allocation. `Drop` of the concrete `B` releases
    /// the underlying memory (e.g. `cuMemFree_v2` for `DeviceBuf`).
    _device_buf: B,
    /// Boxed slice rather than `Vec` so the pointer + length stay stable
    /// without us tracking capacity. `DLTensor.shape` points into this.
    shape: Box<[i64]>,
}

/// Standard deleter, monomorphized per buffer type so the cast back to
/// `ManagedCtx<B>` is sound. The function pointer is what gets stored
/// in `DLManagedTensor.deleter`; the consumer doesn't see `B`.
///
/// # Safety
///
/// `self_ptr` must have been produced by [`to_capsule`] with the same
/// `B` and must not have been deleted before. The DLPack contract
/// guarantees exactly-once delete by the consumer.
unsafe extern "C" fn managed_tensor_deleter<B: AsDevicePtr>(self_ptr: *mut DLManagedTensor) {
    if self_ptr.is_null() {
        return;
    }
    // Reclaim ManagedCtx first so its Drop runs before the outer Box.
    let ctx_ptr = unsafe { (*self_ptr).manager_ctx as *mut ManagedCtx<B> };
    if !ctx_ptr.is_null() {
        unsafe { drop(Box::from_raw(ctx_ptr)) };
    }
    unsafe { drop(Box::from_raw(self_ptr)) };
}

pub(crate) fn dtype_to_dlpack(dtype: Dtype) -> DLDataType {
    let bits = dtype.bitsize() as u8;
    let code = match dtype {
        Dtype::BOOL => DLDataTypeCode::Bool,
        Dtype::I8 | Dtype::I16 | Dtype::I32 | Dtype::I64 => DLDataTypeCode::Int,
        Dtype::U8 | Dtype::U16 | Dtype::U32 | Dtype::U64 => DLDataTypeCode::UInt,
        Dtype::F16 | Dtype::F32 | Dtype::F64 => DLDataTypeCode::Float,
        Dtype::BF16 => DLDataTypeCode::Bfloat,
        // Sub-byte and FP8 variants: the DLPack v1.0 wire format reserves
        // codes for some of these (kDLFloat8_e*, kDLFloat4_*) but framework
        // support is patchy. Opaque keeps us correct on the byte-layout
        // axis; consumers that know the framework-specific encoding can
        // reinterpret. Not great for ergonomics but doesn't lose data.
        Dtype::C64 => DLDataTypeCode::Complex,
        _ => DLDataTypeCode::OpaqueHandle,
    };

    DLDataType {
        code,
        bits,
        lanes: 1,
    }
}

pub(crate) fn cuda_device(ordinal: i32) -> DLDevice {
    DLDevice {
        device_type: DLDeviceType::Cuda,
        device_id: ordinal,
    }
}

/// Wrap a device buffer in a Python `PyCapsule` carrying a
/// `DLManagedTensor`. Returns the capsule ready to hand to
/// `torch.from_dlpack` (or any v0-aware consumer).
///
/// Ownership of `device_buf` transfers into the capsule's
/// `manager_ctx`. Once `from_dlpack` consumes the capsule, the consumer
/// owns the lifecycle; if the capsule is GC'd before consumption, our
/// capsule destructor invokes the deleter so nothing leaks.
///
/// On `PyCapsule_New` failure we invoke the deleter ourselves before
/// returning the error, so the device allocation isn't leaked.
pub(crate) fn to_capsule<B: AsDevicePtr>(
    py: Python<'_>,
    device_buf: B,
    shape: Vec<i64>,
    dtype: DLDataType,
    device: DLDevice,
) -> PyResult<Py<PyAny>> {
    let ndim = shape.len() as i32;
    let data = device_buf.as_device_ptr();

    let ctx = Box::new(ManagedCtx {
        _device_buf: device_buf,
        shape: shape.into_boxed_slice(),
    });

    let shape_ptr = ctx.shape.as_ptr() as *mut i64;
    let ctx_ptr = Box::into_raw(ctx);

    let managed = Box::into_raw(Box::new(DLManagedTensor {
        dl_tensor: DLTensor {
            data,
            device,
            ndim,
            dtype,
            shape: shape_ptr,
            // null = compact row-major; consumers infer strides from shape.
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        },
        manager_ctx: ctx_ptr as *mut c_void,
        deleter: Some(managed_tensor_deleter::<B>),
    }));

    let name_ptr = CAPSULE_NAME.as_ptr() as *const c_char;
    // SAFETY: for all the following unsafe calls, all data was produced by this function and is valid each the call.
    // In case of failure, we clean up the managed tensor ourselves to avoid leaks, since the capsule destructor won't run.
    let capsule_ptr = unsafe {
        pyo3::ffi::PyCapsule_New(managed as *mut c_void, name_ptr, Some(capsule_destructor))
    };
    if capsule_ptr.is_null() {
        unsafe { managed_tensor_deleter::<B>(managed) };
        return Err(PyErr::fetch(py));
    }
    Ok(unsafe { Py::from_owned_ptr(py, capsule_ptr) })
}

unsafe extern "C" fn capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    let name_ptr = CAPSULE_NAME.as_ptr() as *const c_char;
    if unsafe { pyo3::ffi::PyCapsule_IsValid(capsule, name_ptr) } == 0 {
        return;
    }
    let ptr = unsafe { pyo3::ffi::PyCapsule_GetPointer(capsule, name_ptr) };
    if ptr.is_null() {
        return;
    }
    let managed = ptr as *mut DLManagedTensor;
    // SAFETY: we assume `managed` was put into the capsule by `to_capsule` and the
    // capsule was never consumed (IsValid passed).
    unsafe {
        if let Some(deleter) = (*managed).deleter {
            deleter(managed);
        }
    }
}
