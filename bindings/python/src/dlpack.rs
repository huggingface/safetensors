//! DLPack protocol implementation to enable 0-copy tensor transfer

use std::ffi::{c_void, CStr};

use pyo3::{ffi::PyCapsule_New, prelude::*};

use safetensors::Dtype;

#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DLDeviceType {
    Cpu = 1,
    Cuda = 2,
    // XXX: there are more but we do not need them
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DLDataTypeCode {
    Int = 0,
    UInt = 1,
    Float = 2,
    Bfloat = 4,
    Complex = 5,
    Bool = 6,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DLDataType {
    pub code: DLDataTypeCode,
    pub bits: u8,
    pub lanes: u16,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DLDevice {
    pub device_type: DLDeviceType,
    pub device_id: i32,
}

#[repr(C)]
pub struct DLTensor {
    pub data: *mut c_void,
    pub device: DLDevice,
    pub ndim: i32,
    pub dtype: DLDataType,
    pub shape: *mut i64,
    pub strides: *mut i64,
    pub byte_offset: u64,
}

#[repr(C)]
pub struct DLManagedTensor {
    pub dl_tensor: DLTensor,
    pub manager_ctx: *mut c_void,
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

struct CapsuleContext {
    buffer: hmll::Buffer,
    shape: Vec<i64>,
    strides: Vec<i64>,
}

pub fn buffer_to_capsule(
    py: Python<'_>,
    buffer: hmll::Buffer,
    shape: Vec<usize>,
    dtype: Dtype,
    hmll_device: hmll::Device,
) -> PyResult<PyObject> {
    let shape: Vec<i64> = shape.into_iter().map(|s| s as i64).collect();

    let mut strides: Vec<i64> = vec![1; shape.len()];
    for i in (0..strides.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    let ctx = Box::new(CapsuleContext {
        buffer,
        shape,
        strides,
    });
    let ctx_ptr = Box::into_raw(ctx);

    let dl_type = to_dl_dtype(dtype);
    let device = match hmll_device {
        hmll::Device::Cpu => DLDevice {
            device_type: DLDeviceType::Cpu,
            device_id: 0,
        },
        hmll::Device::Cuda(idx) => DLDevice {
            device_type: DLDeviceType::Cuda,
            device_id: idx as i32,
        },
    };
    let managed = Box::new(DLManagedTensor {
        dl_tensor: DLTensor {
            data: unsafe { (*ctx_ptr).buffer.as_ptr() as *mut c_void },
            device,
            ndim: unsafe { (*ctx_ptr).shape.len() as i32 },
            dtype: dl_type,
            shape: unsafe { (*ctx_ptr).shape.as_mut_ptr() },
            strides: unsafe { (*ctx_ptr).strides.as_mut_ptr() },
            byte_offset: 0,
        },
        manager_ctx: ctx_ptr as *mut c_void,
        deleter: Some(capsule_deleter),
    });

    let capsule_name = unsafe { CStr::from_bytes_with_nul_unchecked(b"dltensor\0") };
    let capsule_ptr = unsafe {
        PyCapsule_New(
            Box::into_raw(managed) as *mut c_void,
            capsule_name.as_ptr(),
            None,
        )
    };

    if capsule_ptr.is_null() {
        unsafe {
            drop(Box::from_raw(ctx_ptr));
        }
        return Err(PyErr::fetch(py));
    }

    Ok(unsafe { PyObject::from_owned_ptr(py, capsule_ptr) })
}

fn to_dl_dtype(dtype: Dtype) -> DLDataType {
    let bits = dtype.bitsize() as u8;

    let code = match dtype {
        Dtype::I8 | Dtype::I16 | Dtype::I32 | Dtype::I64 => DLDataTypeCode::Int,
        Dtype::U8 | Dtype::U16 | Dtype::U32 | Dtype::U64 => DLDataTypeCode::UInt,
        Dtype::BOOL => DLDataTypeCode::Bool,
        Dtype::F16 | Dtype::F32 | Dtype::F64 => DLDataTypeCode::Float,
        Dtype::BF16 => DLDataTypeCode::Bfloat,
        Dtype::C64 => DLDataTypeCode::Complex,
        _ => todo!("support exotic dtypes via u8"),
    };

    DLDataType {
        code,
        bits,
        lanes: 1,
    }
}

unsafe extern "C" fn capsule_deleter(managed: *mut DLManagedTensor) {
    if !managed.is_null() {
        let ctx_ptr = (*managed).manager_ctx as *mut CapsuleContext;
        if !ctx_ptr.is_null() {
            drop(Box::from_raw(ctx_ptr));
        }
        drop(Box::from_raw(managed))
    }
}
