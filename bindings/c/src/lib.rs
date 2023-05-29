use core::ffi::{c_int, c_uint};
use core::str::Utf8Error;
use std::ffi::{c_char, CString};
use safetensors::tensor::{SafeTensorError, SafeTensors, TensorView};
use thiserror::Error;

const VERSION: &str = "0.3.0";


#[repr(C)]
pub struct Handle {
    safetensors: SafeTensors<'static>,
}

#[repr(C)]
pub struct View {
    view: TensorView<'static>,
}

#[no_mangle]
pub extern "C" fn safetensors_version() -> c_uint {
    42
}

#[no_mangle]
pub extern "C" fn safetensors_deserialize(
    handle: *mut Handle,
    buffer: *const u8,
    buffer_len: usize,
) -> c_int {
    match unsafe { _deserialize(handle, buffer, buffer_len) } {
        Ok(_) => 0,
        Err(_) => 1,
    }
}

#[no_mangle]
pub unsafe extern "C" fn safetensors_num_tensors(handle: *const Handle) -> c_uint {
    (*handle).safetensors.len() as c_uint
}

#[no_mangle]
pub unsafe extern "C" fn safetensors_destroy(handle: *mut Handle) {
    let tensors = handle.read().safetensors;
    drop(tensors);
}

#[derive(Debug, Error)]
enum CError {
    #[error("{0}")]
    NullPointer(String),

    #[error("{0}")]
    Utf8Error(#[from] Utf8Error),

    #[error("{0}")]
    SafeTensorError(#[from] SafeTensorError),
}

type SafeTensorsResult<T> = Result<T, CError>;

// #[no_mangle]
// pub unsafe extern "C" fn get_tensor(
//     handle: *const Handle,
//     name: *const c_char,
//     view: *mut View,
// ) -> c_int {
//     match _get_tensor(handle, name, view) {
//         Ok(_) => 0,
//         Err(_) => 1,
//     }
// }

// unsafe fn _deserialize(buffer: *const u8, buffer_len: usize, handle: *mut Handle) -> Result<()> {
#[inline]
unsafe fn _deserialize(handle: *mut Handle, buffer: *const u8, buffer_len: usize) -> SafeTensorsResult<()> {
    if buffer.is_null() {
        return Err(CError::NullPointer(
            "Null pointer `buffer` when accessing deserialize".to_string(),
        ));
    }

    let data = unsafe { std::slice::from_raw_parts(buffer, buffer_len) };
    let tensors = SafeTensors::deserialize(&data)?;

    handle.write(Handle { safetensors: tensors});
    Ok(())
}

// unsafe fn _get_tensor(
//     handle: *mut Handle,
//     name: *const c_char,
//     view: *mut View,
// ) -> Result<(), CError> {
//     if name.is_null() {
//         return Err(CError::NullPointer(
//             "Null pointer `name` when accessing get_tensor".to_string(),
//         ));
//     }
//     if handle.is_null() {
//         return Err(CError::NullPointer(
//             "Null pointer `handle` when accessing get_tensor".to_string(),
//         ));
//     }
//     let name = CStr::from_ptr(name).to_str()?;

//     (*view).view = (*handle).safetensors.tensor(name)?;
//     Ok(())
// }
