use core::ffi::{c_char, c_int, CStr};
use core::str::Utf8Error;
use safetensors::tensor::{SafeTensorError, SafeTensors, TensorView};
use thiserror::Error;

#[repr(C)]
pub struct Handle {
    safetensors: SafeTensors<'static>,
}

#[repr(C)]
pub struct View {
    view: TensorView<'static>,
}

#[no_mangle]
pub unsafe extern "C" fn deserialize(
    buffer: *const u8,
    buffer_len: usize,
    handle: *mut Handle,
) -> c_int {
    match _deserialize(buffer, buffer_len, handle) {
        Ok(_) => 0,
        Err(_) => 1,
    }
}

#[no_mangle]
pub unsafe extern "C" fn get_tensor(
    handle: *mut Handle,
    name: *const c_char,
    view: *mut View,
) -> c_int {
    match _get_tensor(handle, name, view) {
        Ok(_) => 0,
        Err(_) => 1,
    }
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

unsafe fn _deserialize(buffer: *const u8, buffer_len: usize, handle: *mut Handle) -> Result<()> {
    if handle.is_null() {
        return Err(CError::NullPointer(
            "Null pointer `handle` when accessing deserialize".to_string(),
        ));
    }
    if buffer.is_null() {
        return Err(CError::NullPointer(
            "Null pointer `buffer` when accessing deserialize".to_string(),
        ));
    }
    let buffer = unsafe { std::slice::from_raw_parts(buffer, buffer_len) };
    (*handle).safetensors = safetensors;
    Ok(())
}

unsafe fn _get_tensor(
    handle: *mut Handle,
    name: *const c_char,
    view: *mut View,
) -> Result<(), CError> {
    if name.is_null() {
        return Err(CError::NullPointer(
            "Null pointer `name` when accessing get_tensor".to_string(),
        ));
    }
    if handle.is_null() {
        return Err(CError::NullPointer(
            "Null pointer `handle` when accessing get_tensor".to_string(),
        ));
    }
    let name = CStr::from_ptr(name).to_str()?;

    (*view).view = (*handle).safetensors.tensor(name)?;
    Ok(())
}
