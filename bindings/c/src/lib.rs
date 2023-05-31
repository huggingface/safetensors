use core::ffi::{c_int, c_uint};
use core::str::Utf8Error;
use safetensors::tensor::{SafeTensorError, SafeTensors, TensorView};
use std::ffi::{c_char, c_ulong, c_ulonglong, CString};
use std::mem::forget;
use thiserror::Error;

type Status = c_int;
const STATUS_OK: Status = 0;

#[derive(Debug, Error)]
enum CError {
    #[error("{0}")]
    NullPointer(String),

    #[error("{0}")]
    Utf8Error(#[from] Utf8Error),

    #[error("{0}")]
    SafeTensorError(#[from] SafeTensorError),
}

impl Into<Status> for CError {
    fn into(self) -> Status {
        match self {
            CError::NullPointer(_) => -1,
            CError::Utf8Error(_) => -2,
            CError::SafeTensorError(err) => {
                println!("{}", err);
                -10
            }
        }
    }
}

#[repr(C)]
pub struct Handle {
    safetensors: SafeTensors<'static>,
}

#[repr(C)]
pub struct View {
    view: TensorView<'static>,
}

#[no_mangle]
pub extern "C" fn safetensors_deserialize(
    handle: *mut *mut Handle,
    buffer: *const u8,
    buffer_len: usize,
) -> Status {
    match unsafe { _deserialize(buffer, buffer_len) } {
        Ok(safetensors) => unsafe {
            let heap_handle = Box::new(Handle { safetensors });
            let raw = Box::into_raw(heap_handle);
            handle.write(raw);

            STATUS_OK
        },
        Err(err) => err.into(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn safetensors_names(
    handle: *const Handle,
    ptr: *mut *const *const c_char,
    len: *mut c_uint,
) -> Status {
    let names = (*handle).safetensors.names();
    let c_names = names
        .into_iter()
        .map(|name| CString::from_vec_unchecked(name.clone().into_bytes()).as_ptr())
        .collect::<Vec<_>>();

    // let c_ptrs = c_names.iter().map(|name| name.as_ptr()).collect::<Vec<_>>();

    unsafe {
        ptr.write(c_names.as_ptr());
        len.write(c_names.len() as c_uint);

        forget(c_names);

        STATUS_OK
    }
}

#[no_mangle]
pub unsafe extern "C" fn safetensors_free_names(
    names: *const *const c_char,
    len: c_uint,
) -> Status {
    let len = len as usize;

    // Get back our vector.
    // Previously we shrank to fit, so capacity == length.
    let v = Vec::from_raw_parts(names.cast_mut(), len, len);

    // Now drop one string at a time.
    for elem in v {
        let s = CString::from_raw(elem.cast_mut());
        drop(s);
    }

    STATUS_OK
}

#[no_mangle]
pub unsafe extern "C" fn safetensors_num_tensors(handle: *const Handle) -> usize {
    (*handle).safetensors.len()
}

#[no_mangle]
pub unsafe extern "C" fn safetensors_destroy(handle: *mut Handle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

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

#[inline(always)]
unsafe fn _deserialize(
    buffer: *const u8,
    buffer_len: usize,
) -> Result<SafeTensors<'static>, CError> {
    if buffer.is_null() {
        return Err(CError::NullPointer(
            "Null pointer `buffer` when accessing deserialize".to_string(),
        ));
    }

    let data = unsafe { std::slice::from_raw_parts(buffer, buffer_len) };
    SafeTensors::deserialize(&data).map_err(|err| CError::SafeTensorError(err))
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
