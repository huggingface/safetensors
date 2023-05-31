use core::ffi::{c_int, c_uint};
use core::str::Utf8Error;
use safetensors::tensor::{SafeTensorError, SafeTensors};
use safetensors::Dtype;
use std::ffi::{c_char, CStr, CString};
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
    dtype: u32,
    rank: usize,
    shapes: *const usize,
    data: *const u8,
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

    // We need to convert the String repr to a C-friendly repr (NUL terminated)
    let c_names = names
        .into_iter()
        .map(|name| {
            // Nul-terminated string
            let s = CString::from_vec_unchecked(name.clone().into_bytes());
            let ptr = s.as_ptr();

            // Advise Rust we will take care of the desallocation (see `safetensors_free_names`)
            forget(s);

            ptr
        })
        .collect::<Vec<_>>();

    unsafe {
        ptr.write(c_names.as_ptr());
        len.write(c_names.len() as c_uint);

        forget(c_names);

        STATUS_OK
    }
}

#[no_mangle]
pub extern "C" fn safetensors_free_names(names: *const *const c_char, len: c_uint) -> Status {
    let len = len as usize;

    unsafe {
        // Get back our vector.
        let v = Vec::from_raw_parts(names.cast_mut(), len, len);

        // Now drop all the string.
        for elem in v {
            let _ = CString::from_raw(elem.cast_mut());
        }
    }

    STATUS_OK
}

#[no_mangle]
pub unsafe extern "C" fn safetensors_num_tensors(handle: *const Handle) -> usize {
    (*handle).safetensors.len()
}

#[no_mangle]
#[allow(improper_ctypes_definitions)]
pub extern "C" fn safetensors_dtype_size(dtype: Dtype) -> usize {
    dtype.size()
}

#[no_mangle]
pub unsafe extern "C" fn safetensors_destroy(handle: *mut Handle) {
    if !handle.is_null() {
        // Restore the heap allocated handle and explicitly drop it
        drop(Box::from_raw(handle));
    }
}

#[no_mangle]
pub extern "C" fn safetensors_get_tensor(
    handle: *const Handle,
    view: *mut *mut View,
    name: *const c_char,
) -> Status {
    match unsafe { _get_tensor(handle, view, name) } {
        Ok(_) => STATUS_OK,
        Err(err) => err.into(),
    }
}

#[no_mangle]
pub extern "C" fn safetensors_free_tensor(ptr: *mut View) -> Status {
    unsafe {
        // Restore the heap allocated view and explicitly drop it
        drop(Box::from_raw(ptr));

        STATUS_OK
    }
}

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

unsafe fn _get_tensor(
    handle: *const Handle,
    ptr: *mut *mut View,
    name: *const c_char,
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

    let st_view = (*handle).safetensors.tensor(name)?;
    let view = Box::new(View {
        dtype: st_view.dtype() as u32,
        rank: st_view.shape().len(),
        shapes: st_view.shape().as_ptr(),
        data: st_view.data().as_ptr(),
    });

    ptr.write(Box::into_raw(view));

    Ok(())
}
