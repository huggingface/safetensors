use core::ffi::c_uint;
use core::str::Utf8Error;
use safetensors::tensor::{SafeTensorError, SafeTensors};
use safetensors::Dtype as RDtype;
use std::ffi::{c_char, CStr, CString};
use std::mem::forget;
use thiserror::Error;

#[repr(C)]
pub enum Status {
    NullPointer = -2,
    Utf8Error,
    Ok,
    InvalidHeader,
    InvalidHeaderDeserialization,
    HeaderTooLarge,
    HeaderTooSmall,
    InvalidHeaderLength,
    TensorNotFound,
    TensorInvalidInfo,
    InvalidOffset,
    IoError,
    JsonError,
    InvalidTensorView,
    MetadataIncompleteBuffer,
    ValidationOverflow,
}

#[derive(Debug, Error)]
#[repr(C)]
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
            CError::NullPointer(_) => Status::NullPointer,
            CError::Utf8Error(_) => Status::Utf8Error,
            CError::SafeTensorError(err) => match err {
                SafeTensorError::InvalidHeader => Status::InvalidHeader,
                SafeTensorError::InvalidHeaderDeserialization => {
                    Status::InvalidHeaderDeserialization
                }
                SafeTensorError::HeaderTooLarge => Status::HeaderTooLarge,
                SafeTensorError::HeaderTooSmall => Status::HeaderTooSmall,
                SafeTensorError::InvalidHeaderLength => Status::InvalidHeaderLength,
                SafeTensorError::TensorNotFound(_) => Status::TensorNotFound,
                SafeTensorError::TensorInvalidInfo => Status::TensorInvalidInfo,
                SafeTensorError::InvalidOffset(_) => Status::InvalidOffset,
                SafeTensorError::IoError(_) => Status::IoError,
                SafeTensorError::JsonError(_) => Status::JsonError,
                SafeTensorError::InvalidTensorView(_, _, _) => Status::InvalidTensorView,
                SafeTensorError::MetadataIncompleteBuffer => Status::MetadataIncompleteBuffer,
                SafeTensorError::ValidationOverflow => Status::ValidationOverflow,
            },
        }
    }
}

/// The various available dtypes. They MUST be in increasing alignment order
#[repr(C)]
pub enum Dtype {
    /// Boolan type
    BOOL,
    /// Unsigned byte
    U8,
    /// Signed byte
    I8,
    /// Signed integer (16-bit)
    I16,
    /// Unsigned integer (16-bit)
    U16,
    /// Half-precision floating point
    F16,
    /// Brain floating point
    BF16,
    /// Signed integer (32-bit)
    I32,
    /// Unsigned integer (32-bit)
    U32,
    /// Floating point (32-bit)
    F32,
    /// Floating point (64-bit)
    F64,
    /// Signed integer (64-bit)
    I64,
    /// Unsigned integer (64-bit)
    U64,
}

impl Dtype {
    fn to_dtype(self) -> RDtype {
        match self {
            Dtype::BOOL => RDtype::BOOL,
            Dtype::U8 => RDtype::U8,
            Dtype::I8 => RDtype::I8,
            Dtype::I16 => RDtype::I16,
            Dtype::U16 => RDtype::U16,
            Dtype::I32 => RDtype::I32,
            Dtype::U32 => RDtype::U32,
            Dtype::I64 => RDtype::I64,
            Dtype::U64 => RDtype::U64,
            Dtype::F16 => RDtype::F16,
            Dtype::BF16 => RDtype::BF16,
            Dtype::F32 => RDtype::F32,
            Dtype::F64 => RDtype::F64,
        }
    }
}

impl From<RDtype> for Dtype {
    fn from(dtype: RDtype) -> Dtype {
        match dtype {
            RDtype::BOOL => Dtype::BOOL,
            RDtype::U8 => Dtype::U8,
            RDtype::I8 => Dtype::I8,
            RDtype::I16 => Dtype::I16,
            RDtype::U16 => Dtype::U16,
            RDtype::I32 => Dtype::I32,
            RDtype::U32 => Dtype::U32,
            RDtype::I64 => Dtype::I64,
            RDtype::U64 => Dtype::U64,
            RDtype::F16 => Dtype::F16,
            RDtype::BF16 => Dtype::BF16,
            RDtype::F32 => Dtype::F32,
            RDtype::F64 => Dtype::F64,
            d => panic!("Unhandled dtype {d:?}"),
        }
    }
}

pub struct Handle {
    safetensors: SafeTensors<'static>,
}

#[repr(C)]
pub struct View {
    dtype: Dtype,
    rank: usize,
    shape: *const usize,
    data: *const u8,
}

/// Attempt to deserialize the content of `buffer`, reading `buffer_len` bytes as a safentesors
/// data buffer.
///
/// # Arguments
///
/// * `handle`: In-Out pointer to store the resulting safetensors reference is sucessfully deserialized
/// * `buffer`: Buffer to attempt to read data from
/// * `buffer_len`: Number of bytes we can safely read from the deserialize the safetensors
///
/// returns: `Status::Ok == 0` if success, any other status code if an error what caught up
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

            Status::Ok
        },
        Err(err) => err.into(),
    }
}

/// Free the resources hold by the safetensors
///
/// # Arguments
///
/// * `handle`: Pointer ot the safetensors we want to release the resources of
///
/// returns: `Status::Ok == 0` if success, any other status code if an error what caught up
#[no_mangle]
pub unsafe extern "C" fn safetensors_destroy(handle: *mut Handle) -> Status {
    if !handle.is_null() {
        // Restore the heap allocated handle and explicitly drop it
        drop(Box::from_raw(handle));
    }

    Status::Ok
}

/// Retrieve the list of tensor's names currently stored in the safetensors
///
/// # Arguments
///
/// * `handle`: Pointer to the underlying safetensors we want to query tensor's names from
/// * `ptr`: In-Out pointer to store the array of strings representing all the tensor's names
/// * `len`: Number of strings stored in `ptr`
///
/// returns: `Status::Ok == 0` if success, any other status code if an error what caught up
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

        Status::Ok
    }
}

/// Free the resources used to represent the list of tensor's names stored in the safetensors.
/// This must follow any call to `safetensors_names()` to clean up underlying resources.
///
/// # Arguments
///
/// * `names`: Pointer to the array of strings we want to release resources of
/// * `len`: Number of strings hold by `names` array
///
/// returns: `Status::Ok == 0` if success, any other status code if an error what caught up
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

    Status::Ok
}

/// Return the number of tensors stored in this safetensors
///
/// # Arguments
///
/// * `handle`: Pointer to the underlying safetensors we want to know the number of tensors of.
///
/// returns: usize Number of tensors in the safetensors
#[no_mangle]
pub unsafe extern "C" fn safetensors_num_tensors(handle: *const Handle) -> usize {
    (*handle).safetensors.len()
}

/// Return the number of bytes required to represent a single element from the specified dtype
///
/// # Arguments
///
/// * `dtype`: The data type we want to know the number of bytes required
///
/// returns: usize Number of bytes for this specific `dtype`
#[no_mangle]
pub extern "C" fn safetensors_dtype_size(dtype: Dtype) -> usize {
    dtype.to_dtype().size()
}

/// Attempt to retrieve the metadata and content for the tensor associated with `name` storing the
/// result to the memory location pointed by `view` pointer.
///
/// # Arguments
///
/// * `handle`: Pointer to the underlying safetensors we want to retrieve the tensor from.
/// * `view`: In-Out pointer to store the tensor if successfully found to belong to the safetensors
/// * `name`: The name of the tensor to retrieve from the safetensors
///
/// returns: `Status::Ok == 0` if success, any other status code if an error what caught up
#[no_mangle]
pub extern "C" fn safetensors_get_tensor(
    handle: *const Handle,
    view: *mut *mut View,
    name: *const c_char,
) -> Status {
    match unsafe { _get_tensor(handle, view, name) } {
        Ok(_) => Status::Ok,
        Err(err) => err.into(),
    }
}

/// Free the resources used by a TensorView to expose metadata + content to the C-FFI layer
///
/// # Arguments
///
/// * `ptr`: Pointer to the TensorView we want to release the underlying resources of
///
/// returns: `Status::Ok = 0` if resources were successfully freed
#[no_mangle]
pub extern "C" fn safetensors_free_tensor(ptr: *mut View) -> Status {
    unsafe {
        // Restore the heap allocated view and explicitly drop it
        drop(Box::from_raw(ptr));

        Status::Ok
    }
}

/// Deserialize the content pointed by `buffer`, reading `buffer_len` number of bytes from it
///
/// # Arguments
///
/// * `buffer`: The raw buffer to read from
/// * `buffer_len`: The number of bytes to safely read from the `buffer`
///
/// returns: Result<SafeTensors, CError>
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

/// Retrieve a tensor from the underlying safetensors pointed by `handle` and referenced by it's `name`.
/// If found, the resulting view will populate the memory location pointed by `ptr`
///
/// # Arguments
///
/// * `handle`: Handle to the underlying safetensors we want to retrieve the tensor from
/// * `ptr`: The in-out pointer to populate if the tensor is found
/// * `name`: The name of the tensor we want to retrieve
///
/// returns: Result<(), CError>
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

    unsafe {
        let shape = st_view.shape().to_vec();
        let view = Box::new(View {
            dtype: st_view.dtype().into(),
            rank: shape.len(),
            shape: shape.as_ptr(),
            data: st_view.data().as_ptr(),
        });
        forget(shape);

        ptr.write(Box::into_raw(view));
    }

    Ok(())
}
