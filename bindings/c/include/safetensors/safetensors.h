#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>

/// The various available dtypes. They MUST be in increasing alignment order
enum class Dtype {
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
};

enum class Status {
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
};

struct Handle;

struct View {
  Dtype dtype;
  uintptr_t rank;
  const uintptr_t *shape;
  const uint8_t *data;
};

extern "C" {

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
Status safetensors_deserialize(Handle **handle,
                               const uint8_t *buffer,
                               uintptr_t buffer_len);

/// Free the resources hold by the safetensors
///
/// # Arguments
///
/// * `handle`: Pointer ot the safetensors we want to release the resources of
///
/// returns: `Status::Ok == 0` if success, any other status code if an error what caught up
Status safetensors_destroy(Handle *handle);

/// Retrieve the list of tensor's names currently stored in the safetensors
///
/// # Arguments
///
/// * `handle`: Pointer to the underlying safetensors we want to query tensor's names from
/// * `ptr`: In-Out pointer to store the array of strings representing all the tensor's names
/// * `len`: Number of strings stored in `ptr`
///
/// returns: `Status::Ok == 0` if success, any other status code if an error what caught up
Status safetensors_names(const Handle *handle, const char *const **ptr, unsigned int *len);

/// Free the resources used to represent the list of tensor's names stored in the safetensors.
/// This must follow any call to `safetensors_names()` to clean up underlying resources.
///
/// # Arguments
///
/// * `names`: Pointer to the array of strings we want to release resources of
/// * `len`: Number of strings hold by `names` array
///
/// returns: `Status::Ok == 0` if success, any other status code if an error what caught up
Status safetensors_free_names(const char *const *names, unsigned int len);

/// Return the number of tensors stored in this safetensors
///
/// # Arguments
///
/// * `handle`: Pointer to the underlying safetensors we want to know the number of tensors of.
///
/// returns: usize Number of tensors in the safetensors
uintptr_t safetensors_num_tensors(const Handle *handle);

/// Return the number of bytes required to represent a single element from the specified dtype
///
/// # Arguments
///
/// * `dtype`: The data type we want to know the number of bytes required
///
/// returns: usize Number of bytes for this specific `dtype`
uintptr_t safetensors_dtype_size(Dtype dtype);

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
Status safetensors_get_tensor(const Handle *handle, View **view, const char *name);

/// Free the resources used by a TensorView to expose metadata + content to the C-FFI layer
///
/// # Arguments
///
/// * `ptr`: Pointer to the TensorView we want to release the underlying resources of
///
/// returns: `Status::Ok = 0` if resources were successfully freed
Status safetensors_free_tensor(View *ptr);

} // extern "C"
