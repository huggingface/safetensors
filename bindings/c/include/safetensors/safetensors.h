#ifdef __cplusplus
extern "C" {
#endif

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * Represents all the status code returned by safetensors
 */
typedef enum {
    SAFETENSORS_OK = 0,
    SAFETENSORS_NULL_BUFFER = -1,
    SAFETENSORS_UTF8_ERROR = -2,
    SAFETENSORS_FORMAT_ERROR = -10
} safetensors_status_t;


/**
 * Represent all the different data type support by safetensors
 */
typedef enum {
    /// Boolean type
    BOOL,
    /// Unsigned byte
    UINT8,
    /// Signed byte
    INT8,
    /// Signed integer (16-bit)
    INT16,
    /// Unsigned integer (16-bit)
    UINT16,
    /// Half-precision floating point
    FLOAT16,
    /// Brain floating point
    BFLOAT16,
    /// Signed integer (32-bit)
    INT32,
    /// Unsigned integer (32-bit)
    UINT32,
    /// Floating point (32-bit)
    FLOAT32,
    /// Floating point (64-bit)
    FLOAT64,
    /// Signed integer (64-bit)
    INT64,
    /// Unsigned integer (64-bit)
    UINT64,
} safetensors_dtype_t;


/**
 * Opaque struct holding safetensors and data references
 */
typedef struct safetensors_handle_t safetensors_handle_t;

/**
 * Represents a tensor deserialized from the underlying safetensors.
 * No copy is made through the FFI layer.
 */
typedef struct safetensors_view_t {
    const safetensors_dtype_t dtype;
    const uintptr_t rank;
    const uintptr_t *shapes;
    const char * data;
} safetensors_view_t;

/**
 * Read `bufferLen` bytes from `buffer` storing the underlying safetensors reference into `handle` pointer.
 * @param handle An in-out pointer used to store the pointer to the deserialized safetensors content.
 * @param buffer The buffer to read from
 * @param bufferLen The number of bytes it is safe to read from the `buffer`
 * @return `safetensors_status_t::SAFETENSORS_OK` if deserialization succeeded, any other status code otherwise
 */
safetensors_status_t safetensors_deserialize(safetensors_handle_t **handle, const char *buffer, size_t bufferLen);


/**
 * Destroy a previously allocated safetensors pointer `handle`
 * @param handle The pointer we want to free the resources of
 */
void safetensors_destroy(safetensors_handle_t *handle);


/**
 * Return the number of tensors contained in this safetensors
 * @param handle The pointer to the safetensors we want to retrieve the number of tensors
 * @return Positive or zero (if empty) unsigned integers indicating the number of tensors
 */
uintptr_t safetensors_num_tensors(const safetensors_handle_t *handle);

/**
 * Return the size (in bytes) of an element of the specified `dtype`
 * @param dtype The dtype to query the number of bytes
 * @return Positive unsigned number
 */
uintptr_t safetensors_dtype_size(safetensors_dtype_t dtype);


/**
 * Return the tensor names currently stored in the underlying `handle` safetensors
 * @param handle The pointer to the safetensors we want to retrieve the tensor names
 * @param ptr In-out pointer to store an array of strings
 * @param len The number of items stored in `ptr` (i.e. number of tensors)
 */
uint32_t safetensors_names(const safetensors_handle_t *handle, char const * const * *ptr, uint32_t *len);

/**
 * Release the underlying memory used to store tensor's names following a call to `safetensors_names`
 * @param names The pointer to the array of string we want to release
 * @param len The number of items in the array
 * @return `safetensors_status_t::SAFETENSORS_OK` if deserialization succeeded, any other status code otherwise
 */
uint32_t safetensors_free_names(const char * const * names, uintptr_t len);

/**
 * Retrieve a view (i.e. no copy) of a tensor referenced by `name` and populator the pointer `view` with the content
 * and metadata for this specific tensor
 * @param handle The pointer to the safetensors we want to retrieve a tensor from
 * @param view In-out pointer to a `safetensors_view_t` struct holding metadata and tensor data for the referenced tensor
 * @param name The name of the tensor (nul-terminated) we want to retrieve the content of from the underlying safetensors
 * @return `safetensors_status_t::SAFETENSORS_OK` if deserialization succeeded, any other status code otherwise
 */
safetensors_status_t safetensors_get_tensor(const safetensors_handle_t *handle, safetensors_view_t **view, const char *name);

/**
 * Release the resources hold by the provided safetensors_view_t previously allocated from a call to `safetensors_get_tensor`
 * @param ptr Pointer to the `safetensors_view_t` we want to release the resources of
 * @return `safetensors_status_t::SAFETENSORS_OK` if deserialization succeeded, any other status code otherwise
 */
safetensors_status_t safetensors_free_tensor(safetensors_view_t *ptr);


#ifdef __cplusplus
}
#endif