#ifdef __cplusplus
extern "C" {
#endif

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

typedef enum {
    SAFETENSORS_OK = 0,
    SAFETENSORS_NULL_BUFFER = -1,
    SAFETENSORS_UTF8_ERROR = -2,
    SAFETENSORS_FORMAT_ERROR = -10
} safetensors_status_t;


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


typedef struct safetensors_handle_t safetensors_handle_t;
typedef struct safetensors_view_t {
    safetensors_dtype_t dtype;
    uintptr_t rank;
    uintptr_t *shapes;
    char * data;
} safetensors_view_t;

/**
 * 
 * @param handle 
 * @param buffer
 * @param buffer_len
 * @return
 */
uint32_t safetensors_deserialize(safetensors_handle_t **handle, const char *buffer, size_t buffer_len);


/*
 * 
 * @param handle
 */
void safetensors_destroy(safetensors_handle_t *handle);


/**
 * 
 * @param handle 
 */
uintptr_t safetensors_num_tensors(const safetensors_handle_t *handle);

/**
 *
 * @param dtype
 * @return
 */
uintptr_t safetensors_dtype_size(safetensors_dtype_t dtype);


/**
 *
 */
uint32_t safetensors_names(const safetensors_handle_t *handle, char const * const * *ptr, uint32_t *len);

/**
 *
 * @param names
 * @param len
 * @return
 */
uint32_t safetensors_free_names(const char * const * names, uintptr_t len);

/**
 *
 * @param handle
 * @param view
 * @param name
 * @return
 */
safetensors_status_t safetensors_get_tensor(const safetensors_handle_t *handle, safetensors_view_t **view, const char *name);

/**
*
* @param ptr
* @return
*/
safetensors_status_t safetensors_free_tensor(safetensors_view_t *ptr);


#ifdef __cplusplus
}
#endif