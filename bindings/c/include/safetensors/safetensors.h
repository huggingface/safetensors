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


typedef struct safetensors_handle_t safetensors_handle_t;
typedef struct safetensors_view_t safetensors_view_t;


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
 safetensors_status_t get_tensor(const safetensors_view_t *handle, safetensors_view_t *view, const char *name);

#ifdef __cplusplus
}
#endif