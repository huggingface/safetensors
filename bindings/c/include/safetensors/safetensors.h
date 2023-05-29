#ifdef __cplusplus
extern "C" {
#endif

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

typedef enum {
    SAFETENSORS_OK = 0,
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
safetensors_status_t safetensors_deserialize(safetensors_handle_t *handle, const char *buffer, size_t buffer_len);


/*
 * 
 * @param handle
 */
void safetensors_destroy(safetensors_handle_t *handle);


/**
 * 
 * @param handle 
 */
size_t safetensors_num_tensors(const safetensors_handle_t *handle);

/**
 *
 * @param handle
 * @param view
 * @param name
 * @return
 */
// safetensors_status_t SAFETENSORS_EXPORT get_tensor(const safetensors_view_t *handle, safetensors_view_t *view, const char *name);

#ifdef __cplusplus
}
#endif