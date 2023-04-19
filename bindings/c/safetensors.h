#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct Handle Handle;

typedef struct View View;

struct Handle *deserialize(const uint8_t *buffer, uintptr_t buffer_len);

struct View *get_tensor(struct Handle *handle, const char *name);
