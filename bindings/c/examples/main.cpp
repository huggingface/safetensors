#include "safetensors/safetensors.h"

#ifdef __unix__
    #include "unix_mmap.h"
#elif defined(_WIN32) || defined(WIN32)
    #include "windows_mmap.h"
#endif

const char* dtype_to_str(Dtype dtype){
    switch(dtype) {
    case Dtype::F32:
        return "F32";
    case Dtype::I32:
        return "I32";
    case Dtype::F16:
        return "F16";
    // TODO Fill in this, this is good enough for a demo.
    default:
        return "Unknown dtype";
    }
}

int main(){
    Handle* handle = nullptr;
    View *tensor = nullptr;
    const char *const * names;
    unsigned int names_length;
    auto filename = "model.safetensors";
    size_t length;

    printf("Starting example\n");
    auto f = map_file(filename, length);
    auto result = safetensors_deserialize(&handle, f, length);
    if (result != Status::Ok){
        printf("Could not open safetensors file %s\n", filename);
        return 1;
    }

    result = safetensors_names(handle, &names, &names_length);
    if (result != Status::Ok){
        printf("Could not get tensor names %s\n", filename);
        return 1;
    }

    for (int i=0; i<names_length; i++){
        auto tensor_name =  names[i];
        result = safetensors_get_tensor(handle, &tensor, tensor_name);
        if (result != Status::Ok){
            printf("Could find tensor %s\n", tensor_name);
            return 1;
        }else{
            printf("Found tensor %s: rank %d\n", tensor_name, tensor->rank);
            printf("  Dtype: %s\n", dtype_to_str(tensor->dtype));
            printf("  Data offsets: %i-%i\n", tensor->start, tensor->stop);
            printf("  Shape:\n    ");
            for (int i=0; i < tensor->rank; i++){
                printf("%d ", tensor->shape[i]);
            }
            printf("\n");
        }
    }
    return 0;
}
