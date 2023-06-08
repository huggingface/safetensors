#include "safetensors/safetensors.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <iostream>
#include <fcntl.h>


const char* map_file(const char* fname, size_t& length)
{
    int fd = open(fname, O_RDONLY);
    if (fd == -1){
        printf("Cannot open file : %s\n", fname);
        printf("Try downloading a file on the hub\n");
        printf("wget https://huggingface.co/gpt2/resolve/main/model.safetensors");
        exit(1);
    }

    // obtain file size
    struct stat sb;
    if (fstat(fd, &sb) == -1){
        printf("Cannot state file\n");
        exit(1);
    }

    length = sb.st_size;

    const char* addr = static_cast<const char*>(mmap(NULL, length, PROT_READ, MAP_PRIVATE, fd, 0u));
    if (addr == MAP_FAILED){
        printf("Cannot map file\n");
        exit(1);
    }

    // TODO close fd at some point in time, call munmap(...)
    return addr;
}

int main(){
    Handle* handle = nullptr;
    size_t length;
    printf("Starting example\n");
    auto f = map_file("model.safetensors", length);
    auto result = safetensors_deserialize(&handle, (const uint8_t*) f, length);
    if (result != Status::Ok){
        printf("Could not open safetensors file");
    }
    View *tensor = nullptr;
    auto tensor_name =  "wpe.weight";
    result = safetensors_get_tensor(handle, &tensor, tensor_name);
    if (result != Status::Ok){
        printf("Could not open find tensor");
    }else{
        printf("Found tensor %s: rank %d\n", tensor_name, tensor->rank);
        printf("Shape:\n    ");
        for (int i=0; i < tensor->rank; i++){
            printf("%d ", tensor->shape[i]);
        }
        printf("\n");
    }
    return 0;
}
