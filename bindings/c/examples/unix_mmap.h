#include <sys/mman.h>
#include <sys/stat.h>
#include <iostream>
#include <fcntl.h>

const uint8_t* map_file(const char* filename, size_t& length){
    int fd = open(filename, O_RDONLY);
    if (fd == -1){
        printf("Cannot open file : %s\n", filename);
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
    return (const uint8_t*) addr;
}
