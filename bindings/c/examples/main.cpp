#include "safetensors/safetensors.h"

#ifdef __unix__
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <iostream>
    #include <fcntl.h>
#elif defined(_WIN32) || defined(WIN32)
    #include <windows.h>
    #include <stdio.h>
    #include <tchar.h>
#endif
  
  
const uint8_t* map_file(const char* fname, size_t& length){
#ifdef __unix__
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
      return (const uint8_t*) addr;
#elif defined(_WIN32) || defined(WIN32)
  HANDLE hFile;
  LPCTSTR pBuf;
  DWORD length;
  LPVOID lpMapAddress;

  // Create the test file. Open it "Create Always" to overwrite any
  // existing file. The data is re-created below
  // mapping = CreateFileMappingW(handle, ptr::null_mut(), protect, 0, 0, ptr::null());
  hFile = CreateFileMapping(
    INVALID_HANDLE_VALUE,
    NULL,
    PAGE_READONLY,
    0,
    MAX_FILE_SIZE,
    filename);               // name of mapping object
                             //
  if (hFile == NULL)
  {
    printf("Cannot open file : %s\n", filename);
    printf("Try downloading a file on the hub\n");
    printf("wget https://huggingface.co/gpt2/resolve/main/model.safetensors");
    return 1;
  }

  length = GetFileSize(hFile,  NULL);
  if (length <= 0){
    printf("Cannot open file : %s\n", filename);
    printf("Try downloading a file on the hub\n");
    printf("wget https://huggingface.co/gpt2/resolve/main/model.safetensors");
  }

  lpMapAddress = MapViewOfFile(hFile,
                               FILE_MAP_READ,
                               0,
                               0,
                               MAX_FILE_SIZE);
                                              
  if (lpMapAddress == NULL)
  {
    printf("lpMapAddress is NULL: last error: %d\n", GetLastError());
    return 1;
  }
  return (const uint8_t*) lpMapAddress;
#endif
}





int main(){
    Handle* handle = nullptr;
    size_t length;
    printf("Starting example\n");
    auto f = map_file("model.safetensors", length);
    auto result = safetensors_deserialize(&handle, f, length);
    if (result != Status::Ok){
        printf("Could not open safetensors file");
    }
    View *tensor = nullptr;
    auto tensor_name =  "wpe.weight";
    result = safetensors_get_tensor(handle, &tensor, tensor_name);
    if (result != Status::Ok){
        printf("Could not open find tensor\n");
        return 1;
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
