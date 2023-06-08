// Copied from https://learn.microsoft.com/en-us/windows/win32/memory/creating-a-view-within-a-file
#include "safetensors/safetensors.h"
#include <windows.h>
#include <stdio.h>
#include <tchar.h>

/* The test file. The code below creates the file and populates it,
   so there is no need to supply it in advance. */

TCHAR * filename = TEXT("model.safetensors"); // the file to be manipulated

#define MAX_FILE_SIZE 0x10000000

int main(void)
{
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


  // Map the view and test the results.

  lpMapAddress = MapViewOfFile(hFile,            // handle to
                                                    // mapping object
                               FILE_MAP_READ, // read/write
                               0,                   // high-order 32
                                                    // bits of file
                                                    // offset
                               0,      // low-order 32
                                                    // bits of file
                                                    // offset
                               BUF_SIZE);      // number of bytes
                                                    // to map
  if (lpMapAddress == NULL)
  {
    _tprintf(TEXT("lpMapAddress is NULL: last error: %d\n"), GetLastError());
    return 1;
  }

  // Calculate the pointer to the data.
  Handle* handle = nullptr;
  Status result = safetensors_deserialize(&handle, (const uint8_t*) lpMapAddress, length);
  if (result != Status::Ok){
      printf("Could not open safetensors file");
  }
  View *tensor = nullptr;
  auto tensor_name =  "wpe.weight";
  result = safetensors_get_tensor(handle, &tensor, tensor_name);
  if (result != Status::Ok){
      printf("Could not open find tensor\n");
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
