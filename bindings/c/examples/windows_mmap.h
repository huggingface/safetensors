#include <windows.h>
#include <stdio.h>
#include <tchar.h>
#define MAX_FILE_SIZE 0x10000000

const uint8_t* map_file(const char* filename, size_t& length){
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
}
