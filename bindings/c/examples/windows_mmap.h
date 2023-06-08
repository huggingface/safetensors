#include <windows.h>
#include <stdio.h>
#include <tchar.h>

const uint8_t* map_file(const char* filename, size_t& length){
  HANDLE file;
  HANDLE hFile;
  LPCTSTR pBuf;
  LPVOID lpMapAddress;

  // Create the test file. Open it "Create Always" to overwrite any
  // existing file. The data is re-created below
  // mapping = CreateFileMappingW(handle, ptr::null_mut(), protect, 0, 0, ptr::null());
  file = ::CreateFileA(filename, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, NULL);

  length = GetFileSize(file,  NULL);
  if (length <= 0){
    printf("Cannot open file : %s\n", filename);
    printf("Try downloading a file on the hub\n");
    printf("wget https://huggingface.co/gpt2/resolve/main/model.safetensors");
    exit(1);
  }

  hFile = CreateFileMapping(
    file,
    NULL,
    PAGE_READONLY,
    FILE_MAP_READ,
    0,
    length,
    NULL);
  if (hFile == NULL)
  {
    printf("Cannot open file : %s\n", filename);
    printf("Try downloading a file on the hub\n");
    printf("wget https://huggingface.co/gpt2/resolve/main/model.safetensors");
    exit(1);
  }


  lpMapAddress = MapViewOfFile(hFile,
                               FILE_MAP_READ,
                               0,
                               0,
                               0);
                                              
  if (lpMapAddress == NULL)
  {
    printf("lpMapAddress is NULL: last error: %d\n", GetLastError());
    exit(1);
  }
  }
  return (const uint8_t*) lpMapAddress;
}
