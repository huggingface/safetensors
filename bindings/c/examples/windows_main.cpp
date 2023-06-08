// Copied from https://learn.microsoft.com/en-us/windows/win32/memory/creating-a-view-within-a-file
#include "safetensors/safetensors.h"
#include <windows.h>
#include <stdio.h>
#include <tchar.h>

/* The test file. The code below creates the file and populates it,
   so there is no need to supply it in advance. */

TCHAR * filename = TEXT("model.safetensors"); // the file to be manipulated

#define BUF_SIZE 256

int main(void)
{
  HANDLE hMapFile;
  LPCTSTR pBuf;
  DWORD dwFileSize;
  LPVOID lpMapAddress;

  // Create the test file. Open it "Create Always" to overwrite any
  // existing file. The data is re-created below
  // mapping = CreateFileMappingW(handle, ptr::null_mut(), protect, 0, 0, ptr::null());
  hFile = CreateFileMapping(
    INVALID_HANDLE_VALUE,
    NULL,
    PAGE_READONLY,
    0,
    BUF_SIZE,
    filename);               // name of mapping object
                             //
  dwFileSize = GetFileSize(hFile,  NULL);
  _tprintf(TEXT("hFile size: %10d\n"), dwFileSize);

  if (hFile == NULL)
  {
    _tprintf(TEXT("hFile is NULL\n"));
    _tprintf(TEXT("Target file is %s\n"),
             filename);
    return 1;
  }

  pBuf = (LPTSTR)MapViewOfFile(hMapFile, // handle to map object
    FILE_MAP_ALL_ACCESS,  // read/write permission
    0,
    0,
    BUF_SIZE);

    if (pBuf == NULL)
    {
        _tprintf(TEXT("Could not map view of file (%d).\n"),
            GetLastError());

        CloseHandle(hMapFile);
        return 1;
    }

  // Verify that the correct file size was written.

  // Create a file mapping object for the file
  // Note that it is a good idea to ensure the file size is not zero
  hMapFile = CreateFileMapping( hFile,          // current file handle
                NULL,           // default security
                PAGE_READWRITE, // read/write permission
                0,              // size of mapping object, high
                dwFileMapSize,  // size of mapping object, low
                NULL);          // name of mapping object

  if (hMapFile == NULL)
  {
    _tprintf(TEXT("hMapFile is NULL: last error: %d\n"), GetLastError() );
    return (2);
  }

  // Map the view and test the results.

  lpMapAddress = MapViewOfFile(hMapFile,            // handle to
                                                    // mapping object
                               FILE_MAP_READ, // read/write
                               0,                   // high-order 32
                                                    // bits of file
                                                    // offset
                               dwFileMapStart,      // low-order 32
                                                    // bits of file
                                                    // offset
                               dwMapViewSize);      // number of bytes
                                                    // to map
  if (lpMapAddress == NULL)
  {
    _tprintf(TEXT("lpMapAddress is NULL: last error: %d\n"), GetLastError());
    return 1;
  }

  // Calculate the pointer to the data.
  Handle* handle = nullptr;
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
