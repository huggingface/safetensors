# Hugging Face Safetensors C-API 

This folder contains the necessary items to build a C-compatible API to include in (almost) any project with
C-compatible ABI.

API definition can be found over `include/safetensors/safetensors.h` and binding can be build through CMake build tool.

# Building the library

```shell
git clone https://github.com/huggingface/safetensors
cd safetensors
cd bindings/c
mkdir build && cd build
cmake ..
make
```

# Add as CMake dependency

```cmake
add_subdirectory(<safetensors_root_dir>/bindings/c)
target_link_libraries(<target> safetensors_c_api)
```
