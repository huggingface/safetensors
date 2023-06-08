cmake_minimum_required(VERSION 3.21)

project(safetensors_c VERSION 0.3.0 LANGUAGES C CXX DESCRIPTION "Simple, safe way to store and distribute tensors")
set(SAFETENSORS_PROJECT_NAME ${PROJECT_NAME})
set(CMAKE_CXX_STANDARD 11)

# Define options
option(SAFETENSORS_BUILD_TESTS "Build the safetensors C-API tests" OFF)
option(SAFETENSORS_BUILD_EXAMPLES "Build the safetensors C-API examples" OFF)

# Define the c library header location
set(SAFETENSORS_C_HEADERS include/safetensors/safetensors.h)
add_library(safetensors_c_api INTERFACE)

target_include_directories(safetensors_c_api INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
)

# Build up Rust part
include(FetchContent)

FetchContent_Declare(
    Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
    GIT_TAG v0.4 # Optionally specify a commit hash, version tag or branch here
)
FetchContent_MakeAvailable(Corrosion)


# Check if we need to build the unit tests
if(SAFETENSORS_BUILD_TESTS)
    corrosion_import_crate(MANIFEST_PATH Cargo.toml CRATE_TYPES staticlib PROFILE release)

    message(STATUS "Building tests")

    enable_testing()

    FetchContent_Declare(
        catch2
        GIT_REPOSITORY https://github.com/catchorg/catch2
        GIT_TAG v3.3.2
    )
    FetchContent_MakeAvailable(catch2)
    add_subdirectory(tests)
else()
    corrosion_import_crate(MANIFEST_PATH Cargo.toml CRATE_TYPES cdylib PROFILE release)
endif()

if(SAFETENSORS_BUILD_EXAMPLES)
    message(STATUS "Building examples")

    add_subdirectory(examples)
endif()
