# Setup catch2
set(CATCH_INSTALL_DOCS CACHE BOOL OFF)
set(CATCH_DEVELOPMENT_BUILD CACHE BOOL OFF)

# Create target
add_executable(safetensors_tests safetensors_c_tests.cpp)

# Find the Rust library
find_library(SAFETENSORS_LIB safetensors  HINTS ${CMAKE_BINARY_DIR})

# Links to the various library
target_link_libraries(safetensors_tests PRIVATE ${SAFETENSORS_LIB} safetensors_c_api Catch2::Catch2WithMain)

include(CTest)
include(Catch)

catch_discover_tests(safetensors_tests)