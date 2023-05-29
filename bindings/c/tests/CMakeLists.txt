# Setup catch2
set(CATCH_INSTALL_DOCS CACHE BOOL OFF)
set(CATCH_DEVELOPMENT_BUILD CACHE BOOL OFF)

# Create target
add_executable(safetensors_tests safetensors_c_tests.cpp)

link_directories(${CMAKE_BINARY_DIR})

target_link_libraries(safetensors_tests PUBLIC safetensors)
target_link_libraries(safetensors_tests PRIVATE Catch2::Catch2WithMain)

include(CTest)
include(Catch)

catch_discover_tests(safetensors_tests)