if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(SAFETENSORS_CARGO_BUILD_FOLDER "release")
    set(SAFETENSORS_CARGO_BUILD_FLAGS "--release")
    elif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    set(SAFETENSORS_CARGO_BUILD_FOLDER "release")
    set(SAFETENSORS_CARGO_TARGET_FOLDER "--release")
else()
    set(SAFETENSORS_CARGO_BUILD_FOLDER "debug")
    set(SAFETENSORS_CARGO_BUILD_FLAGS "")
endif()

message(STATUS "Rust cargo args: ${SAFETENSORS_CARGO_BUILD_FLAGS}")

add_custom_target(
    libsafetensors_rust
    COMMAND cargo build --target-dir . ${SAFETENSORS_CARGO_BUILD_FLAGS}
)

#link_directories(BEFORE ${CMAKE_BINARY_DIR}})
add_dependencies(safetensors_c_api libsafetensors_rust)

add_custom_command(
    TARGET libsafetensors_rust
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/copy_rust_artifacts.cmake
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/copy.cmake
)