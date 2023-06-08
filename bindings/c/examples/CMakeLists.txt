IF (WIN32)
  # set stuff for windows
    add_executable(main main.cpp)
ELSE()
  # set stuff for other systems
    add_executable(main windows_main.cpp)
ENDIF()

target_link_libraries(main PUBLIC safetensors_c_api safetensors)
