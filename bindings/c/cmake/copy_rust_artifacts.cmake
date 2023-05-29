list(APPEND LIBRARY_EXTENSIONS ".a" ".so" ".dylib" ".dll" ".lib")


file(GLOB_RECURSE SAFETENSORS_ARTIFACTS ${CMAKE_BINARY_DIR} [a-z]*safetensors.*)
list(FILTER SAFETENSORS_ARTIFACTS EXCLUDE REGEX "^${CMAKE_BINARY_DIR}/.*/deps")

# Sometime we get .lib file on Windows prefixed with .dll (.dll.lib), let's remove the .dll

foreach (SAFETENSORS_ARTIFACT IN LISTS SAFETENSORS_ARTIFACTS)
    cmake_path(SET SAFETENSORS_ARTIFACT_PATH ${SAFETENSORS_ARTIFACT})
    cmake_path(GET SAFETENSORS_ARTIFACT_PATH EXTENSION LAST_ONLY ARTIFACT_EXT)

    # Check if the extension is in the library extensions
    list(FIND LIBRARY_EXTENSIONS ${ARTIFACT_EXT} IS_LIBRARY_ARTIFACT)
    if(NOT ${IS_LIBRARY_ARTIFACT} EQUAL -1)

        # Sometime we get .lib file on Windows prefixed with .dll (.dll.lib), let's remove the .dll
        string(REGEX REPLACE "(.*)/safetensors.dll.(.*)$" "\\1/safetensors.\\2" SAFETENSORS_ARTIFACT_CLEAN_NAME ${SAFETENSORS_ARTIFACT})
        string(REGEX REPLACE "${CMAKE_BINARY_DIR}/(.*)/(.*safetensors).(.*)$" "${CMAKE_BINARY_DIR}/\\2.\\3" SAFETENSORS_ARTIFACT_TRIM_DEST ${SAFETENSORS_ARTIFACT_CLEAN_NAME})
        message(STATUS "Copying file ${SAFETENSORS_ARTIFACT} to ${SAFETENSORS_ARTIFACT_TRIM_DEST}")

        file(COPY_FILE ${SAFETENSORS_ARTIFACT} ${SAFETENSORS_ARTIFACT_TRIM_DEST} ONLY_IF_DIFFERENT)
    endif()
endforeach ()