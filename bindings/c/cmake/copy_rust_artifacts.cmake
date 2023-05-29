file(GLOB_RECURSE SAFETENSORS_ARTIFACTS ${CMAKE_BINARY_DIR} safetensors.*)
list(FILTER SAFETENSORS_ARTIFACTS EXCLUDE REGEX "^${CMAKE_BINARY_DIR}/.*/deps/safetensors*")
list(LENGTH SAFETENSORS_ARTIFACTS SAFETENSORS_ARTIFACTS_LEN)

# Sometime we get .lib file on Windows prefixed with .dll (.dll.lib), let's remove the .dll

foreach (SAFETENSORS_ARTIFACT IN LISTS SAFETENSORS_ARTIFACTS)
    # Sometime we get .lib file on Windows prefixed with .dll (.dll.lib), let's remove the .dll
    string(REGEX REPLACE "(.*)/safetensors.dll.(.*)$" "\\1/safetensors.\\2" SAFETENSORS_ARTIFACT_CLEAN_NAME ${SAFETENSORS_ARTIFACT})
    string(REGEX REPLACE "(.*)/(.*)/safetensors.(.*)$" "\\1/safetensors.\\3" SAFETENSORS_ARTIFACT_TRIM_DEST ${SAFETENSORS_ARTIFACT_CLEAN_NAME})
    message(STATUS "Copying file ${SAFETENSORS_ARTIFACT} to ${SAFETENSORS_ARTIFACT_TRIM_DEST}")

    file(COPY_FILE ${SAFETENSORS_ARTIFACT} ${SAFETENSORS_ARTIFACT_TRIM_DEST} ONLY_IF_DIFFERENT)
endforeach ()
