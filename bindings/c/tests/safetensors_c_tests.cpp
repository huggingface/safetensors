//
// Created by momo- on 5/28/2023.
//
#ifndef SAFETENSORS_SAFETENSORS_C_TESTS_H
#define SAFETENSORS_SAFETENSORS_C_TESTS_H

#include "iostream"
#include "catch2/catch_test_macros.hpp"
#include "safetensors/safetensors.h"

static const char ONE_ELEMENT_DATA[] = {
    '<', 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    '{', '"', 't', 'e', 's', 't', '"', ':',
    '{', '"', 'd', 't', 'y', 'p', 'e', '"',
    ':', '"', 'I', '3', '2', '"', ',', '"',
    's', 'h', 'a', 'p', 'e', '"', ':', '[',
    '2', ',', '2', ']', ',', '"', 'd', 'a',
    't', 'a', '_', 'o', 'f', 'f', 's', 'e',
    't', 's', '"', ':', '[', '0', ',', '1',
    '6', ']', '}', '}', 0x0, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0
};


TEST_CASE("Deserialize", "[safetensors][cpu]") {
    safetensors_handle_t* handle = nullptr;

    SECTION("Single element") {
        REQUIRE(safetensors_deserialize(&handle, ONE_ELEMENT_DATA, sizeof(ONE_ELEMENT_DATA)) == SAFETENSORS_OK);

        // Deserialize the safetensors and initialize pointer
        SECTION("Deserialize safetensors and init pointer") {
            REQUIRE_FALSE(handle == nullptr);
        }

        SECTION("Retrieve metadata about safetensors") {
            const char *const *names = nullptr;
            uint32_t numTensors = 0;

            // Retrieve info about the safetensors
            auto nb = safetensors_num_tensors(handle);
            REQUIRE(safetensors_names(handle, &names, &numTensors) == SAFETENSORS_OK);
            REQUIRE(numTensors == nb);

            // Free names
            REQUIRE(safetensors_free_names(names, numTensors) == SAFETENSORS_OK);
        }

        return;
    }

    safetensors_destroy(handle);
}



#endif //SAFETENSORS_SAFETENSORS_C_TESTS_H
