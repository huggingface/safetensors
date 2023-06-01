//
// Created by momo- on 5/28/2023.
//
#ifndef SAFETENSORS_SAFETENSORS_C_TESTS_H
#define SAFETENSORS_SAFETENSORS_C_TESTS_H

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

// Header Length = 1 (0x01) and header content [0xF5] is invalid utf8 content
static const uint8_t INVALID_HEADER[] = {
    0x01, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xF5
};

// Header < 8 bytes
static const char TOO_SMALL_HEADER[] = {
        '<', 0x0,
};

// header way too big
static const uint8_t TOO_LARGE_HEADER[] = {
        '<', 0x0, 0x0, 0x0, 0x05, 0xf5, 0xe1, 0x01
};


static const uint8_t INVALID_HEADER_LENGTH[] = {
        0xFF, 0xFF, 0x0,0x0, 0x0, 0x0, 0x0, 0x0, '-'
};

static const uint8_t INVALID_HEADER_DESERIALIZATION[] = {
        0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, '-'
};


TEST_CASE("Ensure dtype size match", "[safetensors][dtype]") {
    REQUIRE(safetensors_dtype_size(safetensors_dtype_t::BOOL) == 1);
    REQUIRE(safetensors_dtype_size(safetensors_dtype_t::UINT8) == 1);
    REQUIRE(safetensors_dtype_size(safetensors_dtype_t::INT8) == 1);
    REQUIRE(safetensors_dtype_size(safetensors_dtype_t::UINT16) == 2);
    REQUIRE(safetensors_dtype_size(safetensors_dtype_t::INT16) == 2);
    REQUIRE(safetensors_dtype_size(safetensors_dtype_t::FLOAT16) == 2);
    REQUIRE(safetensors_dtype_size(safetensors_dtype_t::BFLOAT16) == 2);
    REQUIRE(safetensors_dtype_size(safetensors_dtype_t::UINT32) == 4);
    REQUIRE(safetensors_dtype_size(safetensors_dtype_t::INT32) == 4);
    REQUIRE(safetensors_dtype_size(safetensors_dtype_t::FLOAT32) == 4);
    REQUIRE(safetensors_dtype_size(safetensors_dtype_t::UINT64) == 8);
    REQUIRE(safetensors_dtype_size(safetensors_dtype_t::INT64) == 8);
    REQUIRE(safetensors_dtype_size(safetensors_dtype_t::FLOAT64) == 8);
}


TEST_CASE("Deserialize safetensors", "[safetensors][cpu]") {
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

        SECTION("Retrieve tensor & free") {
            safetensors_view_t *view = nullptr;
            REQUIRE(safetensors_get_tensor(handle, &view, "test") == SAFETENSORS_OK);

            REQUIRE(view->dtype == safetensors_dtype_t::INT32);
            REQUIRE(safetensors_dtype_size(view->dtype) == 4);
            REQUIRE(view->rank == 2);
            REQUIRE(view->shapes[0] == 2);
            REQUIRE(view->shapes[1] == 2);

            REQUIRE(safetensors_free_tensor(view) == SAFETENSORS_OK);
        }

        return;
    }

    safetensors_destroy(handle);
}

TEST_CASE("Error status mapping", "[safetensors][status]") {
    safetensors_handle_t* handle = nullptr;

    SECTION("Null buffer") {
        REQUIRE(safetensors_deserialize(&handle, nullptr, 10) == SAFETENSORS_NULL_BUFFER);
    }

    SECTION("Invalid header") {
        REQUIRE(safetensors_deserialize(&handle, (const char *) INVALID_HEADER, sizeof INVALID_HEADER) == SAFETENSORS_INVALID_HEADER);
    }

    SECTION("Header too small") {
        REQUIRE(safetensors_deserialize(&handle, TOO_SMALL_HEADER, sizeof TOO_SMALL_HEADER) == SAFETENSORS_HEADER_TOO_SMALL);
    }

    SECTION("Header too large") {
        REQUIRE(safetensors_deserialize(&handle, (const char *)TOO_LARGE_HEADER, sizeof TOO_LARGE_HEADER) == SAFETENSORS_HEADER_TOO_LARGE);
    }

    SECTION("Header invalid length (overflow buffer len)") {
        REQUIRE(safetensors_deserialize(&handle, (const char *)INVALID_HEADER_LENGTH, sizeof INVALID_HEADER_LENGTH) == SAFETENSORS_INVALID_HEADER_LENGTH);
    }

    SECTION("Header not JSON") {
        REQUIRE(safetensors_deserialize(&handle, (const char *)INVALID_HEADER_DESERIALIZATION, sizeof INVALID_HEADER_DESERIALIZATION) == SAFETENSORS_INVALID_HEADER_DESERIALIZATION);
    }

    SECTION("Tensor not found") {
        safetensors_view_t *tensor = nullptr;
        REQUIRE(safetensors_deserialize(&handle, ONE_ELEMENT_DATA, sizeof ONE_ELEMENT_DATA) == SAFETENSORS_OK);
        REQUIRE(safetensors_get_tensor(handle, &tensor, "test") == SAFETENSORS_OK);
        REQUIRE(safetensors_get_tensor(handle, &tensor, "test_") == SAFETENSORS_TENSOR_NOT_FOUND);
    }

    if(handle != nullptr)
        safetensors_destroy(handle);
}

#endif //SAFETENSORS_SAFETENSORS_C_TESTS_H
