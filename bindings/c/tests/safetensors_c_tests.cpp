//
// Created by momo- on 5/28/2023.
//
#ifndef SAFETENSORS_SAFETENSORS_C_TESTS_H
#define SAFETENSORS_SAFETENSORS_C_TESTS_H

#include "catch2/catch_test_macros.hpp"
#include "safetensors/safetensors.h"

static const uint8_t ONE_ELEMENT_DATA[] = {
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
static const uint8_t TOO_SMALL_HEADER[] = {
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
    REQUIRE(safetensors_dtype_size(Dtype::BOOL) == 1);
    REQUIRE(safetensors_dtype_size(Dtype::U8) == 1);
    REQUIRE(safetensors_dtype_size(Dtype::I8) == 1);
    REQUIRE(safetensors_dtype_size(Dtype::U16) == 2);
    REQUIRE(safetensors_dtype_size(Dtype::I16) == 2);
    REQUIRE(safetensors_dtype_size(Dtype::F16) == 2);
    REQUIRE(safetensors_dtype_size(Dtype::BF16) == 2);
    REQUIRE(safetensors_dtype_size(Dtype::U32) == 4);
    REQUIRE(safetensors_dtype_size(Dtype::I32) == 4);
    REQUIRE(safetensors_dtype_size(Dtype::F32) == 4);
    REQUIRE(safetensors_dtype_size(Dtype::U64) == 8);
    REQUIRE(safetensors_dtype_size(Dtype::I64) == 8);
    REQUIRE(safetensors_dtype_size(Dtype::F64) == 8);
}


TEST_CASE("Deserialize safetensors", "[safetensors][cpu]") {
    Handle* handle = nullptr;

    SECTION("Single element") {
        REQUIRE(safetensors_deserialize(&handle, ONE_ELEMENT_DATA, sizeof(ONE_ELEMENT_DATA)) == Status::Ok);

        // Deserialize the safetensors and initialize pointer
        SECTION("Deserialize safetensors and init pointer") {
            REQUIRE_FALSE(handle == nullptr);
        }

        SECTION("Retrieve metadata about safetensors") {
            const char *const *names = nullptr;
            uint32_t numTensors = 0;

            // Retrieve info about the safetensors
            auto nb = safetensors_num_tensors(handle);
            REQUIRE(safetensors_names(handle, &names, &numTensors) == Status::Ok);
            REQUIRE(numTensors == nb);

            // Free names
            REQUIRE(safetensors_free_names(names, numTensors) == Status::Ok);
        }

        SECTION("Retrieve tensor & free") {
            View *view = nullptr;
            REQUIRE(safetensors_get_tensor(handle, &view, "test") == Status::Ok);

            REQUIRE(view->dtype == Dtype::I32);
            REQUIRE(safetensors_dtype_size(view->dtype) == 4);
            REQUIRE(view->rank == 2);
            REQUIRE(view->shape[0] == 2);
            REQUIRE(view->shape[1] == 2);

            REQUIRE(safetensors_free_tensor(view) == Status::Ok);
        }

        return;
    }

    safetensors_destroy(handle);
}

TEST_CASE("Error status mapping", "[safetensors][status]") {
    Handle* handle = nullptr;

    SECTION("Null buffer") {
        REQUIRE(safetensors_deserialize(&handle, nullptr, 10) == Status::NullPointer);
    }

    SECTION("Invalid header") {
        REQUIRE(safetensors_deserialize(&handle, INVALID_HEADER, sizeof INVALID_HEADER) == Status::InvalidHeader);
    }

    SECTION("Header too small") {
        REQUIRE(safetensors_deserialize(&handle, TOO_SMALL_HEADER, sizeof TOO_SMALL_HEADER) == Status::HeaderTooSmall);
    }

    SECTION("Header too large") {
        REQUIRE(safetensors_deserialize(&handle, TOO_LARGE_HEADER, sizeof TOO_LARGE_HEADER) == Status::HeaderTooLarge);
    }

    SECTION("Header invalid length (overflow buffer len)") {
        REQUIRE(safetensors_deserialize(&handle, INVALID_HEADER_LENGTH, sizeof INVALID_HEADER_LENGTH) == Status::InvalidHeaderLength);
    }

    SECTION("Header not JSON") {
        REQUIRE(safetensors_deserialize(&handle, INVALID_HEADER_DESERIALIZATION, sizeof INVALID_HEADER_DESERIALIZATION) == Status::InvalidHeaderDeserialization);
    }

    SECTION("Tensor not found") {
        View *tensor = nullptr;
        REQUIRE(safetensors_deserialize(&handle, ONE_ELEMENT_DATA, sizeof ONE_ELEMENT_DATA) == Status::Ok);
        REQUIRE(safetensors_get_tensor(handle, &tensor, "test") == Status::Ok);
        REQUIRE(safetensors_get_tensor(handle, &tensor, "test_") == Status::TensorNotFound);
    }

    if(handle != nullptr)
        safetensors_destroy(handle);
}

#endif //SAFETENSORS_SAFETENSORS_C_TESTS_H
