//
// Created by momo- on 5/28/2023.
//
#ifndef SAFETENSORS_SAFETENSORS_C_TESTS_H
#define SAFETENSORS_SAFETENSORS_C_TESTS_H

#include <iostream>
#include <string>
#include "catch2/catch_test_macros.hpp"
#include "safetensors/safetensors.h"

const std::string ONE_ELEMENT_DATA = R"(b"<\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")";

TEST_CASE("Deserialize", "[safetensors][cpu]") {
    std::cout << safetensors_version() << std::endl;
//    safetensors_handle_t* handle = nullptr;
//
//    SECTION("Single element") {
//        safetensors_deserialize(handle, ONE_ELEMENT_DATA.c_str(), ONE_ELEMENT_DATA.size());
//        REQUIRE(handle != nullptr);
//        REQUIRE(safetensors_num_tensors(handle) == 1);
//    }
//
//    safetensors_destroy(handle);
}



#endif //SAFETENSORS_SAFETENSORS_C_TESTS_H
