// BSD 3- Clause License Copyright (c) 2024, Tecorigin Co., Ltd. All rights
// reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
// OF SUCH DAMAGE.

#include "interface/common/convert.h"

#include <stdexcept>

#include "ual/com/def.h"
namespace tecoops {

using tecoops::ual::common::UALDataType;

tecoopsStatus_t Convert::toStatus(ual::common::Status status) {
    switch (status) {
        case ual::common::Status::SUCCESS:
            return TECOOPS_STATUS_SUCCESS;
        case ual::common::Status::NOT_INITIALIZED:
            return TECOOPS_STATUS_NOT_INITIALIZED;
        case ual::common::Status::BAD_PARAMETER:
            return TECOOPS_STATUS_BAD_PARAM;
        case ual::common::Status::NOT_SUPPORTED:
            return TECOOPS_STATUS_NOT_SUPPORTED;
        case ual::common::Status::NOT_IMPLEMENTED:
            return TECOOPS_STATUS_NOT_SUPPORTED;
        case ual::common::Status::NO_IMPLEMENTATION:
            return TECOOPS_STATUS_NOT_SUPPORTED;
        default:
            return TECOOPS_STATUS_INTERNAL_ERROR;
    }
}

ual::common::UALDataType Convert::toUALDataType(const tecoopsDataType_t data_type) {
    switch (data_type) {
        case tecoopsDataType_t::TECOOPS_DATA_BOOL:
            return UALDataType::UAL_DTYPE_BOOL;
        case tecoopsDataType_t::TECOOPS_DATA_FLOAT:
            return UALDataType::UAL_DTYPE_FLOAT;
        case tecoopsDataType_t::TECOOPS_DATA_HALF:
            return UALDataType::UAL_DTYPE_HALF;
        case tecoopsDataType_t::TECOOPS_DATA_INT8:
            return UALDataType::UAL_DTYPE_INT8;
        case tecoopsDataType_t::TECOOPS_DATA_INT16:
            return UALDataType::UAL_DTYPE_INT16;
        case tecoopsDataType_t::TECOOPS_DATA_INT32:
            return UALDataType::UAL_DTYPE_INT32;
        case tecoopsDataType_t::TECOOPS_DATA_INT64:
            return UALDataType::UAL_DTYPE_INT64;
        case tecoopsDataType_t::TECOOPS_DATA_UINT8:
            return UALDataType::UAL_DTYPE_UINT8;
        case tecoopsDataType_t::TECOOPS_DATA_DOUBLE:
            return UALDataType::UAL_DTYPE_DOUBLE;
        case tecoopsDataType_t::TECOOPS_DATA_BFLOAT16:
            return UALDataType::UAL_DTYPE_BFLOAT16;
        case tecoopsDataType_t::TECOOPS_DATA_COMPLEX_FLOAT:
            return UALDataType::UAL_DTYPE_COMPLEX_FLOAT;  // NOLINT
        default: {
            throw std::runtime_error(
                // NOLINTNEXTLINE
                "tecoopsDataType_t convert to UALDataType failed, not exist tecoopsDataType_t enum "
                "type!");
        };
    }
}

unsigned int Convert::toDescDataTypeSize(const tecoopsDataType_t data_type) {
    switch (data_type) {
        case tecoopsDataType_t::TECOOPS_DATA_INT8:
        case tecoopsDataType_t::TECOOPS_DATA_UINT8:
        case tecoopsDataType_t::TECOOPS_DATA_BOOL: {
            return sizeof(char);
        } break;
        case tecoopsDataType_t::TECOOPS_DATA_BFLOAT16:
        case tecoopsDataType_t::TECOOPS_DATA_HALF:
        case tecoopsDataType_t::TECOOPS_DATA_INT16: {
            return 2 * sizeof(char);
        } break;
        case tecoopsDataType_t::TECOOPS_DATA_FLOAT:
        case tecoopsDataType_t::TECOOPS_DATA_INT32: {
            return 4 * sizeof(char);
        } break;
        case tecoopsDataType_t::TECOOPS_DATA_INT64:
        case tecoopsDataType_t::TECOOPS_DATA_COMPLEX_FLOAT:
        case tecoopsDataType_t::TECOOPS_DATA_DOUBLE: {
            return 8 * sizeof(char);
        } break;
        default: {
            throw std::runtime_error("not exist tecoopsDataType_t enum type!");
        } break;
    }
}

ual::common::UALAlgoType Convert::toUALAlgoType(tecoopsAlgo_t algo) {
    return static_cast<ual::common::UALAlgoType>(algo);
}

const char* Convert::toStatusStr(ual::common::Status status) {
    switch (status) {
        case ual::common::Status::SUCCESS:
            return "SUCCESS";
        case ual::common::Status::BAD_PARAMETER:
            return "BAD_PARAMETER";
        case ual::common::Status::NOT_INITIALIZED:
            return "NOT_INITIALIZED";
        case ual::common::Status::NOT_SUPPORTED:
            return "NOT_SUPPORTED";
        case ual::common::Status::NOT_IMPLEMENTED:
            return "NOT_IMPLEMENTED";
        case ual::common::Status::NO_IMPLEMENTATION:
            return "NO_IMPLEMENTATION";
        default:
            return "UNKNOWN_STATUS";
    }
}

}  // namespace tecoops