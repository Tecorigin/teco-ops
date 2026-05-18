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

namespace tecoops {

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

ual::common::UALDataType Convert::toUALDataType(tecoopsAlgo_t algo) {
    return ual::common::UAL_DTYPE_FLOAT;
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