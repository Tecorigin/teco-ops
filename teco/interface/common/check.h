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

#ifndef TECOOPS_INTERFACE_COMMON_CHECK_H_
#define TECOOPS_INTERFACE_COMMON_CHECK_H_

#include <cstdio>

#include "interface/common/convert.h"
#include "interface/include/tecoops.h"

namespace tecoops {

#define checkUalStatus(a)                                                                          \
    do {                                                                                           \
        tecoops::ual::common::Status __status = a;                                                 \
        if (tecoops::ual::common::Status::SUCCESS != (__status)) {                                 \
            fprintf(stderr, "tecoops runtime error in line %d of file %s : %d \n", __LINE__,       \
                    __FILE__, static_cast<int>(__status));                                         \
            return __status;                                                                       \
        }                                                                                          \
    } while (0);

#define checkUalStatusInTecoops(a)                                                                 \
    do {                                                                                           \
        tecoops::ual::common::Status __status = a;                                                 \
        if (tecoops::ual::common::Status::SUCCESS != (__status)) {                                 \
            fprintf(stderr, "tecoual runtime error in line %d of file %s : %d \n", __LINE__,       \
                    __FILE__, static_cast<int>(__status));                                         \
            return tecoops::Convert::toStatus(__status);                                           \
        }                                                                                          \
    } while (0);

#define checkTecoopsStatus(status)                                                                 \
    do {                                                                                           \
        tecoopsStatus_t __errcode = status;                                                        \
        if (__errcode != TECOOPS_STATUS_SUCCESS) {                                                 \
            fprintf(stderr, "TECOOPS Error: %d\n", __errcode);                                     \
            return __errcode;                                                                      \
        }                                                                                          \
    } while (0);

}  // namespace tecoops

#endif  // TECOOPS_INTERFACE_COMMON_CHECK_H_