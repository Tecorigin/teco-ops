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

#ifndef TECOOPS_INTERFACE_INCLUDE_TECOOPS_H_
#define TECOOPS_INTERFACE_INCLUDE_TECOOPS_H_

#include <sdaa_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    TECOOPS_STATUS_SUCCESS = 0,
    TECOOPS_STATUS_NOT_INITIALIZED = 1,
    TECOOPS_STATUS_ALLOC_FAILED = 2,
    TECOOPS_STATUS_BAD_PARAM = 3,
    TECOOPS_STATUS_INTERNAL_ERROR = 4,
    TECOOPS_STATUS_INVALID_VALUE = 5,
    TECOOPS_STATUS_NOT_SUPPORTED = 6,
} tecoopsStatus_t;

typedef enum {
    TECOOPS_ALGO_0 = 0,
    TECOOPS_ALGO_1,
    TECOOPS_ALGO_2,
    TECOOPS_ALGO_3,
    TECOOPS_ALGO_4,
    TECOOPS_ALGO_5,
    TECOOPS_ALGO_6,
    TECOOPS_ALGO_7,
    TECOOPS_ALGO_8,
    TECOOPS_ALGO_9,
} tecoopsAlgo_t;

typedef struct tecoopsContext *tecoopsHandle_t;

tecoopsStatus_t tecoopsCreate(tecoopsHandle_t *handle);
tecoopsStatus_t tecoopsDestroy(tecoopsHandle_t handle);
tecoopsStatus_t tecoopsSetStream(tecoopsHandle_t handle, sdaaStream_t stream);
tecoopsStatus_t tecoopsGetStream(tecoopsHandle_t handle, sdaaStream_t *stream);
const char *tecoopsGetErrorString(tecoopsStatus_t status);

tecoopsStatus_t tecoopsFlattenRays(
    tecoopsHandle_t handle,
    const int *rays, uint32_t N, uint32_t M, int *res,
    tecoopsAlgo_t algo);

#ifdef __cplusplus
}
#endif

#endif  // TECOOPS_INTERFACE_INCLUDE_TECOOPS_H_