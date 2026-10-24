// BSD 3- Clause License Copyright (c) 2024, Tecorigin Co., Ltd. All rights
// reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
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
#include "interface/common/macro.h"
#include "interface/include/builtin_type.h"
#include "interface/include/tecoops.h"
#include "ual/args/weight_stream_gemv_args.h"
#include "ual/ops/gemv/weight_stream_gemv.hpp"

using tecoops::ual::args::WeightStreamGemvArgs;
using tecoops::ual::args::WeightStreamGemvPatchArgs;
using tecoops::ual::ops::WeightStreamGemvOp;

// The kernel splits N into 32 contiguous row blocks and flushes each core's
// output with one DMA of my_rows halfs, so every DMA size and address must be
// 4 B aligned (the engine's own check_memcpy_async() masks address and size with
// 3; see the MemcpyAsyncStatus enum in sdaa_memcpy.h).  my_rows = N/32, so the
// flush is 4 B aligned exactly when N % 64 == 0, and each core's destination
// offset is a multiple of N/16 bytes.  The activation fetch and the weight slabs
// only need K % 16 == 0.  Anything else is rejected here (fail-closed) rather
// than silently mis-DMAed; the 16 B claim that used to be in this comment is not
// supported by the engine's validator.
static tecoopsStatus_t checkWeightStreamGemvInput(tecoopsHandle_t handle,
                                                  const void *x, const void *w,
                                                  void *y, int n, int k) {
    if (handle == nullptr) {
        return TECOOPS_STATUS_NOT_INITIALIZED;
    }
    if (x == nullptr || w == nullptr || y == nullptr) {
        return TECOOPS_STATUS_BAD_PARAM;
    }
    if (n <= 0 || k <= 0) {
        return TECOOPS_STATUS_BAD_PARAM;
    }
    if (n % 64 != 0 || k % 16 != 0) {
        return TECOOPS_STATUS_BAD_PARAM;
    }
    return TECOOPS_STATUS_SUCCESS;
}

// M=1 weight-streaming GEMV: y[n] = sum_k x[k] * w[n][k], fp16 in / fp32
// accumulate / fp16 out.  Separate kernel entity; no delivered operator is
// modified.
tecoopsStatus_t tecoopsWeightStreamGemv(tecoopsHandle_t handle,
                                        const void *x, const void *w, void *y,
                                        int n, int k) {
    tecoopsStatus_t input_error = checkWeightStreamGemvInput(handle, x, w, y, n, k);
    if (input_error != TECOOPS_STATUS_SUCCESS)
        return input_error;

    WeightStreamGemvArgs arg;
    arg.x = x;
    arg.w = w;
    arg.y = y;
    arg.n = n;
    arg.k = k;

    WeightStreamGemvPatchArgs patch_arg;
    patch_arg.atargs = &arg;

    RUN_OP(WeightStreamGemvOp, arg, patch_arg, handle);
    return TECOOPS_STATUS_SUCCESS;
}
