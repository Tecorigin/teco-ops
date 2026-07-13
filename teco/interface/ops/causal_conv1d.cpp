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

#include "ual/ops/causal_conv1d/causal_conv1d.hpp"

#include "interface/common/convert.h"
#include "interface/common/macro.h"
#include "interface/include/builtin_type.h"
#include "interface/include/tecoops.h"
#include "ual/args/causal_conv1d_args.h"

using tecoops::ual::args::CausalConv1dArgs;
using tecoops::ual::args::CausalConv1dPatchArgs;
using tecoops::ual::ops::CausalConv1dOp;

static tecoopsStatus_t checkCausalConv1dInput(tecoopsHandle_t handle,
                                               int batch, int totalSeqLen,
                                               int dim, int width,
                                               const void *x, const void *weight,
                                               void *out) {
    if (handle == nullptr) {
        return TECOOPS_STATUS_NOT_INITIALIZED;
    }
    if (x == nullptr || weight == nullptr || out == nullptr) {
        return TECOOPS_STATUS_BAD_PARAM;
    }
    if (batch <= 0 || totalSeqLen <= 0 || dim <= 0 || width <= 0) {
        return TECOOPS_STATUS_BAD_PARAM;
    }
    return TECOOPS_STATUS_SUCCESS;
}

tecoopsStatus_t tecoopsCausalConv1d(tecoopsHandle_t handle,
                                    int batch,
                                    int totalSeqLen,
                                    int dim,
                                    int width,
                                    int convStateStride,
                                    int xSeqStride,
                                    int outSeqStride,
                                    int padSlotId,
                                    int api_mode,
                                    int actMode,
                                    const void *x,
                                    void *convState,
                                    const void *weight,
                                    void *out,
                                    const int *queryStartLoc,
                                    const int *convStateIndices,
                                    const int8_t *hasInitialState) {
    tecoopsStatus_t input_error = checkCausalConv1dInput(
        handle, batch, totalSeqLen, dim, width, x, weight, out);
    if (input_error != TECOOPS_STATUS_SUCCESS)
        return input_error;

    CausalConv1dArgs arg;
    arg.spe_num = handle->spe_num;
    arg.batch = batch;
    arg.totalSeqLen = totalSeqLen;
    arg.dim = dim;
    arg.width = width;
    arg.convStateStride = convStateStride;
    arg.xSeqStride = xSeqStride;
    arg.outSeqStride = outSeqStride;
    arg.padSlotId = padSlotId;
    arg.api_mode = api_mode;
    arg.actMode = actMode;
    arg.x = x;
    arg.convState = convState;
    arg.weight = weight;
    arg.out = out;
    arg.queryStartLoc = queryStartLoc;
    arg.convStateIndices = convStateIndices;
    arg.hasInitialState = hasInitialState;

    CausalConv1dPatchArgs patch_arg;
    patch_arg.atargs = &arg;
    patch_arg.data_type = tecoops::ual::common::UAL_DTYPE_HALF;

    RUN_OP(CausalConv1dOp, arg, patch_arg, handle);
    return TECOOPS_STATUS_SUCCESS;
}
