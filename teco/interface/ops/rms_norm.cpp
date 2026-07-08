// BSD 3-Clause License
//
// Copyright (c) 2024, Tecorigin Co., Ltd.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "ual/ops/rms_norm/rms_norm.hpp"
#include "interface/common/convert.h"
#include "interface/common/macro.h"
#include "interface/include/builtin_type.h"
#include "interface/include/tecoops.h"
#include "ual/args/rms_norm_args.h"

using tecoops::ual::args::RmsNormArgs;
using tecoops::ual::args::RmsNormPatchArgs;
using tecoops::ual::ops::RmsNormOp;

static tecoopsStatus_t checkRmsNormInput(tecoopsHandle_t handle,
                                          const void *input, const void *weight,
                                          const void *residual,
                                          void *output, void *residual_out,
                                          int num_tokens, int hidden_size, float eps) {
    if (handle == nullptr) {
        return TECOOPS_STATUS_NOT_INITIALIZED;
    }
    if (input == nullptr || weight == nullptr) {
        return TECOOPS_STATUS_BAD_PARAM;
    }
    if (output == nullptr) {
        return TECOOPS_STATUS_BAD_PARAM;
    }
    if (residual != nullptr && residual_out == nullptr) {
        return TECOOPS_STATUS_BAD_PARAM;
    }
    if (num_tokens <= 0 || hidden_size <= 0) {
        return TECOOPS_STATUS_BAD_PARAM;
    }
    return TECOOPS_STATUS_SUCCESS;
}

tecoopsStatus_t tecoopsRmsNorm(
    tecoopsHandle_t handle,
    const void *input,
    const void *weight,
    const void *residual,
    void *output,
    void *residual_out,
    int num_tokens,
    int hidden_size,
    float eps) {

    tecoopsStatus_t input_error = checkRmsNormInput(
        handle, input, weight, residual, output, residual_out,
        num_tokens, hidden_size, eps);
    if (input_error != TECOOPS_STATUS_SUCCESS)
        return input_error;

    RmsNormArgs arg;
    arg.spe_num = handle->spe_num;
    arg.input = input;
    arg.weight = weight;
    arg.residual = residual;
    arg.output = output;
    arg.residual_out = residual_out;
    arg.num_tokens = num_tokens;
    arg.hidden_size = hidden_size;
    arg.stride = hidden_size;
    arg.eps = eps;
    arg.has_residual = (residual != nullptr) ? 1 : 0;

    RmsNormPatchArgs patch_arg;
    patch_arg.atargs = &arg;
    patch_arg.data_type = tecoops::ual::common::UAL_DTYPE_HALF;

    RUN_OP(RmsNormOp, arg, patch_arg, handle);
    return TECOOPS_STATUS_SUCCESS;
}
