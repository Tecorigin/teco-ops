// BSD 3-Clause License
//
// Copyright (c) 2024, Tecorigin Co., Ltd.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
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

#include "ual/ops/ms_deform_attn_forward/ms_deform_attn_forward.hpp"

#include "interface/common/convert.h"
#include "interface/common/macro.h"
#include "interface/include/builtin_type.h"
#include "interface/include/tecoops.h"

using tecoops::Convert;
using tecoops::ual::args::MsDeformAttnForwardArgs;
using tecoops::ual::args::MsDeformAttnForwardPatchArgs;
using tecoops::ual::ops::MsDeformAttnForwardOp;

tecoopsStatus_t tecoopsMsDeformAttnForward(
    tecoopsHandle_t handle,
    const void *value,
    const int64_t *spatial_shapes,
    const void *sampling_locations,
    const void *attention_weights,
    void *output,
    int batch,
    int value_len,
    int num_heads,
    int head_dim,
    int num_queries,
    int num_levels,
    int num_points,
    tecoopsDataType_t data_type,
    tecoopsAlgo_t algo) {
    if (handle == nullptr || value == nullptr || spatial_shapes == nullptr ||
        sampling_locations == nullptr || attention_weights == nullptr || output == nullptr) {
        return TECOOPS_STATUS_BAD_PARAM;
    }
    if (batch <= 0 || value_len <= 0 || num_heads <= 0 || head_dim <= 0 || head_dim > 128 ||
        num_queries <= 0 || num_levels <= 0 || num_levels > 8 || num_points <= 0) {
        return TECOOPS_STATUS_BAD_PARAM;
    }

    MsDeformAttnForwardArgs arg{};
    arg.spe_num = handle->spe_num;
    arg.value = value;
    arg.spatial_shapes = spatial_shapes;
    arg.sampling_locations = sampling_locations;
    arg.attention_weights = attention_weights;
    arg.output = output;
    arg.batch = batch;
    arg.value_len = value_len;
    arg.num_heads = num_heads;
    arg.head_dim = head_dim;
    arg.num_queries = num_queries;
    arg.num_levels = num_levels;
    arg.num_points = num_points;

    MsDeformAttnForwardPatchArgs patch_arg{};
    patch_arg.atargs = &arg;
    patch_arg.data_type = Convert::toUALDataType(data_type);
    patch_arg.algo = Convert::toUALAlgoType(algo);

    RUN_OP(MsDeformAttnForwardOp, arg, patch_arg, handle);
    return TECOOPS_STATUS_SUCCESS;
}
