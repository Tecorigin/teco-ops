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

#ifndef TECOOPS_UAL_ARGS_MS_DEFORM_ATTN_FORWARD_ARGS_H_
#define TECOOPS_UAL_ARGS_MS_DEFORM_ATTN_FORWARD_ARGS_H_

#include <cstdint>

#include "ual/com/def.h"

namespace tecoops {
namespace ual {
namespace args {

// Inference-only forward for MultiScaleDeformableAttention.
//
// value:               [N, S, M, D]
// spatial_shapes:      [L, 2] int64, (H_l, W_l)
// sampling_locations:  [N, Lq, M, L, P, 2]
// attention_weights:   [N, Lq, M, L, P]
// output:              [N, Lq, M, D]
//
// The kernel implements the same align_corners=False, zero-padding bilinear
// interpolation as torch.nn.functional.grid_sample.  It is deliberately a
// forward-only UAL primitive: autograd training continues to use the existing
// grid_sample compatibility path in the model adapter.
struct MsDeformAttnForwardArgs {
    int spe_num;
    const void *value;
    const int64_t *spatial_shapes;
    const void *sampling_locations;
    const void *attention_weights;
    void *output;
    int batch;
    int value_len;
    int num_heads;
    int head_dim;
    int num_queries;
    int num_levels;
    int num_points;
};

struct MsDeformAttnForwardPatchArgs {
    MsDeformAttnForwardArgs *atargs;
    common::UALDataType data_type;
    common::UALAlgoType algo;
};

}  // namespace args
}  // namespace ual
}  // namespace tecoops

#endif  // TECOOPS_UAL_ARGS_MS_DEFORM_ATTN_FORWARD_ARGS_H_
