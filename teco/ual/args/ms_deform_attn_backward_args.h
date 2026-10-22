// BSD 3-Clause License
//
// Copyright (c) 2024, Tecorigin Co., Ltd.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef TECOOPS_UAL_ARGS_MS_DEFORM_ATTN_BACKWARD_ARGS_H_
#define TECOOPS_UAL_ARGS_MS_DEFORM_ATTN_BACKWARD_ARGS_H_

#include <cstdint>
#include "ual/com/def.h"

namespace tecoops {
namespace ual {
namespace args {

// First-order gradients of the query-major MSDA forward. grad_value is a
// float32 [N,S,M,D] workspace: the paired zero kernel resets it on the same
// stream before accumulation. For half inputs the paired cast kernel writes
// grad_value_fp16 after accumulation; grad_locations/grad_weights have unique
// writers and the input dtype. grad_value_fp16 is unused for float inputs.
struct MsDeformAttnBackwardArgs {
    int spe_num;
    const void *value;
    const int64_t *spatial_shapes;
    const void *sampling_locations;
    const void *attention_weights;
    const void *grad_output;
    float *grad_value;
    void *grad_value_fp16;
    void *grad_locations;
    void *grad_weights;
    // Optional caller-owned List workspace; unused by the original atomic path.
    int32_t *value_heads;
    int32_t *value_next;
    float *node_wx;
    float *node_wy;
    int batch;
    int value_len;
    int num_heads;
    int head_dim;
    int num_queries;
    int num_levels;
    int num_points;
};

struct MsDeformAttnBackwardPatchArgs {
    MsDeformAttnBackwardArgs *atargs;
    common::UALDataType data_type;
    common::UALAlgoType algo;
};

}  // namespace args
}  // namespace ual
}  // namespace tecoops
#endif  // TECOOPS_UAL_ARGS_MS_DEFORM_ATTN_BACKWARD_ARGS_H_
