// BSD 3-Clause License
//
// Copyright (c) 2026, Tecorigin Co., Ltd.
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
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
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

#include "ual/ops/rotary_embedding/rotary_embedding.hpp"
#include "interface/common/convert.h"
#include "interface/common/macro.h"
#include "interface/include/builtin_type.h"
#include "interface/include/tecoops.h"
#include "ual/args/rotary_embedding_args.h"

using tecoops::ual::args::RotaryEmbeddingArgs;
using tecoops::ual::args::RotaryEmbeddingPatchArgs;
using tecoops::ual::ops::RotaryEmbeddingOp;

static tecoopsStatus_t checkRotaryEmbeddingInput(
    tecoopsHandle_t handle, const void *q, const void *k, const void *cos_sin_cache,
    const int64_t *positions, void *q_out, void *k_out, int num_tokens, int num_q_heads,
    int num_kv_heads, int head_dim, int is_neox) {
    if (handle == nullptr) {
        return TECOOPS_STATUS_NOT_INITIALIZED;
    }
    if (q == nullptr || k == nullptr || cos_sin_cache == nullptr || positions == nullptr) {
        return TECOOPS_STATUS_BAD_PARAM;
    }
    if (q_out == nullptr || k_out == nullptr) {
        return TECOOPS_STATUS_BAD_PARAM;
    }
    if (num_tokens <= 0 || num_q_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0) {
        return TECOOPS_STATUS_BAD_PARAM;
    }
    if ((head_dim % 2) != 0) {
        return TECOOPS_STATUS_BAD_PARAM;
    }
    // NeoX (half-split) layout only.  A non-NeoX request fails closed instead of
    // silently running the wrong rotation.
    if (is_neox != 1) {
        return TECOOPS_STATUS_NOT_SUPPORTED;
    }
    return TECOOPS_STATUS_SUCCESS;
}

tecoopsStatus_t tecoopsRotaryEmbedding(
    tecoopsHandle_t handle,
    const void *q,
    const void *k,
    const void *cos_sin_cache,
    const int64_t *positions,
    void *q_out,
    void *k_out,
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int q_stride,
    int k_stride,
    int q_out_stride,
    int k_out_stride,
    int cache_stride,
    int is_neox) {

    tecoopsStatus_t input_error = checkRotaryEmbeddingInput(
        handle, q, k, cos_sin_cache, positions, q_out, k_out,
        num_tokens, num_q_heads, num_kv_heads, head_dim, is_neox);
    if (input_error != TECOOPS_STATUS_SUCCESS)
        return input_error;

    const int q_row = num_q_heads * head_dim;
    const int k_row = num_kv_heads * head_dim;
    if (q_stride < q_row || k_stride < k_row ||
        q_out_stride < q_row || k_out_stride < k_row || cache_stride < head_dim) {
        return TECOOPS_STATUS_BAD_PARAM;
    }

    RotaryEmbeddingArgs arg;
    arg.spe_num = handle->spe_num;
    arg.q = q;
    arg.k = k;
    arg.cos_sin_cache = cos_sin_cache;
    arg.positions = positions;
    arg.q_out = q_out;
    arg.k_out = k_out;
    arg.num_tokens = num_tokens;
    arg.num_q_heads = num_q_heads;
    arg.num_kv_heads = num_kv_heads;
    arg.head_dim = head_dim;
    arg.q_stride = q_stride;
    arg.k_stride = k_stride;
    arg.q_out_stride = q_out_stride;
    arg.k_out_stride = k_out_stride;
    arg.cache_stride = cache_stride;

    RotaryEmbeddingPatchArgs patch_arg;
    patch_arg.atargs = &arg;
    patch_arg.data_type = tecoops::ual::common::UAL_DTYPE_HALF;

    RUN_OP(RotaryEmbeddingOp, arg, patch_arg, handle);
    return TECOOPS_STATUS_SUCCESS;
}
