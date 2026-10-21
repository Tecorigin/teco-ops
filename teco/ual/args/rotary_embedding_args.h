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

#ifndef TECO_UAL_ARGS_ROTARY_EMBEDDING_ARGS_H_
#define TECO_UAL_ARGS_ROTARY_EMBEDDING_ARGS_H_

#include <cstdint>
#include "ual/com/def.h"

namespace tecoops {
namespace ual {
namespace args {

// rotary_embedding: fused NeoX-style (half-split) rotary position embedding
// applied to q and k in one launch.
//
//   q / q_out        [num_tokens, num_q_heads * head_dim]     fp16
//   k / k_out        [num_tokens, num_kv_heads * head_dim]    fp16
//   cos_sin_cache    [max_position, head_dim]                 fp16
//                    row p = [ cos(p, 0..D/2-1) | sin(p, 0..D/2-1) ]
//   positions        [num_tokens]                             int64
//
// Row addressing uses an explicit stride so that a non-contiguous fused-QKV
// slice (row stride = full qkv width) is handled without materialising a copy.
// q_out may alias q and k_out may alias k; the kernel reads both halves of a
// pair before storing them.
//
// Arithmetic (per token t, head h, d in [0, head_dim/2)); c = cos[t][d],
// s = sin[t][d], x1 = x[t][h][d], x2 = x[t][h][d + head_dim/2]:
//   o1 = fp16( fp32(x1)*fp32(c) ) - fp16( fp32(x2)*fp32(s) )   -> q_out[t][h][d]
//   o2 = fp16( fp32(x2)*fp32(c) ) + fp16( fp32(x1)*fp32(s) )   -> q_out[t][h][d+H]
// Every fp16 op is rounded to fp16 exactly once, matching the fp16 reference
// chain (vLLM RotaryEmbedding.forward_static + ApplyRotaryEmb.forward_static).
struct RotaryEmbeddingArgs {
    int spe_num;                  // slave core count
    const void *q;                // [num_tokens, q_stride] fp16
    const void *k;                // [num_tokens, k_stride] fp16
    const void *cos_sin_cache;    // [max_position, cache_stride] fp16
    const int64_t *positions;     // [num_tokens] int64
    void *q_out;                  // [num_tokens, q_out_stride] fp16
    void *k_out;                  // [num_tokens, k_out_stride] fp16
    int num_tokens;               // T
    int num_q_heads;              // Hq
    int num_kv_heads;             // Hkv
    int head_dim;                 // D, must be even
    int q_stride;                 // row stride of q, in halves
    int k_stride;                 // row stride of k, in halves
    int q_out_stride;             // row stride of q_out, in halves
    int k_out_stride;             // row stride of k_out, in halves
    int cache_stride;             // row stride of cos_sin_cache, in halves
};

struct RotaryEmbeddingPatchArgs {
    RotaryEmbeddingArgs *atargs;
    common::UALDataType data_type;   // UAL_DTYPE_HALF
};

}  // namespace args
}  // namespace ual
}  // namespace tecoops

#endif  // TECO_UAL_ARGS_ROTARY_EMBEDDING_ARGS_H_
