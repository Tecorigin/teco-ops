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

#ifndef ZOO_TECO_ROTARY_EMBEDDING_ROTARY_EMBEDDING_H_  // NOLINT
#define ZOO_TECO_ROTARY_EMBEDDING_ROTARY_EMBEDDING_H_

#include "zoo/teco/executor.h"

namespace optest {

// Self-built zoo case executor for tecoopsRotaryEmbedding (task-8).  The cases
// live in test/zoo/teco/rotary_embedding/test_case/ and follow the real
// MiniCPM5-1B geometry: Hq=16, Hkv=2, head_dim=128, NeoX layout, fp16, with the
// fused-QKV row stride of case_2 being the padded one the model actually uses.
class RotaryEmbeddingExecutor : public TecoExecutor {
 public:
    RotaryEmbeddingExecutor() {}
    ~RotaryEmbeddingExecutor() {}

    void paramCheck();
    void paramParse();
    void paramGeneration();
    void compute();
    void cpuCompute();
    int64_t getTheoryOps() override;
    int64_t getTheoryIoSize() override;

 private:
    const void *q_;
    const void *k_;
    const void *cos_sin_cache_;
    const int64_t *positions_;
    void *q_out_;
    void *k_out_;

    int num_tokens_;
    int num_q_heads_;
    int num_kv_heads_;
    int head_dim_;
    int q_stride_;
    int k_stride_;
    int q_out_stride_;
    int k_out_stride_;
    int cache_stride_;
    int max_position_;
    int is_neox_;

    int64_t PositionAt(int t) const { return (static_cast<int64_t>(t) * 7 + 3) % max_position_; }
};

}  // namespace optest

#endif  // ZOO_TECO_ROTARY_EMBEDDING_ROTARY_EMBEDDING_H_  // NOLINT
