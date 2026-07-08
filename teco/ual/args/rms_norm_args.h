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

#ifndef TECO_UAL_ARGS_RMS_NORM_ARGS_H_
#define TECO_UAL_ARGS_RMS_NORM_ARGS_H_

#include <cstdint>
#include "ual/com/def.h"

namespace tecoops {
namespace ual {
namespace args {

// rms_norm: RMS Normalization + optional residual add (fused)
//
// 公式 (RMS mode):
//   x' = has_residual ? (input + residual) : input
//   rms = sqrt(mean(x'^2) + eps)
//   output = x' / rms * weight
//   residual_out = x' (if has_residual)
//
// input:       [num_tokens, hidden_size]       fp16  HBM
// weight:      [hidden_size]                   fp16  HBM
// residual:    [num_tokens, hidden_size]       fp16  HBM, nullptr=无
// output:      [num_tokens, hidden_size]       fp16  HBM
// residual_out:[num_tokens, hidden_size]       fp16  HBM
struct RmsNormArgs {
    int spe_num;            // slave core 数量
    const void *input;      // [num_tokens, stride]  输入 hidden states
    const void *weight;     // [hidden_size]          RMSNorm 权重 gamma
    const void *residual;   // [num_tokens, hidden_size] 残差，可为 nullptr
    void *output;           // [num_tokens, hidden_size] norm 后输出
    void *residual_out;     // [num_tokens, hidden_size] skip connection 输出
    int num_tokens;         // batch size N
    int hidden_size;        // 隐藏维度 D
    int stride;             // 输入行步长（>= hidden_size）
    float eps;              // epsilon
    int has_residual;       // 是否有 residual (0/1)
};

struct RmsNormPatchArgs {
    RmsNormArgs *atargs;
    common::UALDataType data_type;   // UAL_DTYPE_HALF
};

}  // namespace args
}  // namespace ual
}  // namespace tecoops

#endif  // TECO_UAL_ARGS_RMS_NORM_ARGS_H_
