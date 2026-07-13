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

#ifndef TECO_UAL_ARGS_CAUSAL_CONV1D_ARGS_H_
#define TECO_UAL_ARGS_CAUSAL_CONV1D_ARGS_H_

#include <cstdint>
#include "ual/com/def.h"

namespace tecoops {
namespace ual {
namespace args {

// causal_conv1d: 1D causal convolution with silu activation
//
// x:               [total_seqlen, dim]             输入序列
// convState:       [batch, width-1, dim]           conv 状态（读写）
// weight:          [width, dim]                    卷积权重
// out:             [total_seqlen, dim]             输出序列
// queryStartLoc:   [batch+1]                       query 起始位置索引
// convStateIndices:[batch]                         conv state 索引
// hasInitialState: [batch]                         是否有初始状态标记
struct CausalConv1dArgs {
    int spe_num;
    int batch;
    int totalSeqLen;
    int dim;
    int width;
    int convStateStride;
    int xSeqStride;
    int outSeqStride;
    int padSlotId;
    int api_mode;
    int actMode;

    const void *x;                          // [total_seqlen, dim]
    void *convState;                        // [-1, width-1, dim]
    const void *weight;                     // [width, dim]
    void *out;                              // [total_seqlen, dim]

    const int *queryStartLoc;               // [batch+1]
    const int *convStateIndices;            // [batch]
    const int8_t *hasInitialState;          // [batch]
};

struct CausalConv1dPatchArgs {
    CausalConv1dArgs *atargs;
    common::UALDataType data_type;
};

}  // namespace args
}  // namespace ual
}  // namespace tecoops

#endif  // TECO_UAL_ARGS_CAUSAL_CONV1D_ARGS_H_
