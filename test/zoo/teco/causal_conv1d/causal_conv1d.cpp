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

#include <stdio.h>
#include <cstring>
#include <iostream>
#include <string>
#include "zoo/teco/convert.h"
#include "common/time.hpp"
#include "zoo/teco/causal_conv1d/causal_conv1d.h"
#include "interface/include/tecoops.h"

namespace optest {

void CausalConv1dExecutor::paramCheck() {
    int nin = parser_->inputs().size();
    int nout = parser_->outputs().size();

    // 6 inputs: x, weight, query_start_loc, cache_indices, has_initial_state, convState
    // 2 outputs: out, convState (updated)
    if (nin != 6 || nout != 2) {
        ALLOG(ERROR) << "causal_conv1d requires 6 inputs and 2 outputs, got "
                     << nin << " in, " << nout << " out.";
        throw std::invalid_argument(std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
}

void CausalConv1dExecutor::paramParse() {
    auto meta_x = parser_->input(0);
    auto meta_weight = parser_->input(1);

    // x shape: [total_seqlen, dim]
    totalSeqLen_ = meta_x->shape[0];
    dim_ = meta_x->shape[1];

    // weight shape: [width, dim]
    width_ = meta_weight->shape[0];

    // Parse kernel params from prototxt
    auto causal_param = parser_->getProtoNode()->tecokernel_param().causal_conv1d_param();
    padSlotId_ = causal_param.pad_slot_id();
    api_mode_ = causal_param.api_mode();
    actMode_ = causal_param.act_mode();
    convStateStride_ = causal_param.conv_state_stride();
    xSeqStride_ = causal_param.x_seq_stride();
    outSeqStride_ = causal_param.out_seq_stride();

    // Derive batch from query_start_loc shape
    auto meta_qsl = parser_->input(2);
    batch_ = meta_qsl->shape[0] - 1;
}

void CausalConv1dExecutor::paramGeneration() {
    x_ = dev_input[0];
    weight_ = dev_input[1];
    queryStartLoc_ = static_cast<const int *>(dev_input[2]);
    convStateIndices_ = static_cast<const int *>(dev_input[3]);
    hasInitialState_ = static_cast<const int8_t *>(dev_input[4]);

    // dev_output[0] = out, dev_output[1] = updated convState
    out_ = dev_output[0];
    convState_ = dev_output[1];

    // convState is read-write: copy initial state from input to output buffer
    int64_t convStateBytes =
        static_cast<int64_t>(batch_) * (width_ - 1) * dim_ * sizeof(uint16_t);
    checkScdaErrors(scdaMemcpy(convState_, dev_input[5], convStateBytes, MemcpyDeviceToDevice));
}

void CausalConv1dExecutor::compute() {
    checkTECOOPS(tecoopsCausalConv1d(
        handle_,
        batch_,
        totalSeqLen_,
        dim_,
        width_,
        convStateStride_,
        xSeqStride_,
        outSeqStride_,
        padSlotId_,
        api_mode_,
        actMode_,
        x_,
        convState_,
        weight_,
        out_,
        queryStartLoc_,
        convStateIndices_,
        hasInitialState_));
}

void CausalConv1dExecutor::cpuCompute() {
    pythonComputeCPU("cpu");
}

int64_t CausalConv1dExecutor::getTheoryOps() {
    int64_t ops = static_cast<int64_t>(totalSeqLen_) * dim_ * 20;
    return ops;
}

int64_t CausalConv1dExecutor::getTheoryIoSize() {
    constexpr int kHalfSize = 2;
    int64_t totalTokens = static_cast<int64_t>(totalSeqLen_);
    int64_t size = 0;
    // read: x
    size += static_cast<int64_t>(totalTokens) * dim_ * kHalfSize;
    // read: weight
    size += static_cast<int64_t>(width_) * dim_ * kHalfSize;
    // read/write: convState (batch * (width-1) * dim)
    size += static_cast<int64_t>(batch_) * (width_ - 1) * dim_ * kHalfSize * 2;
    // write: out
    size += static_cast<int64_t>(totalTokens) * dim_ * kHalfSize;
    // read: queryStartLoc, cacheIndices, hasInitialState (negligible)
    return size;
}

}  // namespace optest
