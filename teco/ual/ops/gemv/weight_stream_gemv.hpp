// BSD 3- Clause License Copyright (c) 2024, Tecorigin Co., Ltd. All rights
// reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
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

#ifndef TECOOPS_UAL_OPS_GEMV_WEIGHT_STREAM_GEMV_HPP_
#define TECOOPS_UAL_OPS_GEMV_WEIGHT_STREAM_GEMV_HPP_

#include "ual/ops/base_op.hpp"
#include "ual/com/log.h"
#include "ual/ops/gemv/find_weight_stream_gemv.h"
#include "ual/kernel/weight_stream_gemv/weight_stream_gemv.h"

namespace tecoops {
namespace ual {
namespace ops {

using tecoops::ual::args::WeightStreamGemvArgs;
using tecoops::ual::args::WeightStreamGemvPatchArgs;
using tecoops::ual::common::Status;

struct WeightStreamGemvType {
    using ArgsType = WeightStreamGemvArgs;
    using PatchType = WeightStreamGemvPatchArgs;
    using RetType = void;
    using PImplType = void (*)(ArgsType);
};

static WeightStreamGemvType::PImplType WeightStreamGemvAlgos[] = {
    teco_slave_weight_stream_gemv,  // index 0: M=1 weight-streaming GEMV
};

static const char *WeightStreamGemvDiscription[] = {
    "teco_slave_weight_stream_gemv",
};

struct WeightStreamGemvOp : public BaseOp<WeightStreamGemvOp, WeightStreamGemvType> {
 public:
    using ArgsType = typename WeightStreamGemvType::ArgsType;
    using PatchType = typename WeightStreamGemvType::PatchType;
    using RetType = typename WeightStreamGemvType::RetType;
    using PImplType = typename WeightStreamGemvType::PImplType;

    WeightStreamGemvOp() = default;
    ~WeightStreamGemvOp() = default;

    static const char *name() { return "weight_stream_gemv"; }

    Status findImpl(const PatchType *args) {
        int index = findWeightStreamGemvBranch(args);
        if (index == -1) {
            ERROR("weight_stream_gemv branch is not exit!");
            return Status::NOT_IMPLEMENTED;
        }
        setInstance(WeightStreamGemvAlgos[index], WeightStreamGemvDiscription[index]);
        return Status::SUCCESS;
    }
};

}  // namespace ops
}  // namespace ual
}  // namespace tecoops

#endif  // TECOOPS_UAL_OPS_GEMV_WEIGHT_STREAM_GEMV_HPP_
