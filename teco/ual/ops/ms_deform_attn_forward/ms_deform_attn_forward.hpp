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

#ifndef TECOOPS_UAL_OPS_MS_DEFORM_ATTN_FORWARD_H_
#define TECOOPS_UAL_OPS_MS_DEFORM_ATTN_FORWARD_H_

#include "ual/kernel/ms_deform_attn_forward/ms_deform_attn_forward.h"
#include "ual/ops/base_op.hpp"
#include "ual/ops/ms_deform_attn_forward/find_ms_deform_attn_forward.h"

namespace tecoops {
namespace ual {
namespace ops {

using args::MsDeformAttnForwardArgs;
using args::MsDeformAttnForwardPatchArgs;

struct MsDeformAttnForwardType {
    using ArgsType = MsDeformAttnForwardArgs;
    using PatchType = MsDeformAttnForwardPatchArgs;
    using RetType = void;
    using PImplType = void (*)(ArgsType);
};

static MsDeformAttnForwardType::PImplType MsDeformAttnForwardAlgos[] = {
    teco_slave_ms_deform_attn_forward_fp32,
    teco_slave_ms_deform_attn_forward_fp16,
};

static const char *MsDeformAttnForwardDescriptions[] = {
    "teco_slave_ms_deform_attn_forward_fp32",
    "teco_slave_ms_deform_attn_forward_fp16",
};

struct MsDeformAttnForwardOp
    : public BaseOp<MsDeformAttnForwardOp, MsDeformAttnForwardType> {
    static const char *name() { return "ms_deform_attn_forward"; }

    common::Status findImpl(const MsDeformAttnForwardPatchArgs *args) {
        int index = findMsDeformAttnForwardBranch(args);
        if (index < 0) {
            ERROR("ms_deform_attn_forward dtype is not supported!");
            return common::Status::NOT_IMPLEMENTED;
        }
        setInstance(MsDeformAttnForwardAlgos[index],
                    MsDeformAttnForwardDescriptions[index]);
        return common::Status::SUCCESS;
    }
};

}  // namespace ops
}  // namespace ual
}  // namespace tecoops

#endif  // TECOOPS_UAL_OPS_MS_DEFORM_ATTN_FORWARD_H_
