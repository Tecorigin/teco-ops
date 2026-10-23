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

#ifndef TECOOPS_UAL_OPS_RMS_NORM_RMS_NORM_COMPAT_HPP_
#define TECOOPS_UAL_OPS_RMS_NORM_RMS_NORM_COMPAT_HPP_

#include "ual/ops/base_op.hpp"
#include "ual/com/log.h"
#include "ual/ops/rms_norm/find_rms_norm_compat.h"
#include "ual/kernel/rms_norm/rms_norm_compat.h"

namespace tecoops {
namespace ual {
namespace ops {

using tecoops::ual::args::RmsNormCompatArgs;
using tecoops::ual::args::RmsNormCompatPatchArgs;
using tecoops::ual::common::Status;

struct RmsNormCompatType {
    using ArgsType = RmsNormCompatArgs;
    using PatchType = RmsNormCompatPatchArgs;
    using RetType = void;
    using PImplType = void (*)(ArgsType);
};

static RmsNormCompatType::PImplType RmsNormCompatAlgos[] = {
    teco_slave_rms_norm_fp16_compat,      // index 0: reference-exact epilogue
    teco_slave_rms_norm_fp16_compat_add,  // index 1: + reference residual add
};

static const char *RmsNormCompatDiscription[] = {
    "teco_slave_rms_norm_fp16_compat",
    "teco_slave_rms_norm_fp16_compat_add",
};

struct RmsNormCompatOp : public BaseOp<RmsNormCompatOp, RmsNormCompatType> {
 public:
    using ArgsType = typename RmsNormCompatType::ArgsType;
    using PatchType = typename RmsNormCompatType::PatchType;
    using RetType = typename RmsNormCompatType::RetType;
    using PImplType = typename RmsNormCompatType::PImplType;

    RmsNormCompatOp() = default;
    ~RmsNormCompatOp() = default;

    static const char *name() { return "rms_norm_compat"; }

    Status findImpl(const PatchType *args) {
        int index = findRmsNormCompatBranch(args);
        if (index == -1) {
            ERROR("rms_norm_compat branch is not exit!");
            return Status::NOT_IMPLEMENTED;
        }
        setInstance(RmsNormCompatAlgos[index], RmsNormCompatDiscription[index]);
        return Status::SUCCESS;
    }
};

}  // namespace ops
}  // namespace ual
}  // namespace tecoops

#endif  // TECOOPS_UAL_OPS_RMS_NORM_RMS_NORM_COMPAT_HPP_
