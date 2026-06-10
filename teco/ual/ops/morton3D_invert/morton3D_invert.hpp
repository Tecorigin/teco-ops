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

#ifndef TECOOPS_UAL_OPS_MORTON3D_INVERT_HPP_
#define TECOOPS_UAL_OPS_MORTON3D_INVERT_HPP_

#include "ual/kernel/morton3D_invert/morton3D_invert.h"
#include "ual/com/log.h"
#include "ual/args/morton3D_invert_args.h"
#include "ual/com/def.h"
#include "ual/ops/base_op.hpp"
#include "ual/ops/morton3D_invert/find_morton3D_invert.h"

using tecoops::ual::args::Morton3DInvertArgs;
using tecoops::ual::args::Morton3DInvertPatchArgs;
using namespace tecoops::ual::common;

namespace tecoops {
namespace ual {
namespace ops {

struct Morton3DInvertType {
    using ArgsType = Morton3DInvertArgs;
    using PatchType = Morton3DInvertPatchArgs;
    using RetType = void;
    using PImplType = void (*)(ArgsType);
};

static Morton3DInvertType::PImplType Morton3DInvertAlgos[] = {
    tecoKernelMorton3DInvertInt,
};

static const char *Morton3DInvertDiscription[] = {
    "tecoKernelMorton3DInvertInt",
};

struct Morton3DInvertOp : public BaseOp<Morton3DInvertOp, Morton3DInvertType> {
 public:
    using ArgsType = typename Morton3DInvertType::ArgsType;
    using PatchType = typename Morton3DInvertType::PatchType;
    using RetType = typename Morton3DInvertType::RetType;
    using PImplType = typename Morton3DInvertType::PImplType;

    static const char *name() { return "morton3d_invert"; }

    common::Status findImpl(const PatchType *args) {
        int index = findMorton3DInvertBranch(args);
        if (index == -1) {
            ERROR("morton3d_invert branch is not exit!");
            return common::Status::NOT_IMPLEMENTED;
        }
        setInstance(Morton3DInvertAlgos[index], Morton3DInvertDiscription[index]);
        return common::Status::SUCCESS;
    }
};

}  // namespace ops
}  // namespace ual
}  // namespace tecoops

#endif  // TECOOPS_UAL_OPS_MORTON3D_INVERT_HPP_