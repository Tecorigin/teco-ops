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

#ifndef TECOOPS_UAL_OPS_FLATTEN_RAYS_FLATTEN_RAYS_HPP_
#define TECOOPS_UAL_OPS_FLATTEN_RAYS_FLATTEN_RAYS_HPP_

#include "ual/kernel/flatten_rays/flatten_rays.h"
#include "ual/com/log.h"
#include "ual/args/flatten_rays_args.h"
#include "ual/com/def.h"
#include "ual/ops/base_op.hpp"
#include "ual/ops/flatten_rays/find_flatten_rays.h"

using tecoops::ual::args::FlattenRaysArgs;
using tecoops::ual::args::FlattenRaysPatchArgs;
using namespace tecoops::ual::common;

namespace tecoops {
namespace ual {
namespace ops {

struct FlattenRaysType {
    using ArgsType = FlattenRaysArgs;
    using PatchType = FlattenRaysPatchArgs;
    using RetType = void;
    using PImplType = void (*)(ArgsType);
};

static FlattenRaysType::PImplType FlattenRaysAlgos[] = {
    tecoKernelFlattenRaysInt,
};

static const char *FlattenRaysDiscription[] = {
    "tecoKernelFlattenRaysInt",
};

struct FlattenRaysOp : public BaseOp<FlattenRaysOp, FlattenRaysType> {
 public:
    using ArgsType = typename FlattenRaysType::ArgsType;
    using PatchType = typename FlattenRaysType::PatchType;
    using RetType = typename FlattenRaysType::RetType;
    using PImplType = typename FlattenRaysType::PImplType;

    static const char *name() { return "flatten_rays"; }

    common::Status findImpl(const PatchType *args) {
        int index = findFlattenRaysBranch(args);
        if (index == -1) {
            ERROR("flatten_rays branch is not exit!");
            return common::Status::NOT_IMPLEMENTED;
        }
        setInstance(FlattenRaysAlgos[index], FlattenRaysDiscription[index]);
        return common::Status::SUCCESS;
    }
};

}  // namespace ops
}  // namespace ual
}  // namespace tecoops

#endif  // TECOOPS_UAL_OPS_FLATTEN_RAYS_FLATTEN_RAYS_HPP_