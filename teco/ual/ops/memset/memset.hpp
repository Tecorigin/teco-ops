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

#ifndef TECOOPS_UAL_OPS_MEMSET_MEMSET_HPP_
#define TECOOPS_UAL_OPS_MEMSET_MEMSET_HPP_

#include "ual/ops/base_op.hpp"
#include "ual/com/log.h"
#include "ual/ops/memset/find_memset.h"
#include "ual/kernel/memset/memset.h" 


namespace tecoops {
namespace ual {
namespace ops {

using tecoops::ual::args::MemsetArgs;
using tecoops::ual::common::Status;

struct MemsetType {
    using ArgsType = MemsetArgs;      // using implement kernel args
    using PatchType = MemsetArgs;  // using dispatch args
    using RetType = void;
    using PImplType = void (*)(ArgsType);
};

static MemsetType::PImplType MemsetAlgos[] = {
    teco_slave_memset_4B_align,
};

// algo name
static const char *MemsetDiscription[] = {
    "teco_slave_memset_4B_align",
};
struct MemsetOp : public BaseOp<MemsetOp, MemsetType> {
 public:
    using ArgsType = typename MemsetType::ArgsType;
    using PatchType = typename MemsetType::PatchType;
    using RetType = typename MemsetType::RetType;
    using PImplType = typename MemsetType::PImplType;

    MemsetOp() = default;
    ~MemsetOp() = default;

    static const char *name() { return "memset"; }

    Status findImpl(const PatchType *args) {
        int index = findMemsetBranch(args);
        if (index == -1) {
            ERROR("memset branch is not exit!");
            return Status::NOT_IMPLEMENTED;
        }
        setInstance(MemsetAlgos[index], MemsetDiscription[index]);
        return Status::SUCCESS;
    }
};

}  // namespace ops
}  // namespace ual
}  // namespace tecoops

#endif  // TECOOPS_UAL_OPS_MEMSET_MEMSET_HPP_
