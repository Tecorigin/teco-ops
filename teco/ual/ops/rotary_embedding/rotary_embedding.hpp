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

#ifndef TECOOPS_UAL_OPS_ROTARY_EMBEDDING_ROTARY_EMBEDDING_HPP_
#define TECOOPS_UAL_OPS_ROTARY_EMBEDDING_ROTARY_EMBEDDING_HPP_

#include "ual/ops/base_op.hpp"
#include "ual/com/log.h"
#include "ual/ops/rotary_embedding/find_rotary_embedding.h"
#include "ual/kernel/rotary_embedding/rotary_embedding.h"

namespace tecoops {
namespace ual {
namespace ops {

using tecoops::ual::args::RotaryEmbeddingArgs;
using tecoops::ual::args::RotaryEmbeddingPatchArgs;
using tecoops::ual::common::Status;

struct RotaryEmbeddingType {
    using ArgsType = RotaryEmbeddingArgs;
    using PatchType = RotaryEmbeddingPatchArgs;
    using RetType = void;
    using PImplType = void (*)(ArgsType);
};

static RotaryEmbeddingType::PImplType RotaryEmbeddingAlgos[] = {
    teco_slave_rotary_embedding_fp16,  // index 0: NeoX (half-split) fp16 rotary embedding
};

static const char *RotaryEmbeddingDiscription[] = {
    "teco_slave_rotary_embedding_fp16",
};

struct RotaryEmbeddingOp : public BaseOp<RotaryEmbeddingOp, RotaryEmbeddingType> {
 public:
    using ArgsType = typename RotaryEmbeddingType::ArgsType;
    using PatchType = typename RotaryEmbeddingType::PatchType;
    using RetType = typename RotaryEmbeddingType::RetType;
    using PImplType = typename RotaryEmbeddingType::PImplType;

    RotaryEmbeddingOp() = default;
    ~RotaryEmbeddingOp() = default;

    static const char *name() { return "rotary_embedding"; }

    Status findImpl(const PatchType *args) {
        int index = findRotaryEmbeddingBranch(args);
        if (index == -1) {
            ERROR("rotary_embedding branch is not exit!");
            return Status::NOT_IMPLEMENTED;
        }
        setInstance(RotaryEmbeddingAlgos[index], RotaryEmbeddingDiscription[index]);
        return Status::SUCCESS;
    }
};

}  // namespace ops
}  // namespace ual
}  // namespace tecoops

#endif  // TECOOPS_UAL_OPS_ROTARY_EMBEDDING_ROTARY_EMBEDDING_HPP_
