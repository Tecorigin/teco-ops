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

#ifndef TECOOPS_UAL_OPS_ROTARY_EMBEDDING_FIND_ROTARY_EMBEDDING_H_
#define TECOOPS_UAL_OPS_ROTARY_EMBEDDING_FIND_ROTARY_EMBEDDING_H_

#include "ual/args/rotary_embedding_args.h"

namespace tecoops {
namespace ual {
namespace ops {

using tecoops::ual::args::RotaryEmbeddingPatchArgs;

// Returns the kernel index (0 = teco_slave_rotary_embedding_fp16) or -1 when the
// patch does not describe a supported invocation.  Only the NeoX (half-split)
// layout is implemented: there is exactly one kernel specialization, so nothing
// on the hot path selects between alternatives.
int findRotaryEmbeddingBranch(const RotaryEmbeddingPatchArgs *arg);

}  // namespace ops
}  // namespace ual
}  // namespace tecoops
#endif  // TECOOPS_UAL_OPS_ROTARY_EMBEDDING_FIND_ROTARY_EMBEDDING_H_
