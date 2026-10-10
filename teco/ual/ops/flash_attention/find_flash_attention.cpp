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

// #include "ual/ops/ops_com/discriptor_finder.h"
#include "ual/ops/flash_attention/find_flash_attention.h"
#include "ual/kernel/flash_attention/flash_attention.h"
#include "ual/com/log.h"

namespace tecoops {
namespace ual {
namespace ops {

using tecoops::ual::args::FlashAttentionPatchArgs;


int findFlashAttentionBranch(const FlashAttentionPatchArgs *args) {
    const int head_dim = args->rvargs->size_per_head;
    if (head_dim == 256) {
        return 1;
    }
    if (head_dim == 512) {
        // The D512 entry is flash_attention_half<32>: it tiles K/V with BN == 32 and derives
        // the paged block index as j / block_size, so any block size other than 32 is already
        // outside its contract.  Reject it here (fail closed) instead of silently computing
        // wrong results; findImpl() turns -1 into Status::NOT_IMPLEMENTED.
        if (args->rvargs->block_size != 32) {
            ERROR("flash_attention D512 needs block_size == 32, got %d\n",
                  args->rvargs->block_size);
            return -1;
        }
        return 2;
    }

    // ALGO0 = teco_slave_flash_attention_half -> flash_attention_half<128>.
    // Its Step B (o_accum rescale) and Step E (o_accum accumulate) paths now
    // cover BM == 128 as well, and both advance over the head dimension in
    // 16-float SIMD strides, so head_dim must be a multiple of 16 or the tail
    // columns are silently dropped.  head_dim is a run-time value inside the
    // kernel (`int BK = args.size_per_head`), so this precondition cannot be a
    // static_assert; it is bound once here, at operator dispatch, and -1 makes
    // findImpl() return Status::NOT_IMPLEMENTED (the existing fail-closed
    // convention) instead of silently returning wrong numbers.
    if (head_dim % 16 != 0) {
        ERROR("flash_attention ALGO0 needs size_per_head %% 16 == 0, got %d\n", head_dim);
        return -1;
    }

    int algo = 0;
    // teco_slave_flash_attention_half
    return algo;
}

}  // namespace ops
}  // namespace ual
}  // namespace tecoops
