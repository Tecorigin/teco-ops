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

#include "ual/ops/reshape_and_cache/find_reshape_and_cache.h"

namespace tecoops {
namespace ual {
namespace ops {

using tecoops::ual::args::ReshapeAndCacheArgs;

int findReshapeAndCacheBranch(const ReshapeAndCacheArgs *arg) {
    // Branch 1 is the SPM-staged DMA path.  The DMA descriptor engine needs
    // 64-byte aligned global addresses and 64-byte multiple sections, so the
    // branch is only taken when every derived stride satisfies that; otherwise
    // branch 0 (the reference blocking copy) keeps the operator correct for any
    // geometry.  Dispatch-time decision, never a hot-path branch.
    constexpr int kDmaAlign = 64;
    const int head_bytes = arg->size_per_head * static_cast<int>(sizeof(short));
    const int row_bytes  = arg->num_kv_heads * head_bytes;
    const int strides[5] = {arg->block_stride, arg->head_stride, arg->entry_stride,
                            arg->key_stride, arg->value_stride};
    if (head_bytes <= 0 || row_bytes <= 0) return 0;
    if (head_bytes % kDmaAlign != 0 || row_bytes % kDmaAlign != 0) return 0;
    for (int i = 0; i < 5; ++i) {
        if (strides[i] <= 0 || strides[i] % kDmaAlign != 0) return 0;
    }
    return 1;
}

}  // namespace ops
}  // namespace ual
}  // namespace tecoops
