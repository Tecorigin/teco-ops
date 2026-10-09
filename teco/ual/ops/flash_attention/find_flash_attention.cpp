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
#include "ual/kernel/flash_attention/flash_attention_prefill.h"
#include "ual/com/log.h"

namespace tecoops {
namespace ual {
namespace ops {

using tecoops::ual::args::FlashAttentionPatchArgs;


int findFlashAttentionBranch(const FlashAttentionPatchArgs *args) {
    // algo 00: teco_slave_flash_attention_half (reference, any layout)
    // algo 01: teco_slave_flash_attention_prefill (gemm32-fused P1+P2 variant)
    //          support domain: fp16, size_per_head == 128, block_size == 32
    //          (M128_N32 tiling; BN must equal the physical cache page size).
    //          Outside this domain fall back to the reference kernel.
    int algo = 0;
    if (args != nullptr && args->rvargs != nullptr) {
        const bool half_type = (args->data_type == tecoops::ual::common::UAL_DTYPE_HALF);
        if (half_type && args->rvargs->size_per_head == 128 && args->rvargs->block_size == 32) {
            algo = 1;  // teco_slave_flash_attention_prefill
        }
    }
    return algo;
}

}  // namespace ops
}  // namespace ual
}  // namespace tecoops
