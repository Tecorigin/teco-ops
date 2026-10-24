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
#include <cstdint>

#include "ual/ops/flash_attention/find_flash_attention.h"
#include "ual/kernel/flash_attention/flash_attention.h"
#include "ual/com/log.h"

namespace tecoops {
namespace ual {
namespace ops {

using tecoops::ual::args::FlashAttentionPatchArgs;


int findFlashAttentionBranch(const FlashAttentionPatchArgs *args) {
    const int head_dim = args->rvargs->size_per_head;
    const int table_dim = args->rvargs->block_table_dim;
    if (head_dim <= 0 || table_dim <= 0) {
        ERROR("flash_attention needs positive head_dim and block_table_dim, got %d/%d\n",
              head_dim, table_dim);
        return -1;
    }

    // ALGO0 strides the head dimension in fixed 32-column
    // tiles: the PV GEMM runs `for (int hb = 0; hb < BK / gemm_bn; hb++)` with `gemm_bn = 32`
    // (BK == head_dim), so a head_dim % 32 != 0 leaves the trailing columns of o_accum never
    // accumulated, and the surrounding 16-lane SIMD rescale/normalize paths walk past the end of
    // the BM*BK buffer.  Measured on the delivered lib: head_dim == 48 yields a device output
    // whose only non-finite elements are NaNs while the CPU reference is finite, and the official
    // DIFF1 harness aborts with TEST_STATUS_RESULT_NAN_ERROR ("Only found result computed on TECO
    // is NaN").  The previous ALGO0-only `% 16` form under-specified this: 48 passes `% 16` and
    // still hits the defect.  head_dim is a run-time value inside the kernel
    // (`int BK = args.size_per_head`), so this precondition cannot be a static_assert; it is bound
    // once here, at operator dispatch, and -1 makes findImpl() return Status::NOT_IMPLEMENTED (the
    // existing fail-closed convention) instead of silently returning NaN.
    if (head_dim % 32 != 0) {
        ERROR("flash_attention needs size_per_head %% 32 == 0, got %d\n", head_dim);
        return -1;
    }

    // Sum ALGO0's SPM allocations with a conservatively 64-byte-aligned block table;
    // all other allocations are already multiples of 64 for 32-aligned head dimensions.
    // Use int64_t before multiplication so a large descriptor cannot wrap the budget check.
    const int64_t table_bytes = (int64_t{table_dim} * sizeof(int) + 63) / 64 * 64;
    const int64_t spm_bytes = int64_t{1152} * head_dim + 27136 + table_bytes;
    // D256/D512 require their separate kernels and algorithm-array entries; do not return
    // those absent entries, or route these dimensions into the oversized ALGO0 allocation.
    if (spm_bytes > 235 * 1024) {
        ERROR("flash_attention ALGO0 exceeds 235 KiB SPM (head_dim=%d, block_table_dim=%d)\n",
              head_dim, table_dim);
        return -1;
    }

    // ALGO0 (teco_slave_flash_attention_half -> flash_attention_half<128>): the kernel hard-codes
    // `int BN = 32;  // N block, 32 for ace; block_size` and derives the paged K/V block index as
    // `int cur_block = j / args.block_size;  // BN == block_size`, so the page stride it addresses
    // is only the tile stride when block_size == 32.  Any other block_size silently reads the
    // wrong pages instead of failing.  This checkout registers only ALGO0; D256/D512 specialized
    // entries must be added together with their kernel declarations and algorithm arrays.
    // Returning -1 keeps the existing
    // fail-closed convention and makes findImpl() return Status::NOT_IMPLEMENTED.
    if (args->rvargs->block_size != 32) {
        ERROR("flash_attention ALGO0 (head_dim=%d) needs block_size == 32, got %d\n",
              head_dim, args->rvargs->block_size);
        return -1;
    }

    int algo = 0;
    // teco_slave_flash_attention_half
    return algo;
}

}  // namespace ops
}  // namespace ual
}  // namespace tecoops
