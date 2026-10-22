// BSD 3-Clause License
//
// Copyright (c) 2024, Tecorigin Co., Ltd.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "ual/ops/ms_deform_attn_forward/find_ms_deform_attn_forward.h"

namespace tecoops {
namespace ual {
namespace ops {

// Binds the ms_deform_attn_forward implementation once per call, outside the
// kernel (AGENTS.md rule 6: no hot-path selector, no getenv).
//
// Three implementations are bound once per call, in this order of preference:
//
//   slot 4/5  whole-plane staging (zz_ms_deform_attn_forward_plane Tue).  One
//             blocked transfer carries the whole per-batch value plane, so the
//             same value leaf is never re-transferred for a second sampling
//             point or level.  Requires the data independent capacity test
//                 value_len * num_heads * head_dim <= 128
//             because the staged record is T value_record[128].
//   slot 2/3  span-merge (z_ms_deform_attn_forward_span TU).  The span-merge
//             kernel loads every corner block of one (level, point) with ONE
//             blocked transfer into T value_record[128].  The widest such span
//             is 2*width pixels and width <= value_len for every level, so
//                 2 * value_len * num_heads * head_dim <= 128
//             is a conservative, data independent guarantee that the record
//             never overflows (it implies the tighter per-level form
//             2*width*num_heads*head_dim <= 128 that the CPU bit-exact model
//             uses).
//   slot 0/1  the delivered kernel verbatim.
//
// Every predicate is data independent and evaluated outside the kernel, so the
// specialization is fail-closed on the shapes it was validated for and the hot
// path keeps no selector branch and no getenv.
int findMsDeformAttnForwardBranch(
    const args::MsDeformAttnForwardPatchArgs *arg) {
    const args::MsDeformAttnForwardArgs *a = arg->atargs;
    const bool shaped =
        a != nullptr && a->value_len > 0 && a->num_heads > 0 && a->head_dim > 0;
    const bool plane_fits = shaped && 1LL * a->value_len * a->num_heads * a->head_dim <= 128;
    const bool span_fits = shaped && 2LL * a->value_len * a->num_heads * a->head_dim <= 128;
    // Slot 6 is the compacted row-span specialization (fp32 only), for geometries where
    // neither the whole-plane nor the span-merge staging fits.
    const bool rowspan_fits = shaped && a->head_dim <= 128 &&
                              1LL * a->num_heads * a->head_dim + a->head_dim <= 512;
    if (arg->data_type == common::UALDataType::UAL_DTYPE_FLOAT) {
        return plane_fits ? 4 : (span_fits ? 2 : (rowspan_fits ? 6 : 0));
    }
    if (arg->data_type == common::UALDataType::UAL_DTYPE_HALF) {
        return plane_fits ? 5 : (span_fits ? 3 : 1);
    }
    return -1;
}

}  // namespace ops
}  // namespace ual
}  // namespace tecoops
