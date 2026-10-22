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

#include "ual/ops/ms_deform_attn_backward/find_ms_deform_attn_backward.h"

namespace tecoops {
namespace ual {
namespace ops {

namespace {

// The output-owned (gather) specialization scans every sampling point once per
// grad_value element it owns.  One scan is a handful of scalar loads plus two
// floorf - the same order as one sampling point's geometry derivation in the
// 2026-10-10 stage ablation (~0.4 us per derivation with two active SPEs) -
// while the implementation it replaces issues 16 float atomics per sampling
// point, measured at 11.262 us of a 15.331 us kernel on the scored shape with
// two active SPEs and still ~9 us with four.  Keep the gather only while the
// per-SPE scan stays clearly below the cost of those atomics; otherwise the
// original kernel keeps its atomic scatter (and its reset launch).
constexpr int64_t kMsdaGatherMaxScanPairsPerSpe = 32;

}  // namespace

bool msDeformAttnBackwardGradValueOwned(
    const args::MsDeformAttnBackwardPatchArgs *arg) {
    if (arg->data_type != common::UALDataType::UAL_DTYPE_FLOAT) {
        return false;
    }
    const args::MsDeformAttnBackwardArgs *atargs = arg->atargs;
    const int64_t owners =
        static_cast<int64_t>(atargs->batch) * atargs->num_queries * atargs->num_heads;
    if (owners >= atargs->spe_num) {
        return false;  // the owner grain already keeps every SPE busy
    }
    const int64_t samples = owners * atargs->num_levels * atargs->num_points;
    const int64_t outputs = static_cast<int64_t>(atargs->batch) * atargs->value_len *
                            atargs->num_heads * atargs->head_dim;
    const int64_t per_spe = (outputs + atargs->spe_num - 1) / atargs->spe_num;
    return per_spe * samples <= kMsdaGatherMaxScanPairsPerSpe;
}

int findMsDeformAttnBackwardBranch(
    const args::MsDeformAttnBackwardPatchArgs *arg) {
    if (arg->data_type == common::UALDataType::UAL_DTYPE_FLOAT) {
        return msDeformAttnBackwardGradValueOwned(arg) ? 2 : 0;
    }
    if (arg->data_type == common::UALDataType::UAL_DTYPE_HALF) {
        return 1;
    }
    return -1;
}

}  // namespace ops
}  // namespace ual
}  // namespace tecoops
