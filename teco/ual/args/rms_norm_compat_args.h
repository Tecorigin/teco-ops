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

#ifndef TECOOPS_UAL_ARGS_RMS_NORM_COMPAT_ARGS_H_
#define TECOOPS_UAL_ARGS_RMS_NORM_COMPAT_ARGS_H_

#include <cstdint>
#include "ual/com/def.h"

namespace tecoops {
namespace ual {
namespace args {

// rms_norm_compat: reference-exact RMSNorm epilogue (Gemma-4 compatibility
// specialization).
//
// The row-wise statistic (mean of squares + rsqrt) is NOT recomputed here.  The
// reference implementation vllm.ir.ops.rms_norm obtains it from two ATen kernels
// whose results this kernel cannot reproduce bit-for-bit (the device `mean`
// reduction order is a level-2/3 reduction that matches neither a plain
// sequential sum nor the correctly rounded exact mean, and the device `rsqrt` is
// a 1-ULP approximation that is neither 1/sqrt nor pow(-0.5)).  The caller
// therefore passes the per-row `rstd` tensor produced by the reference chain
// itself, and this kernel performs only the part the reference does *after* the
// statistic, in the reference's own rounding order:
//
//   plain:  y = fp16( fp16( fp32(x) * rstd ) * weight )
//   fused:  xr = fp32(x) + fp32(residual)
//           residual_out = fp16(xr)
//           y = fp16( fp16( xr * rstd ) * weight )
//
// The delivered teco_slave_rms_norm_fp16 kernel evaluates the same chain in fp32
// and rounds once; its arithmetic is untouched here (separate kernel entity,
// selected once at op registration).
//
// input:        [num_tokens, hidden_size]  fp16  HBM
// weight:       [hidden_size]              fp16  HBM
// rstd:         [num_tokens]               fp32  HBM
// residual:     [num_tokens, hidden_size]  fp16  HBM, nullptr = none
// output:       [num_tokens, hidden_size]  fp16  HBM
// residual_out: [num_tokens, hidden_size]  fp16  HBM, nullptr unless residual
struct RmsNormCompatArgs {
    const void *input;
    const void *weight;
    const void *rstd;
    const void *residual;
    void *output;
    void *residual_out;
    int num_tokens;
    int hidden_size;
    int has_residual;
};

struct RmsNormCompatPatchArgs {
    RmsNormCompatArgs *atargs;
};

}  // namespace args
}  // namespace ual
}  // namespace tecoops

#endif  // TECOOPS_UAL_ARGS_RMS_NORM_COMPAT_ARGS_H_
