// BSD 3- Clause License Copyright (c) 2024, Tecorigin Co., Ltd. All rights
// reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
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

#ifndef TECOOPS_UAL_ARGS_WEIGHT_STREAM_GEMV_ARGS_H_
#define TECOOPS_UAL_ARGS_WEIGHT_STREAM_GEMV_ARGS_H_

#include <cstdint>
#include "ual/com/def.h"

namespace tecoops {
namespace ual {
namespace args {

// weight_stream_gemv: M=1 decode weight-streaming GEMV.
//
//   y[n] = sum_k x[k] * w[n][k]        (fp32 accumulation, fp16 operands/result)
//
// Decode-time linear layers in this model run with a single activation row, so
// the whole cost is streaming the [N, K] weight matrix out of HBM.  The vendor
// M=1 GEMM measured 30-78 GB/s effective on the real Gemma-4-12B-it shapes while
// a pure streaming read of the same bytes reaches 92-197 GB/s (measured, see the
// c13-gemma gate probe), so the kernel is organised around contiguous slab DMA
// of whole weight rows rather than around a tiled GEMM.
//
//   x: [K]     fp16  HBM, contiguous
//   w: [N, K]  fp16  HBM, contiguous row-major
//   y: [N]     fp16  HBM, contiguous
struct WeightStreamGemvArgs {
    const void *x;
    const void *w;
    void *y;
    int n;
    int k;
};

struct WeightStreamGemvPatchArgs {
    WeightStreamGemvArgs *atargs;
};

}  // namespace args
}  // namespace ual
}  // namespace tecoops

#endif  // TECOOPS_UAL_ARGS_WEIGHT_STREAM_GEMV_ARGS_H_
