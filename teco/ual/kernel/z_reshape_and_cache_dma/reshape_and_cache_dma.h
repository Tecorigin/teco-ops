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

// reshape_and_cache 的 SPM + DMA 设备实现。
//
// 这个 TU 的目录名以 'z' 开头，是为了在 `teco/CMakeLists.txt` 的
// `file(GLOB_RECURSE SRC_UAL_KERNEL "${OPS_DIR}/ual/kernel/*/*.scpp")` 里
// **排在字典序最后**（当前最后一个是 weight_stream_gemv）。整个 .so 是一个 LTO
// 链接单元，设备模块的**相对顺序**会改变已交付 kernel 的 codegen —— 把新增设备
// 代码放进一个排在最后的新 TU，可以让之前已交付的 TU 逐字节保持原位
// （gemv 先例：追加在最后的 TU ≈ 0% 代价；把代码长进 reshape_and_cache_fp16.scpp
// 这种排在 rms_norm 之前的 TU 则会让 rms_norm 退 +5%…+12%）。
// 顺序由 verify_rac_dma_arrangement.sh 做 fail-closed 断言。

#ifndef TECO_UAL_KERNEL_Z_RESHAPE_AND_CACHE_DMA_RESHAPE_AND_CACHE_DMA_H_
#define TECO_UAL_KERNEL_Z_RESHAPE_AND_CACHE_DMA_RESHAPE_AND_CACHE_DMA_H_

#include "ual/args/reshape_and_cache_args.h"

using tecoops::ual::args::ReshapeAndCacheArgs;

__global__ void teco_slave_reshape_and_cache_fp16_dma(ReshapeAndCacheArgs args);

#endif  // TECO_UAL_KERNEL_Z_RESHAPE_AND_CACHE_DMA_RESHAPE_AND_CACHE_DMA_H_
