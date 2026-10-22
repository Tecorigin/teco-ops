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

#include "ual/ops/ms_deform_attn_backward/ms_deform_attn_backward.hpp"
#include "interface/common/convert.h"
#include "interface/common/ms_deform_attn_list_capacity.h"
#include "interface/common/macro.h"
#include "interface/include/builtin_type.h"
#include "interface/include/tecoops.h"

using tecoops::Convert;
using tecoops::ual::args::MsDeformAttnBackwardArgs;
using tecoops::ual::args::MsDeformAttnBackwardPatchArgs;
using tecoops::ual::ops::MsDeformAttnBackwardOp;
using tecoops::ual::ops::MsDeformAttnBackwardCastOp;
using tecoops::ual::ops::MsDeformAttnBackwardZeroOp;

namespace {
// 形状自适应调度阈值。work 定义见下方 List 入口：batch * num_queries * num_heads *
// num_levels * num_points。小工作量时 Average Hardware Time 由 kernel 启动开销主导
// （2026-10-09 实测官方 case_0 = 6.09 us ≈ 一次 ~6.07 us 的 launch 地板），
// 走 2-launch legacy atomic 路径可省去 list_init / list_reduce 两次 launch；
// 大工作量时保持 3-launch list 路径（真实 shape 下减少 launch 数无意义，
// 而历史模型级证据显示 atomic/CAS 变体在真实 shape 反而慢 1.64-3.5x）。
// 官方 case_0 的 work = 1*2*1*1*2 = 4；内部多尺度真实 shape 的 work = 5,689,088。
constexpr int64_t kMsdaAtomicDispatchWorkThreshold = 10000;
}  // namespace

tecoopsStatus_t tecoopsMsDeformAttnBackward(
    tecoopsHandle_t handle,
    const void *value,
    const int64_t *spatial_shapes,
    const void *sampling_locations,
    const void *attention_weights,
    const void *grad_output,
    float *grad_value,
    void *grad_value_fp16,
    void *grad_locations,
    void *grad_weights,
    int batch, int value_len, int num_heads, int head_dim,
    int num_queries, int num_levels, int num_points,
    tecoopsDataType_t data_type, tecoopsAlgo_t algo) {
    if (!handle || !value || !spatial_shapes || !sampling_locations ||
        !attention_weights || !grad_output || !grad_value ||
        !grad_locations || !grad_weights || batch <= 0 || value_len <= 0 ||
        num_heads <= 0 || head_dim <= 0 || head_dim > 128 || num_queries <= 0 ||
        num_levels <= 0 || num_levels > 8 || num_points <= 0 ||
        (data_type != TECOOPS_DATA_FLOAT && data_type != TECOOPS_DATA_HALF) ||
        (data_type == TECOOPS_DATA_HALF &&
         (!grad_value_fp16 || grad_value_fp16 == static_cast<void *>(grad_value)))) {
        return TECOOPS_STATUS_BAD_PARAM;
    }
    MsDeformAttnBackwardArgs arg{};
    arg.spe_num = handle->spe_num;
    arg.value = value;
    arg.spatial_shapes = spatial_shapes;
    arg.sampling_locations = sampling_locations;
    arg.attention_weights = attention_weights;
    arg.grad_output = grad_output;
    arg.grad_value = grad_value;
    arg.grad_value_fp16 = grad_value_fp16;
    arg.grad_locations = grad_locations;
    arg.grad_weights = grad_weights;
    arg.batch = batch;
    arg.value_len = value_len;
    arg.num_heads = num_heads;
    arg.head_dim = head_dim;
    arg.num_queries = num_queries;
    arg.num_levels = num_levels;
    arg.num_points = num_points;
    MsDeformAttnBackwardPatchArgs patch{};
    patch.atargs = &arg;
    patch.data_type = Convert::toUALDataType(data_type);
    patch.algo = Convert::toUALAlgoType(algo);
    // The output-owned specialization is the sole writer of every grad_value
    // element, so the reset launch is only needed by the atomic scatter path.
    if (!tecoops::ual::ops::msDeformAttnBackwardGradValueOwned(&patch)) {
        RUN_OP(MsDeformAttnBackwardZeroOp, arg, patch, handle);
    }
    RUN_OP(MsDeformAttnBackwardOp, arg, patch, handle);
    if (data_type == TECOOPS_DATA_HALF) {
        RUN_OP(MsDeformAttnBackwardCastOp, arg, patch, handle);
    }
    return TECOOPS_STATUS_SUCCESS;
}

tecoopsStatus_t tecoopsMsDeformAttnBackwardList(
    tecoopsHandle_t handle,
    const void *value,
    const int64_t *spatial_shapes,
    const void *sampling_locations,
    const void *attention_weights,
    const void *grad_output,
    float *grad_value,
    void *grad_value_fp16,
    void *grad_locations,
    void *grad_weights,
    int32_t *value_heads, int32_t *value_next, float *node_wx, float *node_wy,
    int batch, int value_len, int num_heads, int head_dim,
    int num_queries, int num_levels, int num_points,
    tecoopsDataType_t data_type, tecoopsAlgo_t algo) {
    if (!handle || !value || !spatial_shapes || !sampling_locations ||
        !attention_weights || !grad_output || !grad_value ||
        !grad_locations || !grad_weights || batch <= 0 || value_len <= 0 ||
        num_heads <= 0 || head_dim <= 0 || head_dim > 128 || num_queries <= 0 ||
        num_levels <= 0 || num_levels > 8 || num_points <= 0 ||
        (data_type != TECOOPS_DATA_FLOAT && data_type != TECOOPS_DATA_HALF) ||
        (data_type == TECOOPS_DATA_HALF &&
         (!grad_value_fp16 || grad_value_fp16 == static_cast<void *>(grad_value)))) {
        return TECOOPS_STATUS_BAD_PARAM;
    }
    int64_t head_count = 0, node_count = 0;
    if (!value_heads || !value_next || !node_wx || !node_wy ||
        !tecoops::msdaListCount({batch, value_len, num_heads}, &head_count) ||
        !tecoops::msdaListCount({4, batch, num_queries, num_heads, num_levels, num_points}, &node_count)) {
        return TECOOPS_STATUS_BAD_PARAM;
    }
    // 形状自适应调度：在入口一次性绑定实现，kernel 热路径不留运行期分支（AGENTS.md 规则 6）。
    const int64_t dispatch_work = static_cast<int64_t>(batch) * num_queries * num_heads *
                                  num_levels * num_points;
    if (dispatch_work < kMsdaAtomicDispatchWorkThreshold) {
        return tecoopsMsDeformAttnBackward(handle, value, spatial_shapes, sampling_locations,
                                          attention_weights, grad_output, grad_value,
                                          grad_value_fp16, grad_locations, grad_weights, batch,
                                          value_len, num_heads, head_dim, num_queries, num_levels,
                                          num_points, data_type, algo);
    }
    MsDeformAttnBackwardArgs arg{};
    arg.spe_num = handle->spe_num;
    arg.value = value;
    arg.spatial_shapes = spatial_shapes;
    arg.sampling_locations = sampling_locations;
    arg.attention_weights = attention_weights;
    arg.grad_output = grad_output;
    arg.grad_value = grad_value;
    arg.grad_value_fp16 = grad_value_fp16;
    arg.grad_locations = grad_locations;
    arg.grad_weights = grad_weights;
    arg.value_heads = value_heads;
    arg.value_next = value_next;
    arg.node_wx = node_wx;
    arg.node_wy = node_wy;
    arg.batch = batch;
    arg.value_len = value_len;
    arg.num_heads = num_heads;
    arg.head_dim = head_dim;
    arg.num_queries = num_queries;
    arg.num_levels = num_levels;
    arg.num_points = num_points;
    MsDeformAttnBackwardPatchArgs patch{};
    patch.atargs = &arg;
    patch.data_type = Convert::toUALDataType(data_type);
    patch.algo = Convert::toUALAlgoType(algo);
    RUN_OP(tecoops::ual::ops::MsDeformAttnBackwardListInitOp, arg, patch, handle);
    RUN_OP(tecoops::ual::ops::MsDeformAttnBackwardListProducerOp, arg, patch, handle);
    RUN_OP(tecoops::ual::ops::MsDeformAttnBackwardListReduceOp, arg, patch, handle);
    if (data_type == TECOOPS_DATA_HALF) {
        RUN_OP(MsDeformAttnBackwardCastOp, arg, patch, handle);
    }
    return TECOOPS_STATUS_SUCCESS;
}
