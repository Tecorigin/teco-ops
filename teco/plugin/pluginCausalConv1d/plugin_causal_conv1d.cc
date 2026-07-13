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

#include <plugin/register_op.h>
#include <sdaa_runtime.h>

#include <memory>
#include <string>
#include <vector>
#include <iostream>

#include "interface/include/tecoops.h"

namespace TECO_INFER {

struct plugin_causal_conv1d_fnAttrs
    : public tvm::AttrsNode<plugin_causal_conv1d_fnAttrs> {
  int64_t pad_slot_id;
  int64_t conv_state_stride;
  int64_t x_seq_stride;
  int64_t out_seq_stride;
  int64_t api_mode;
  int64_t act_mode;
  TVM_DECLARE_ATTRS(plugin_causal_conv1d_fnAttrs,
                     "relay.attrs.plugin_causal_conv1d_fnAttrs") {
    TVM_ATTR_FIELD(pad_slot_id).set_default(-1).describe("Padding slot ID");
    TVM_ATTR_FIELD(conv_state_stride).set_default(0)
        .describe("Stride between conv state entries (in elements)");
    TVM_ATTR_FIELD(x_seq_stride).set_default(0)
        .describe("Stride along dim dimension for x");
    TVM_ATTR_FIELD(out_seq_stride).set_default(0)
        .describe("Stride along dim dimension for out");
    TVM_ATTR_FIELD(api_mode).set_default(0).describe("API mode (0=respect hasInitialState)");
    TVM_ATTR_FIELD(act_mode).set_default(0).describe("Activation mode");
  }
};
TVM_REGISTER_NODE_TYPE(plugin_causal_conv1d_fnAttrs);

class PluginCausalConv1dFnImpl : public AbstractPluginOp {
 public:
  PluginCausalConv1dFnImpl() = default;

  void InferOutputShape(
      const std::vector<std::vector<int>>& total_input_shape, int n_input,
      std::vector<std::vector<int>>& total_output_shape, int n_output,
      const OpAttr& attr) {
    // output 0 (out): same shape as input 0 (x)
    total_output_shape[0] = total_input_shape[0];
    // output 1 (convState_out): same shape as input 5 (convState)
    total_output_shape[1] = total_input_shape[5];
  }

  void Enqueue(std::shared_ptr<ComputeContext>& ctx) {
    std::cout << "CALL PluginCausalConv1dFn enqueue" << std::endl;

    void* x_dev = ctx->GetInputDataPtr("x");
    void* weight_dev = ctx->GetInputDataPtr("weight");
    void* query_start_loc_dev = ctx->GetInputDataPtr("queryStartLoc");
    void* cache_indices_dev = ctx->GetInputDataPtr("convStateIndices");
    void* has_init_dev = ctx->GetInputDataPtr("hasInitialState");
    void* conv_state_dev = ctx->GetInputDataPtr("convState");

    void* out_dev = ctx->GetOutputDataPtr(0);
    void* conv_state_out_dev = ctx->GetOutputDataPtr(1);

    int64_t pad_slot_id, conv_state_stride, x_seq_stride, out_seq_stride;
    int64_t api_mode, act_mode;
    ctx->GetAttr("pad_slot_id", pad_slot_id);
    ctx->GetAttr("conv_state_stride", conv_state_stride);
    ctx->GetAttr("x_seq_stride", x_seq_stride);
    ctx->GetAttr("out_seq_stride", out_seq_stride);
    ctx->GetAttr("api_mode", api_mode);
    ctx->GetAttr("act_mode", act_mode);

    std::vector<int> x_shape;
    ctx->GetInputShape("x", x_shape);
    int total_seqlen = x_shape[0];
    int dim = x_shape[1];

    std::vector<int> weight_shape;
    ctx->GetInputShape("weight", weight_shape);
    int width = weight_shape[0];

    std::vector<int> qsl_shape;
    ctx->GetInputShape("queryStartLoc", qsl_shape);
    int batch = qsl_shape[0] - 1;

    std::cout << "PluginCausalConv1dFn: batch=" << batch
              << " totalSeqLen=" << total_seqlen
              << " dim=" << dim << " width=" << width
              << " convStateStride=" << conv_state_stride
              << " padSlotId=" << pad_slot_id << std::endl;

    sdaaStream_t stream = ctx->GetStream();

    tecoopsHandle_t handle;
    tecoopsCreate(&handle);
    tecoopsSetStream(handle, stream);

    // 清零输出 buffer（被 skip 的位置不会被 kernel 写入）
    int64_t outBytes = static_cast<int64_t>(total_seqlen) * dim * 2;
    tecoopsMemset(handle, out_dev, 0, outBytes);

    std::vector<int> conv_state_shape;
    ctx->GetInputShape("convState", conv_state_shape);
    int64_t conv_total_entries = conv_state_shape[0];
    int64_t conv_total_bytes = conv_total_entries * conv_state_shape[1] * conv_state_shape[2] * 2;
    tecoopsMemset(handle, conv_state_out_dev, 0, conv_total_bytes);

    // ConvState 原地更新：C API 读写 input buffer，再将结果拷贝到 output
    tecoopsCausalConv1d(
        handle,
        batch,
        total_seqlen,
        dim,
        width,
        static_cast<int>(conv_state_stride),
        static_cast<int>(x_seq_stride),
        static_cast<int>(out_seq_stride),
        static_cast<int>(pad_slot_id),
        static_cast<int>(api_mode),
        static_cast<int>(act_mode),
        static_cast<const void*>(x_dev),
        conv_state_dev,
        static_cast<const void*>(weight_dev),
        out_dev,
        static_cast<const int*>(query_start_loc_dev),
        static_cast<const int*>(cache_indices_dev),
        static_cast<const int8_t*>(has_init_dev));

    // 将整个 convState 从 input buffer 拷贝到 output buffer
    sdaaMemcpy(conv_state_out_dev, conv_state_dev, conv_total_bytes,
               sdaaMemcpyDeviceToDevice);
  }
};

REGISTER_PLUGIN_OP_IMPL(plugin_causal_conv1d_fn, PluginCausalConv1dFnImpl)

PLUGIN_REGISTER_OP("plugin_causal_conv1d_fn")
    .Input("x")
    .Type("Tensor")
    .Desc("Input tensor [total_seqlen, dim]")
    .Input("weight")
    .Type("Tensor")
    .Desc("Weight tensor [width, dim]")
    .Input("queryStartLoc")
    .Type("Tensor")
    .Desc("Query start locations [batch + 1] int32")
    .Input("convStateIndices")
    .Type("Tensor")
    .Desc("Conv state cache indices [batch] int32")
    .Input("hasInitialState")
    .Type("Tensor")
    .Desc("Whether each batch has an initial state [batch] int8")
    .Input("convState")
    .Type("Tensor")
    .Desc("Conv state cache [num_entries, width-1, dim]")
    .AttrType<plugin_causal_conv1d_fnAttrs>()
    .NumOfOut(2)
    .Register();

}  // namespace TECO_INFER
