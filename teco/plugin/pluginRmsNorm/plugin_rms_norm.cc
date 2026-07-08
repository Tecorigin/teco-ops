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

#include "interface/include/tecoops.h"

namespace TECO_INFER {

struct plugin_rms_normAttrs
    : public tvm::AttrsNode<plugin_rms_normAttrs> {
  double eps;
  TVM_DECLARE_ATTRS(plugin_rms_normAttrs,
                     "relay.attrs.plugin_rms_normAttrs") {
    TVM_ATTR_FIELD(eps).set_default(1e-5).describe("Epsilon for numerical stability");
  }
};
TVM_REGISTER_NODE_TYPE(plugin_rms_normAttrs);

struct plugin_rms_norm_addAttrs
    : public tvm::AttrsNode<plugin_rms_norm_addAttrs> {
  double eps;
  TVM_DECLARE_ATTRS(plugin_rms_norm_addAttrs,
                     "relay.attrs.plugin_rms_norm_addAttrs") {
    TVM_ATTR_FIELD(eps).set_default(1e-5).describe("Epsilon for numerical stability");
  }
};
TVM_REGISTER_NODE_TYPE(plugin_rms_norm_addAttrs);

// ================================================================
// plugin_rms_norm: 纯 RMSNorm (input, weight -> output)
// ================================================================
class PluginRmsNormImpl : public AbstractPluginOp {
 public:
  PluginRmsNormImpl() = default;

  void InferOutputShape(
      const std::vector<std::vector<int>>& total_input_shape, int n_input,
      std::vector<std::vector<int>>& total_output_shape, int n_output,
      const OpAttr& attr) {
    total_output_shape[0] = total_input_shape[0];
  }

  void Enqueue(std::shared_ptr<ComputeContext>& ctx) {
    void* input_dev   = ctx->GetInputDataPtr(0);
    void* weight_dev  = ctx->GetInputDataPtr(1);
    void* output_dev  = ctx->GetOutputDataPtr(0);

    float eps;
    ctx->GetAttr("eps", eps);

    std::vector<int> input_shape;
    ctx->GetInputShape(0, input_shape);
    int num_tokens  = input_shape[0];
    int hidden_size = input_shape[1];

    sdaaStream_t stream = ctx->GetStream();

    tecoopsHandle_t handle;
    tecoopsCreate(&handle);
    tecoopsSetStream(handle, stream);

    tecoopsRmsNorm(handle,
                   input_dev, weight_dev,
                   nullptr, output_dev, nullptr,
                   num_tokens, hidden_size, static_cast<float>(eps));

    tecoopsDestroy(handle);
  }
};

REGISTER_PLUGIN_OP_IMPL(plugin_rms_norm, PluginRmsNormImpl)

PLUGIN_REGISTER_OP("plugin_rms_norm")
    .Input("input")
    .Type("Tensor")
    .Desc("Input tensor [num_tokens, hidden_size]")
    .Input("weight")
    .Type("Tensor")
    .Desc("Weight tensor [hidden_size]")
    .AttrType<plugin_rms_normAttrs>()
    .Register();

// ================================================================
// plugin_rms_norm_add: RMSNorm + residual add
//   (input, weight, residual -> output, residual_out)
// ================================================================
class PluginRmsNormAddImpl : public AbstractPluginOp {
 public:
  PluginRmsNormAddImpl() = default;

  void InferOutputShape(
      const std::vector<std::vector<int>>& total_input_shape, int n_input,
      std::vector<std::vector<int>>& total_output_shape, int n_output,
      const OpAttr& attr) {
    total_output_shape[0] = total_input_shape[0];
    total_output_shape[1] = total_input_shape[0];
  }

  void Enqueue(std::shared_ptr<ComputeContext>& ctx) {
    void* input_dev    = ctx->GetInputDataPtr(0);
    void* weight_dev   = ctx->GetInputDataPtr(1);
    void* residual_dev = ctx->GetInputDataPtr(2);
    void* output_dev   = ctx->GetOutputDataPtr(0);
    void* res_out_dev  = ctx->GetOutputDataPtr(1);

    float eps;
    ctx->GetAttr("eps", eps);

    std::vector<int> input_shape;
    ctx->GetInputShape(0, input_shape);
    int num_tokens  = input_shape[0];
    int hidden_size = input_shape[1];

    sdaaStream_t stream = ctx->GetStream();

    tecoopsHandle_t handle;
    tecoopsCreate(&handle);
    tecoopsSetStream(handle, stream);

    tecoopsRmsNorm(handle,
                   input_dev, weight_dev,
                   residual_dev, output_dev, res_out_dev,
                   num_tokens, hidden_size, static_cast<float>(eps));

    tecoopsDestroy(handle);
  }
};

REGISTER_PLUGIN_OP_IMPL(plugin_rms_norm_add, PluginRmsNormAddImpl)

namespace {
PLUGIN_REGISTER_OP("plugin_rms_norm_add")
    .Input("input")
    .Type("Tensor")
    .Desc("Input tensor [num_tokens, hidden_size]")
    .Input("weight")
    .Type("Tensor")
    .Desc("Weight tensor [hidden_size]")
    .Input("residual")
    .Type("Tensor")
    .Desc("Residual tensor [num_tokens, hidden_size]")
    .AttrType<plugin_rms_norm_addAttrs>()
    .NumOfOut(2)
    .Register();
}  // namespace

}  // namespace TECO_INFER
