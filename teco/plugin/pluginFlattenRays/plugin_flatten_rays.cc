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

struct plugin_flatten_raysAttrs
    : public tvm::AttrsNode<plugin_flatten_raysAttrs> {
  int64_t M;
  TVM_DECLARE_ATTRS(plugin_flatten_raysAttrs,
                     "relay.attrs.plugin_flatten_raysAttrs") {
    TVM_ATTR_FIELD(M).set_default(0).describe("Total number of steps");
  }
};
TVM_REGISTER_NODE_TYPE(plugin_flatten_raysAttrs);

class PluginFlattenRaysImpl : public AbstractPluginOp {
 public:
  PluginFlattenRaysImpl() = default;

  void InferOutputShape(
      const std::vector<std::vector<int>>& total_input_shape, int n_input,
      std::vector<std::vector<int>>& total_output_shape, int n_output,
      const OpAttr& attr) {
    int64_t M = attr.GetAttr<int64_t>("M");
    std::cout << "M = " << M << std::endl;
    total_output_shape[0] = {static_cast<int>(M)};
  }

  void Enqueue(std::shared_ptr<ComputeContext>& ctx) {
    std::cout << "CALL PluginFlattenRays enqueue" << std::endl;
    void* rays_dev = ctx->GetInputDataPtr("rays");
    void* res_dev = ctx->GetOutputDataPtr(0);

    int64_t M;
    ctx->GetAttr("M", M);

    std::vector<int> rays_shape;
    ctx->GetInputShape("rays", rays_shape);
    uint32_t N = rays_shape[0];

    sdaaStream_t stream = ctx->GetStream();

    tecoopsHandle_t handle;
    tecoopsCreate(&handle);
    tecoopsSetStream(handle, stream);

    tecoopsFlattenRays(handle,
                       static_cast<const int*>(rays_dev),
                       N,
                       static_cast<uint32_t>(M),
                       static_cast<int*>(res_dev),
                       TECOOPS_ALGO_0);

    tecoopsDestroy(handle);
  }
};

REGISTER_PLUGIN_OP_IMPL(plugin_flatten_rays, PluginFlattenRaysImpl)

PLUGIN_REGISTER_OP("plugin_flatten_rays")
    .Input("rays")
    .Type("Tensor")
    .Desc("Rays input with shape [N, 2]")
    .AttrType<plugin_flatten_raysAttrs>()
    .Register();

}  // namespace TECO_INFER
