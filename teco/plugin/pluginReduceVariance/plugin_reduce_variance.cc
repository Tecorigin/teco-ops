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

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "interface/include/tecoops.h"

namespace TECO_INFER {

struct plugin_reduce_varianceAttrs
    : public tvm::AttrsNode<plugin_reduce_varianceAttrs> {
  int64_t axis;
  int64_t correction;

  TVM_DECLARE_ATTRS(plugin_reduce_varianceAttrs,
                    "relay.attrs.plugin_reduce_varianceAttrs") {
    TVM_ATTR_FIELD(axis).set_default(0).describe("Reduction axis");
    TVM_ATTR_FIELD(correction).set_default(0).describe("Correction, 0 or 1");
  }
};

TVM_REGISTER_NODE_TYPE(plugin_reduce_varianceAttrs);

static int NormalizeAxis(int64_t axis, int dims) {
  if (dims <= 0) {
    throw std::runtime_error("plugin_reduce_variance: input dims must be positive");
  }

  if (axis < 0) {
    axis += dims;
  }

  if (axis < 0 || axis >= dims) {
    throw std::runtime_error("plugin_reduce_variance: axis out of range");
  }

  return static_cast<int>(axis);
}

static std::vector<int> MakeOutputShapeKeepDim(const std::vector<int>& input_shape,
                                               int axis) {
  std::vector<int> output_shape = input_shape;
  output_shape[axis] = 1;
  return output_shape;
}

class PluginReduceVarianceImpl : public AbstractPluginOp {
 public:
  PluginReduceVarianceImpl() = default;

  void InferOutputShape(
      const std::vector<std::vector<int>>& total_input_shape, int n_input,
      std::vector<std::vector<int>>& total_output_shape, int n_output,
      const OpAttr& attr) {
    int64_t axis_attr = attr.GetAttr<int64_t>("axis");

    const std::vector<int>& x_shape = total_input_shape[0];
    int axis = NormalizeAxis(axis_attr, static_cast<int>(x_shape.size()));

    total_output_shape[0] = MakeOutputShapeKeepDim(x_shape, axis);

  }

  void Enqueue(std::shared_ptr<ComputeContext>& ctx) {
    std::cout << "CALL PluginReduceVariance enqueue" << std::endl;

    void* x_dev = ctx->GetInputDataPtr("x");
    void* y_dev = ctx->GetOutputDataPtr(0);

    int64_t axis_attr;
    int64_t correction_attr;
    ctx->GetAttr("axis", axis_attr);
    ctx->GetAttr("correction", correction_attr);

    if (correction_attr != 0 && correction_attr != 1) {
      throw std::runtime_error("plugin_reduce_variance: correction must be 0 or 1");
    }

    std::vector<int> x_shape;
    ctx->GetInputShape("x", x_shape);

    int dims = static_cast<int>(x_shape.size());
    int axis = NormalizeAxis(axis_attr, dims);
    std::vector<int> y_shape = MakeOutputShapeKeepDim(x_shape, axis);

    sdaaStream_t stream = ctx->GetStream();

    tecoopsHandle_t handle;
    tecoopsCreate(&handle);
    tecoopsSetStream(handle, stream);

    tecoopsTensorDescriptor_t x_desc;
    tecoopsTensorDescriptor_t y_desc;
    tecoopsCreateTensorDescriptor(&x_desc);
    tecoopsCreateTensorDescriptor(&y_desc);

    tecoopsSetTensorNdDescriptor(
        x_desc,
        TECOOPS_DATA_FLOAT,
        dims,
        x_shape.data(),
        nullptr);

    tecoopsSetTensorNdDescriptor(
        y_desc,
        TECOOPS_DATA_FLOAT,
        dims,
        y_shape.data(),
        nullptr);

    tecoopsStatus_t status = tecoopsReduceVariance(
        handle,
        axis,
        static_cast<int>(correction_attr),
        x_desc,
        x_dev,
        y_desc,
        y_dev);

    if (status != TECOOPS_STATUS_SUCCESS) {
      std::cerr << "tecoopsReduceVariance failed: "
                << tecoopsGetErrorString(status) << std::endl;
    }

    tecoopsDestroyTensorDescriptor(x_desc);
    tecoopsDestroyTensorDescriptor(y_desc);
    tecoopsDestroy(handle);
  }
};

REGISTER_PLUGIN_OP_IMPL(plugin_reduce_variance, PluginReduceVarianceImpl)

PLUGIN_REGISTER_OP("plugin_reduce_variance")
    .Input("x")
    .Type("Tensor")
    .Desc("Input tensor")
    .AttrType<plugin_reduce_varianceAttrs>()
    .Register();

}  // namespace TECO_INFER
