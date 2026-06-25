# BSD 3- Clause License Copyright (c) 2024, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.

import os
import ctypes
import onnx
from onnx import helper, TensorProto
import tvm
from tvm.plugin import plugins
import tecoinference
from tvm.contrib.teco_infer_dyn import dyn
import numpy as np


def output_shape_keepdim(input_shape, axis):
    axis = axis if axis >= 0 else axis + len(input_shape)
    out_shape = list(input_shape)
    out_shape[axis] = 1
    return tuple(out_shape)


def create_plugin_reduce_variance_onnx_model(input_shape, axis, correction):
    input_tensor = helper.make_tensor_value_info(
        "x",
        TensorProto.FLOAT,
        input_shape,
    )

    output_shape = output_shape_keepdim(input_shape, axis)
    output_tensor = helper.make_tensor_value_info(
        "output",
        TensorProto.FLOAT,
        output_shape,
    )

    node = helper.make_node(
        "plugin_reduce_variance",
        inputs=["x"],
        outputs=["output"],
        axis=axis,
        correction=correction,
        domain="my_custom_ops",
        version=1,
    )

    graph = helper.make_graph(
        [node],
        "plugin_reduce_variance_graph",
        [input_tensor],
        [output_tensor],
    )

    model = helper.make_model(graph)
    model.opset_import.append(helper.make_opsetid("my_custom_ops", 1))
    return model


plugins.register_op(
    op_name="plugin_reduce_variance",
    inputs=["x"],
    attrs={
        "axis": "int",
        "correction": "int",
    },
)


def run_case(input_shape, axis, correction):
    print("=" * 60)
    print(f"input_shape={input_shape}, axis={axis}, correction={correction}")

    np.random.seed(42)
    x_data = np.random.randn(*input_shape).astype("float32")

    print("-"*60)

    print(x_data)

    print("-"*60)
    model = create_plugin_reduce_variance_onnx_model(
        input_shape=input_shape,
        axis=axis,
        correction=correction,
    )

    mod, params = tvm.relay.frontend.from_onnx(
        model,
        {"x": input_shape},
    )

    print("plugin_reduce_variance_ir:")
    print(mod)

    fbs_model = dyn.to_teco_infer_dyn(mod, {}, "teco_dyn")
    print("#"*60)
    print(fbs_model)
    print("#"*60)
    engine = tecoinference.Engine(fbs_model)
    ctx = engine.create_context()

    ctx.set_input(0, x_data)
    ctx.executor_run()
    out = ctx.get_output(0)
    print("-------------")
    print(out)
    print("-------------")
    expected = np.var(
        x_data,
        axis=axis,
        ddof=correction,
        keepdims=True,
    ).astype("float32")

    print("out shape:", out.shape)
    print("expected shape:", expected.shape)
    print("max abs diff:", np.max(np.abs(out - expected)))

    if np.allclose(out, expected, rtol=1e-4, atol=1e-4):
        print("✓ Test PASSED")
        return True
    else:
        print("✗ Test FAILED")
        print("out:")
        print(out)
        print("expected:")
        print(expected)
        return False


if __name__ == "__main__":
    results = []

    # axis == dims - 1，对应 right 分支
    results.append(run_case((2, 3, 4), axis=2, correction=0))

    # 0 < axis < dims - 1，对应 middle 分支
    results.append(run_case((2, 3, 4), axis=1, correction=0))

    # axis == 0，对应 left 分支
    results.append(run_case((2, 3, 4), axis=0, correction=0))

    # correction=1，样本方差
    results.append(run_case((2, 3, 4), axis=1, correction=1))

    print("=" * 60)
    if all(results):
        print("All plugin_reduce_variance tests PASSED!")
    else:
        print("Some plugin_reduce_variance tests FAILED!")
