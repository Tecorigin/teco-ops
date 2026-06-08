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

import ctypes
import os
import onnx
from onnx import helper, TensorProto
import tvm
from tvm import relay
import tvm.relay

# plugin_lib_path = os.path.join(os.path.dirname(__file__), "build", "plugin", "example", "libTecoInferPlugin.so")
# if os.path.exists(plugin_lib_path):
#     ctypes.CDLL(plugin_lib_path)
# else:
#     raise FileNotFoundError(f"Plugin library not found: {plugin_lib_path}")

from tvm.plugin import plugins
import tecoinference
from tvm.contrib.teco_infer_dyn import dyn
import numpy as np


def create_plugin_flatten_rays_onnx_model(op_type, input_shapes, output_shape, attributes=None):
    input_names = list(input_shapes.keys())
    inputs = [
        helper.make_tensor_value_info(name, TensorProto.INT32, shape)
        for name, shape in zip(input_names, input_shapes)
    ]
    output_name = "output"
    output = helper.make_tensor_value_info(output_name, TensorProto.INT32, output_shape)

    attributes = attributes or {}
    flatten_rays_node = helper.make_node(
        op_type,
        input_names,
        [output_name],
        **attributes,
        domain="my_custom_ops",
        version=1
    )

    graph = helper.make_graph(
        [flatten_rays_node],
        op_type,
        inputs,
        [output],
    )
    model = helper.make_model(graph)
    opset_imports = model.opset_import
    opset_imports.append(helper.make_opsetid("my_custom_ops", 1))
    return model


plugins.register_op(
    op_name="plugin_flatten_rays",
    inputs=["rays"],
    attrs={"M": "int"}
)

def compute_expected_flatten_rays(rays_data, M):
    N = rays_data.shape[0]
    result = np.zeros(M, dtype=np.int32)
    for i in range(N):
        offset = rays_data[i, 0]
        num_steps = rays_data[i, 1]
        result[offset:offset + num_steps] = i
    return result


def test_plugin_flatten_rays_basic():
    print("Testing plugin_flatten_rays basic...")

    rays_data = np.array([
        [0, 3],
        [3, 2],
        [5, 4]
    ], dtype=np.int32)

    N = 3
    M = 9

    model_flatten = create_plugin_flatten_rays_onnx_model(
        op_type="plugin_flatten_rays",
        input_shapes={"rays": (N, 2)},
        output_shape=(M,),
        attributes={"M": M}
    )

    mod, params = tvm.relay.frontend.from_onnx(
        model_flatten, {"rays": (N, 2)}
    )
    print("plugin_flatten_rays_ir:", mod)
    
    fbs_model = dyn.to_teco_infer_dyn(mod, {}, "teco_dyn")
    engine = tecoinference.Engine(fbs_model)
    ctx = engine.create_context()
    
    ctx.set_input(0, rays_data)
    ctx.executor_run()
    out = ctx.get_output(0)
    
    print("Input rays_data: ", rays_data)
    print("N={}, M={}".format(N, M))
    print("out: ", out)
    
    expected = compute_expected_flatten_rays(rays_data, M)
    print("expected: ", expected)
    
    if np.array_equal(out, expected):
        print("✓ Basic test PASSED!")
        return True
    else:
        print("✗ Basic test FAILED!")
        print("Difference:", out - expected)
        return False


def test_plugin_flatten_rays_random():
    print("\nTesting plugin_flatten_rays with random data...")
    
    np.random.seed(42)
    N = 10
    
    num_steps_list = np.random.randint(1, 10, size=N)
    offsets = np.zeros(N, dtype=np.int32)
    offsets[1:] = np.cumsum(num_steps_list[:-1])
    rays_data = np.stack([offsets, num_steps_list], axis=1).astype(np.int32)
    M = int(num_steps_list.sum())
    
    print("Number of rays: {}".format(N))
    print("Total steps: {}".format(M))
    print("Rays data shape: {}".format(rays_data.shape))
    
    model_flatten = create_plugin_flatten_rays_onnx_model(
        op_type="plugin_flatten_rays",
        input_shapes={"rays": (N, 2)},
        output_shape=(M,),
        attributes={"M": M}
    )

    mod, params = tvm.relay.frontend.from_onnx(
        model_flatten, {"rays": (N, 2)}
    )

    fbs_model = dyn.to_teco_infer_dyn(mod, {}, "teco_dyn")
    engine = tecoinference.Engine(fbs_model)
    ctx = engine.create_context()

    ctx.set_input(0, rays_data)
    ctx.executor_run()
    out = ctx.get_output(0)

    expected = compute_expected_flatten_rays(rays_data, M)

    success = True
    for i in range(N):
        offset = int(offsets[i])
        num_steps = int(num_steps_list[i])
        segment = out[offset:offset + num_steps]

        if not np.all(segment == i):
            print("✗ Ray {}: expected all {}, got {}".format(i, i, segment))
            success = False

    if success:
        print("✓ Random test PASSED!")
    else:
        print("✗ Random test FAILED!")

    return success


if __name__ == "__main__":
    print("=" * 60)
    print("plugin_flatten_rays Test Suite")
    print("=" * 60)

    test1_passed = test_plugin_flatten_rays_basic()
    test2_passed = test_plugin_flatten_rays_random()

    print("\n" + "=" * 60)
    if test1_passed and test2_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)
