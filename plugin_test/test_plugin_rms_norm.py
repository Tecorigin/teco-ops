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
"""Plugin 测试: rms_norm"""
import numpy as np
from tvm.plugin import plugins
import tecoinference
from tvm.contrib.teco_infer_dyn import dyn
from onnx import helper, TensorProto
import tvm
from tvm import relay


plugins.register_op(
    op_name="plugin_rms_norm",
    inputs=["input", "weight"],
    attrs={"eps": "float"}
)

plugins.register_op(
    op_name="plugin_rms_norm_add",
    inputs=["input", "weight", "residual"],
    attrs={"eps": "float"},
    n_out=2
)


def create_onnx_model_pure(input_shape, eps=1e-5):
    """纯 RMSNorm: input + weight -> output"""
    inputs = [
        helper.make_tensor_value_info("input", TensorProto.FLOAT16, input_shape),
        helper.make_tensor_value_info("weight", TensorProto.FLOAT16, [input_shape[1]]),
    ]
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT16, input_shape)

    node = helper.make_node(
        "plugin_rms_norm",
        ["input", "weight"],
        ["output"],
        eps=float(eps),
        domain="my_custom_ops",
        version=1
    )

    graph = helper.make_graph([node], "rms_norm_pure", inputs, [output])
    model = helper.make_model(graph)
    model.opset_import.append(helper.make_opsetid("my_custom_ops", 1))
    return model


def create_onnx_model_add(input_shape, eps=1e-5):
    """RMSNorm + Add: input + weight + residual -> output + residual_out"""
    inputs = [
        helper.make_tensor_value_info("input", TensorProto.FLOAT16, input_shape),
        helper.make_tensor_value_info("weight", TensorProto.FLOAT16, [input_shape[1]]),
        helper.make_tensor_value_info("residual", TensorProto.FLOAT16, input_shape),
    ]
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT16, input_shape)
    res_out = helper.make_tensor_value_info("residual_out", TensorProto.FLOAT16, input_shape)

    node = helper.make_node(
        "plugin_rms_norm_add",
        ["input", "weight", "residual"],
        ["output", "residual_out"],
        eps=float(eps),
        domain="my_custom_ops",
        version=1
    )

    graph = helper.make_graph([node], "rms_norm_add", inputs, [output, res_out])
    model = helper.make_model(graph)
    model.opset_import.append(helper.make_opsetid("my_custom_ops", 1))
    return model


def rms_norm_ref(x, weight, eps=1e-5):
    """NumPy 参考: RMSNorm"""
    xf = x.astype(np.float32)
    wf = weight.astype(np.float32)
    rstd = 1.0 / np.sqrt(np.mean(xf ** 2, axis=-1, keepdims=True) + eps)
    return (xf * rstd * wf).astype(np.float16)


def rms_norm_add_ref(x, residual, weight, eps=1e-5):
    """NumPy 参考: residual add + RMSNorm"""
    xf = x.astype(np.float32)
    rf = residual.astype(np.float32)
    wf = weight.astype(np.float32)
    x_add = xf + rf
    rstd = 1.0 / np.sqrt(np.mean(x_add ** 2, axis=-1, keepdims=True) + eps)
    out = (x_add * rstd * wf).astype(np.float16)
    return out, x_add.astype(np.float16)


def test_plugin_rms_norm_pure():
    """纯 RMSNorm（多维度）"""
    print("=" * 60)
    print("Test: plugin_rms_norm (pure)")
    num_tokens = 64

    norm_cases = [
        (128,  1e-6),
        (128,  1e-5),
        (1536, 1e-6),
        (2048, 1e-5),
        (4096, 1e-6),
    ]

    all_passed = True
    for hidden_size, eps in norm_cases:
        x = np.random.randn(num_tokens, hidden_size).astype(np.float16)
        w = np.random.randn(hidden_size).astype(np.float16)

        expected = rms_norm_ref(x, w, eps)

        model = create_onnx_model_pure((num_tokens, hidden_size), eps)
        mod, _ = tvm.relay.frontend.from_onnx(model, {
            "input": (num_tokens, hidden_size),
            "weight": (hidden_size,),
        })

        fbs_model = dyn.to_teco_infer_dyn(mod, {}, "teco_dyn")
        engine = tecoinference.Engine(fbs_model)
        ctx = engine.create_context()
        ctx.set_input(0, x)
        ctx.set_input(1, w)
        ctx.executor_run()
        out = ctx.get_output(0)

        max_err = np.abs(out.astype(np.float32) - expected.astype(np.float32)).max()
        passed = max_err < 0.01
        all_passed = all_passed and passed
        print(f"  hidden={hidden_size:4d}  eps={eps:.0e}  max_error = {max_err:.6e}  {'PASSED' if passed else 'FAILED'}")

    return all_passed


def test_plugin_rms_norm_add():
    """RMSNorm + residual add（多维度）"""
    print("=" * 60)
    print("Test: plugin_rms_norm (add)")
    num_tokens = 64

    add_cases = [
        (1536, 1e-6),
        (2048, 1e-5),
        (4096, 1e-6),
    ]

    all_passed = True
    for hidden_size, eps in add_cases:
        x = np.random.randn(num_tokens, hidden_size).astype(np.float16)
        r = np.random.randn(num_tokens, hidden_size).astype(np.float16)
        w = np.random.randn(hidden_size).astype(np.float16)

        expected_out, expected_res = rms_norm_add_ref(x, r, w, eps)

        model = create_onnx_model_add((num_tokens, hidden_size), eps)
        mod, _ = tvm.relay.frontend.from_onnx(model, {
            "input": (num_tokens, hidden_size),
            "weight": (hidden_size,),
            "residual": (num_tokens, hidden_size),
        })

        fbs_model = dyn.to_teco_infer_dyn(mod, {}, "teco_dyn")
        engine = tecoinference.Engine(fbs_model)
        ctx = engine.create_context()
        ctx.set_input(0, x)
        ctx.set_input(1, w)
        ctx.set_input(2, r)
        ctx.executor_run()
        out = ctx.get_output(0)
        res_out = ctx.get_output(1)

        max_err_out = np.abs(out.astype(np.float32) - expected_out.astype(np.float32)).max()
        max_err_res = np.abs(res_out.astype(np.float32) - expected_res.astype(np.float32)).max()
        passed = max_err_out < 0.01 and max_err_res < 0.01
        all_passed = all_passed and passed
        print(f"  hidden={hidden_size:4d}  eps={eps:.0e}  out_max_err = {max_err_out:.6e}  res_max_err = {max_err_res:.6e}  {'PASSED' if passed else 'FAILED'}")

    return all_passed


if __name__ == "__main__":
    r1 = test_plugin_rms_norm_pure()
    r2 = test_plugin_rms_norm_add()
    print()
    print("ALL PASSED" if (r1 and r2) else "SOME FAILED")
