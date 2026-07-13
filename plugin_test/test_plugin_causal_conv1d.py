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
"""Plugin 测试: causal_conv1d"""
import torch
import torch.nn.functional as F
import numpy as np
from tvm.plugin import plugins
import tecoinference
from tvm.contrib.teco_infer_dyn import dyn
from onnx import helper, TensorProto
import tvm
from tvm import relay


plugins.register_op(
    op_name="plugin_causal_conv1d_fn",
    inputs=["x", "weight", "queryStartLoc", "convStateIndices",
            "hasInitialState", "convState"],
    attrs={
        "pad_slot_id": "int",
        "conv_state_stride": "int",
        "x_seq_stride": "int",
        "out_seq_stride": "int",
        "api_mode": "int",
        "act_mode": "int",
    },
    n_out=2,
)


def silu(x):
    return x / (1.0 + np.exp(-x))


def causal_conv1d_ref(
    x, weight, query_start_loc, cache_indices, has_initial_state,
    conv_states, pad_slot_id):
    """CPU 参考: causal conv1d

    Args:
        x:              (dim, total_seqlen) float32
        weight:         (width, dim) float32
        query_start_loc: (batch+1,) int32
        cache_indices:   (batch,) int32
        has_initial_state: (batch,) int8  (0=无, 1=有)
        conv_states:    (num_entries, width-1, dim) float32
        pad_slot_id:    int

    Returns:
        out:         (dim, total_seqlen) float32
        conv_states: 更新后的状态 (in-place)
    """
    dim, total_seqlen = x.shape
    width = weight.shape[0]
    batch = query_start_loc.shape[0] - 1

    weight_3d = weight.T.unsqueeze(1)  # [dim, 1, width] → groups=dim conv weight
    dtype_in = x.dtype
    out = torch.zeros_like(x)

    for i in range(batch):
        state_idx = int(cache_indices[i])
        if state_idx == pad_slot_id:
            continue

        start = int(query_start_loc[i])
        end = int(query_start_loc[i + 1])
        seq_len = end - start

        if seq_len == 0:
            continue

        # seq_x: [dim, seq_len]
        seq_x = x[:, start:end]

        # prefix: [1, dim, width-1]
        if has_initial_state[i]:
            prefix = conv_states[state_idx].T.unsqueeze(0)  # [1, dim, width-1]
        else:
            prefix = torch.zeros(1, dim, width - 1, dtype=seq_x.dtype, device=seq_x.device)

        # full_x: [1, dim, width-1+seq_len]
        full_x = torch.cat([prefix, seq_x.unsqueeze(0)], dim=-1)

        # conv_out: [1, dim, seq_len]
        conv_out = F.conv1d(full_x, weight_3d, padding=0, groups=dim)
        conv_out = F.silu(conv_out).to(dtype_in)

        # write to out
        out[:, start:end] = conv_out.squeeze(0)

        # update conv state
        new_state = full_x[:, :, -(width - 1):].squeeze(0).T.contiguous()
        conv_states[state_idx] = new_state

    return out, conv_states


def create_onnx_model(total_seqlen, dim, width, num_entries, batch,
                      pad_slot_id=-1, conv_state_stride=None,
                      x_seq_stride=None, out_seq_stride=None):
    """创建 causal_conv1d ONNX 模型"""
    if conv_state_stride is None:
        conv_state_stride = (width - 1) * dim
    if x_seq_stride is None:
        x_seq_stride = dim
    if out_seq_stride is None:
        out_seq_stride = dim

    inputs = [
        helper.make_tensor_value_info("x", TensorProto.FLOAT16,
                                       (total_seqlen, dim)),
        helper.make_tensor_value_info("weight", TensorProto.FLOAT16,
                                       (width, dim)),
        helper.make_tensor_value_info("queryStartLoc", TensorProto.INT32,
                                       (batch + 1,)),
        helper.make_tensor_value_info("convStateIndices", TensorProto.INT32,
                                       (batch,)),
        helper.make_tensor_value_info("hasInitialState", TensorProto.INT8,
                                       (batch,)),
        helper.make_tensor_value_info("convState", TensorProto.FLOAT16,
                                       (num_entries, width - 1, dim)),
    ]

    outputs = [
        helper.make_tensor_value_info("out", TensorProto.FLOAT16,
                                       (total_seqlen, dim)),
        helper.make_tensor_value_info("convState_out", TensorProto.FLOAT16,
                                       (num_entries, width - 1, dim)),
    ]

    node = helper.make_node(
        "plugin_causal_conv1d_fn",
        ["x", "weight", "queryStartLoc", "convStateIndices",
         "hasInitialState", "convState"],
        ["out", "convState_out"],
        pad_slot_id=int(pad_slot_id),
        conv_state_stride=int(conv_state_stride),
        x_seq_stride=int(x_seq_stride),
        out_seq_stride=int(out_seq_stride),
        api_mode=0,
        act_mode=0,
        domain="my_custom_ops",
        version=1,
    )

    graph = helper.make_graph([node], "causal_conv1d", inputs, outputs)
    model = helper.make_model(graph)
    model.opset_import.append(helper.make_opsetid("my_custom_ops", 1))
    return model


def _test_one_case(dim, total_seqlen, width, num_entries,
                   seq_lens, cache_indices, has_init, pad_slot_id=-1):
    """单个 plugin causal_conv1d 测试"""
    batch = len(seq_lens)

    # 构造 query_start_loc
    qsl = [0]
    for sl in seq_lens:
        qsl.append(qsl[-1] + sl)
    qsl_np = np.array(qsl, dtype=np.int32)
    cidx_np = np.array(cache_indices, dtype=np.int32)
    has_init_np = np.array(has_init, dtype=np.int8)

    # 随机输入 (total_seqlen, dim) layout → ONNX/C API 的布局
    x_np = np.random.randn(total_seqlen, dim).astype(np.float16)
    w_np = np.random.randn(width, dim).astype(np.float16)
    conv_np = np.random.randn(num_entries, width - 1, dim).astype(np.float16)

    # ---- CPU 参考 (dim-first) ----
    # 参考使用 torch，需要 transpose 到 [dim, total_seqlen]
    x_t = torch.from_numpy(x_np.T.copy())  # [totalSeqLen, dim] → [dim, totalSeqLen]
    w_t = torch.from_numpy(w_np)
    conv_t = torch.from_numpy(conv_np.copy())
    qsl_t = torch.from_numpy(qsl_np)
    cidx_t = torch.from_numpy(cidx_np)
    has_init_t = torch.from_numpy(has_init_np)

    out_ref_t, conv_ref_t = causal_conv1d_ref(
        x_t, w_t, qsl_t, cidx_t, has_init_t, conv_t, pad_slot_id)

    # 参考返回 [dim, totalSeqLen] → transpose 回 [totalSeqLen, dim] 比对
    out_ref = out_ref_t.numpy().T.astype(np.float16)
    conv_ref = conv_ref_t.numpy().astype(np.float16)

    # ---- ONNX / 推理引擎 ----
    model = create_onnx_model(
        total_seqlen, dim, width, num_entries, batch, pad_slot_id)
    input_shapes = {
        "x": (total_seqlen, dim),
        "weight": (width, dim),
        "queryStartLoc": (batch + 1,),
        "convStateIndices": (batch,),
        "hasInitialState": (batch,),
        "convState": (num_entries, width - 1, dim),
    }
    mod, _ = tvm.relay.frontend.from_onnx(model, input_shapes)

    fbs_model = dyn.to_teco_infer_dyn(mod, {}, "teco_dyn")
    engine = tecoinference.Engine(fbs_model)
    ctx = engine.create_context()
    ctx.set_input(0, x_np)
    ctx.set_input(1, w_np)
    ctx.set_input(2, qsl_np)
    ctx.set_input(3, cidx_np)
    ctx.set_input(4, has_init_np)
    ctx.set_input(5, conv_np)
    ctx.executor_run()
    out = ctx.get_output(0)
    conv_out = ctx.get_output(1)

    max_err_out = np.abs(out.astype(np.float32) - out_ref.astype(np.float32)).max()
    max_err_conv = np.abs(conv_out.astype(np.float32) - conv_ref.astype(np.float32)).max()
    return max_err_out, max_err_conv


def test_causal_conv1d():
    """基本 causal_conv1d plugin 精度测试"""
    print("=" * 60)
    print("Test: plugin_causal_conv1d_fn")
    all_passed = True

    cases = [
        # (dim, total_seqlen, width, num_entries, [seq_lens], cache_indices, [has_init])
        (2048,   8, 4, 1, [8],         [0],   [0]),     # 单batch, 无初态
        (1280, 512, 4, 2, [256, 256], [0, 1], [0, 0]),  # 双batch, 无初态
        (2048,   8, 4, 1, [8],         [0],   [1]),     # 单batch, 有初态
    ]

    for dim, total_seqlen, width, num_entries, seq_lens, cache_indices, has_init in cases:
        max_err_out, max_err_conv = _test_one_case(
            dim, total_seqlen, width, num_entries, seq_lens, cache_indices, has_init)

        passed = max_err_out < 2e-2 and max_err_conv < 2e-2
        all_passed = all_passed and passed
        info = f"dim={dim:4d} seq={total_seqlen:4d} w={width} batch={len(seq_lens)}"
        print(f"  {info}  out_err={max_err_out:.6e}  conv_err={max_err_conv:.6e}  {'PASSED' if passed else 'FAILED'}")

    return all_passed


def test_causal_conv1d_pad():
    """测试 pad_slot_id 跳过逻辑"""
    print("=" * 60)
    print("Test: plugin_causal_conv1d_fn pad skip")
    all_passed = True

    cases = [
        # 第一个 batch 的 cache_index == pad_slot_id，应被跳过
        (256, 12, 4, 4, [6, 6], [-1, 2], [0, 1], -1),
    ]

    for dim, total_seqlen, width, num_entries, seq_lens, cache_indices, has_init, pad_slot_id in cases:
        max_err_out, max_err_conv = _test_one_case(
            dim, total_seqlen, width, num_entries,
            seq_lens, cache_indices, has_init, pad_slot_id)

        passed = max_err_out < 2e-2 and max_err_conv < 2e-2
        all_passed = all_passed and passed
        info = f"dim={dim:4d} seq={total_seqlen:4d} w={width} batch={len(seq_lens)} pad_skip"
        print(f"  {info}  out_err={max_err_out:.6e}  conv_err={max_err_conv:.6e}  {'PASSED' if passed else 'FAILED'}")

    return all_passed


if __name__ == "__main__":
    print("=" * 60)
    print("plugin_causal_conv1d_fn Test Suite")
    print("=" * 60)
    r1 = test_causal_conv1d()
    r2 = test_causal_conv1d_pad()
    print("\n" + "=" * 60)
    print("ALL PASSED" if (r1 and r2) else "SOME FAILED")
    print("=" * 60)
