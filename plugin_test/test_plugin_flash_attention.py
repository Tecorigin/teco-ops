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

import math
import numpy as np
from tvm.plugin import plugins
import tecoinference
from tvm.contrib.teco_infer_dyn import dyn
from onnx import helper, TensorProto
import tvm
from tvm import relay

# Register op once at module level
try:
    plugins.register_op(
        op_name="plugin_flash_attention",
        inputs=["q", "k_cache", "v_cache", "block_table", "cu_seqlens_q", "seqused_k"],
        attrs={"max_seqlen_q": "int", "max_seqlen_k": "int", "max_block_num": "int"}
    )
except AssertionError:
    pass  # already registered


def create_plugin_onnx_model(input_shapes, attributes):
    """Build ONNX model with plugin_flash_attention node."""
    input_names = list(input_shapes.keys())
    inputs = []
    for name, shape in input_shapes.items():
        dtype = TensorProto.INT32 if name in ("block_table", "cu_seqlens_q", "seqused_k") else TensorProto.FLOAT16
        inputs.append(helper.make_tensor_value_info(name, dtype, shape))
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT16, input_shapes["q"])

    node = helper.make_node(
        "plugin_flash_attention",
        input_names,
        ["output"],
        **attributes,
        domain="my_custom_ops",
        version=1
    )
    graph = helper.make_graph([node], "plugin_flash_attention", inputs, [output])
    model = helper.make_model(graph)
    model.opset_import.append(helper.make_opsetid("my_custom_ops", 1))
    return model


def test_prefill():
    """prefill: L=S=64"""
    print("  prefill (L=S=64)...", end=" ")
    B, H, KV, HS, BS = 1, 8, 4, 64, 32
    L = S = 64
    blk = (S + BS - 1) // BS
    np.random.seed(42)

    q = np.random.randn(L, H, HS).astype(np.float16)
    kc = np.random.randn(blk, KV, BS, HS).astype(np.float16)
    vc = np.random.randn(blk, KV, BS, HS).astype(np.float16)
    bt = np.zeros((1, blk), dtype=np.int32)
    cu = np.array([0, L], dtype=np.int32)
    sq = np.array([S], dtype=np.int32)

    model = create_plugin_onnx_model(
        {"q": (L, H, HS), "k_cache": (blk, KV, BS, HS), "v_cache": (blk, KV, BS, HS),
         "block_table": (1, blk), "cu_seqlens_q": (2,), "seqused_k": (1,)},
        {"max_seqlen_q": L, "max_seqlen_k": S, "max_block_num": blk}
    )
    mod, params = tvm.relay.frontend.from_onnx(
        model, {"q": (L, H, HS), "k_cache": (blk, KV, BS, HS), "v_cache": (blk, KV, BS, HS),
                "block_table": (1, blk), "cu_seqlens_q": (2,), "seqused_k": (1,)})
    fbs_model = dyn.to_teco_infer_dyn(mod, {}, "teco_dyn")
    engine = tecoinference.Engine(fbs_model)
    ctx = engine.create_context()

    ctx.set_input(0, q)
    ctx.set_input(1, kc)
    ctx.set_input(2, vc)
    ctx.set_input(3, bt)
    ctx.set_input(4, cu)
    ctx.set_input(5, sq)
    ctx.executor_run()
    out = ctx.get_output(0)
    assert out.shape == (L, H, HS), f"shape mismatch: {out.shape}"
    print(f"OK (shape={out.shape})")


def test_decode():
    """decode: L=1, S=128"""
    print("  decode (L=1, S=128)...", end=" ")
    H, KV, HS, BS = 8, 4, 64, 128
    L, S = 1, 128
    blk = (S + BS - 1) // BS
    np.random.seed(43)

    q = np.random.randn(L, H, HS).astype(np.float16)
    kc = np.random.randn(blk, KV, BS, HS).astype(np.float16)
    vc = np.random.randn(blk, KV, BS, HS).astype(np.float16)
    bt = np.zeros((1, blk), dtype=np.int32)
    cu = np.array([0, L], dtype=np.int32)
    sq = np.array([S], dtype=np.int32)

    model = create_plugin_onnx_model(
        {"q": (L, H, HS), "k_cache": (blk, KV, BS, HS), "v_cache": (blk, KV, BS, HS),
         "block_table": (1, blk), "cu_seqlens_q": (2,), "seqused_k": (1,)},
        {"max_seqlen_q": L, "max_seqlen_k": S, "max_block_num": blk}
    )
    mod, params = tvm.relay.frontend.from_onnx(
        model, {"q": (L, H, HS), "k_cache": (blk, KV, BS, HS), "v_cache": (blk, KV, BS, HS),
                "block_table": (1, blk), "cu_seqlens_q": (2,), "seqused_k": (1,)})
    fbs_model = dyn.to_teco_infer_dyn(mod, {}, "teco_dyn")
    engine = tecoinference.Engine(fbs_model)
    ctx = engine.create_context()

    ctx.set_input(0, q)
    ctx.set_input(1, kc)
    ctx.set_input(2, vc)
    ctx.set_input(3, bt)
    ctx.set_input(4, cu)
    ctx.set_input(5, sq)
    ctx.executor_run()
    out = ctx.get_output(0)
    print(f"OK (shape={out.shape})")


def test_chunked_prefill():
    """chunked prefill: L=32, S=64"""
    print("  chunked prefill (L=32, S=64)...", end=" ")
    H, KV, HS, BS = 8, 4, 64, 64
    L, S = 32, 64
    blk = (S + BS - 1) // BS
    np.random.seed(44)

    q = np.random.randn(L, H, HS).astype(np.float16)
    kc = np.random.randn(blk, KV, BS, HS).astype(np.float16)
    vc = np.random.randn(blk, KV, BS, HS).astype(np.float16)
    bt = np.zeros((1, blk), dtype=np.int32)
    cu = np.array([0, L], dtype=np.int32)
    sq = np.array([S], dtype=np.int32)

    model = create_plugin_onnx_model(
        {"q": (L, H, HS), "k_cache": (blk, KV, BS, HS), "v_cache": (blk, KV, BS, HS),
         "block_table": (1, blk), "cu_seqlens_q": (2,), "seqused_k": (1,)},
        {"max_seqlen_q": L, "max_seqlen_k": S, "max_block_num": blk}
    )
    mod, params = tvm.relay.frontend.from_onnx(
        model, {"q": (L, H, HS), "k_cache": (blk, KV, BS, HS), "v_cache": (blk, KV, BS, HS),
                "block_table": (1, blk), "cu_seqlens_q": (2,), "seqused_k": (1,)})
    fbs_model = dyn.to_teco_infer_dyn(mod, {}, "teco_dyn")
    engine = tecoinference.Engine(fbs_model)
    ctx = engine.create_context()

    ctx.set_input(0, q)
    ctx.set_input(1, kc)
    ctx.set_input(2, vc)
    ctx.set_input(3, bt)
    ctx.set_input(4, cu)
    ctx.set_input(5, sq)
    ctx.executor_run()
    out = ctx.get_output(0)
    print(f"OK (shape={out.shape})")


if __name__ == "__main__":
    print("=" * 60)
    print("plugin_flash_attention Test Suite")
    print("=" * 60)

    test_prefill()
    test_decode()
    test_chunked_prefill()

    print("=" * 60)
    print("All smoke tests passed (no crash = OK)")
    print("=" * 60)
