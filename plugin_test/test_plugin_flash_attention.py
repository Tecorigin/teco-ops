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


def reference_paged_flash_attention(q, k_cache, v_cache, block_table,
                                   cu_seqlens_q, seqused_k):
    """Small CPU reference that exercises cumulative q metadata for B > 1."""
    total_tokens, num_heads, head_size = q.shape
    batch_size = block_table.shape[0]
    kv_heads = k_cache.shape[1]
    block_size = k_cache.shape[2]
    out = np.zeros_like(q)
    scale = 1.0 / math.sqrt(head_size)

    for b in range(batch_size):
        q_start, q_end = int(cu_seqlens_q[b]), int(cu_seqlens_q[b + 1])
        q_len = q_end - q_start
        k_len = int(seqused_k[b])
        num_blocks = (k_len + block_size - 1) // block_size
        # The kernel repeats each KV head for a contiguous group of query
        # heads (the same layout as repeat_interleave in the Python API
        # reference), rather than interleaving KV heads by modulo.
        num_kv_groups = num_heads // kv_heads
        for h in range(num_heads):
            kv_h = h // num_kv_groups
            keys = np.concatenate(
                [k_cache[int(block_table[b, logical_block]), kv_h]
                 for logical_block in range(num_blocks)], axis=0)[:k_len]
            values = np.concatenate(
                [v_cache[int(block_table[b, logical_block]), kv_h]
                 for logical_block in range(num_blocks)], axis=0)[:k_len]
            scores = (q[q_start:q_end, h].astype(np.float32)
                      @ keys.astype(np.float32).T) * scale
            causal_start = max(k_len - q_len, 0)
            for qi in range(q_len):
                allowed = causal_start + qi + 1
                if allowed < k_len:
                    scores[qi, allowed:] = -np.inf
            scores -= np.max(scores, axis=-1, keepdims=True)
            probs = np.exp(scores)
            probs /= np.sum(probs, axis=-1, keepdims=True)
            out[q_start:q_end, h] = (probs @ values.astype(np.float32)).astype(q.dtype)
    return out


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


def test_mixed_batch():
    """Mixed batch: cumulative q lengths must be converted per batch on device."""
    # The current flash-attention kernel tiles K/V with BN=32.  Keep this
    # ABI regression test on the supported cache block size so it exercises
    # cumulative q metadata instead of an unrelated unsupported shape.
    print("  mixed batch (B=2, q=[2,1], kv=[64,32])...", end=" ")
    B, H, KV, HS, BS = 2, 4, 2, 64, 32
    q_lens = [2, 1]
    kv_lens = [64, 32]
    total_q = sum(q_lens)
    num_blocks = 3
    np.random.seed(45)

    q = np.random.randn(total_q, H, HS).astype(np.float16)
    kc = np.random.randn(num_blocks, KV, BS, HS).astype(np.float16)
    vc = np.random.randn(num_blocks, KV, BS, HS).astype(np.float16)
    bt = np.array([[0, 1], [2, 0]], dtype=np.int32)
    cu = np.array([0, 2, 3], dtype=np.int32)
    sq = np.array(kv_lens, dtype=np.int32)

    input_shapes = {
        "q": (total_q, H, HS),
        "k_cache": (num_blocks, KV, BS, HS),
        "v_cache": (num_blocks, KV, BS, HS),
        "block_table": (B, 2),
        "cu_seqlens_q": (B + 1,),
        "seqused_k": (B,),
    }
    model = create_plugin_onnx_model(
        input_shapes,
        {"max_seqlen_q": max(q_lens), "max_seqlen_k": max(kv_lens),
         "max_block_num": num_blocks})
    mod, params = tvm.relay.frontend.from_onnx(model, input_shapes)
    fbs_model = dyn.to_teco_infer_dyn(mod, {}, "teco_dyn")
    engine = tecoinference.Engine(fbs_model)
    ctx = engine.create_context()
    for index, value in enumerate((q, kc, vc, bt, cu, sq)):
        ctx.set_input(index, value)
    ctx.executor_run()
    out = ctx.get_output(0)
    ref = reference_paged_flash_attention(q, kc, vc, bt, cu, sq)
    max_err = float(np.max(np.abs(out.astype(np.float32) - ref.astype(np.float32))))
    assert max_err < 0.1, f"mixed-batch max_err={max_err}"
    print(f"OK (max_err={max_err:.6f})")


if __name__ == "__main__":
    print("=" * 60)
    print("plugin_flash_attention Test Suite")
    print("=" * 60)

    test_prefill()
    test_decode()
    test_chunked_prefill()
    test_mixed_batch()

    print("=" * 60)
    print("All smoke tests passed (no crash = OK)")
    print("=" * 60)
