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

"""Plugin 测试: flash_attention.

Contract Note:
    All attributes registered for this plugin (see ATTR_TYPES / plugin_flash_attentionAttrs)
    must appear in the ONNX node attributes. TVM's get_relay_func
    (tvm/plugin/plugin_register.py) iterates over all registered attribute names
    and accesses attrs[key] without falling back to default values.
    To avoid KeyError, create_plugin_onnx_model provides default attribute values
    (has_softmax_scale=0, softmax_scale=0.0) for any omitted optional attributes.
"""

import math
import numpy as np
from tvm.plugin import plugins
import tecoinference
from tvm.contrib.teco_infer_dyn import dyn
from onnx import helper, TensorProto
import tvm
from tvm import relay

# Attr types registered for plugin_flash_attention in TVM.
# All attributes must match plugin_flash_attentionAttrs in C++.
ATTR_TYPES = {
    "max_seqlen_q": "int",
    "max_seqlen_k": "int",
    "max_block_num": "int",
    "has_softmax_scale": "int",
    "softmax_scale": "float",
}

# Default attributes provided when constructing the ONNX node.
# Legacy path: has_softmax_scale=0, softmax_scale=0.0 (kernel keeps default 1/sqrt(head_size)).
# Note: ONNX attribute types must match registered types (int and float, not str).
DEFAULT_ATTRS = {
    "has_softmax_scale": 0,
    "softmax_scale": 0.0,
}

# Register op once at module level
try:
    plugins.register_op(
        op_name="plugin_flash_attention",
        inputs=["q", "k_cache", "v_cache", "block_table", "q_seq_lens", "seqused_k"],
        attrs=ATTR_TYPES,
    )
except AssertionError:
    pass  # already registered


def create_plugin_onnx_model(input_shapes, attributes=None):
    """Build ONNX model with plugin_flash_attention node.

    Note:
        All registered plugin attributes must appear in the ONNX node attributes because
        TVM get_relay_func indexes attrs[key] without falling back to defaults.
        DEFAULT_ATTRS supplies default values (has_softmax_scale=0, softmax_scale=0.0),
        which callers can override via attributes.
    """
    input_names = list(input_shapes.keys())
    inputs = []
    for name, shape in input_shapes.items():
        dtype = TensorProto.INT32 if name in ("block_table", "q_seq_lens", "seqused_k") else TensorProto.FLOAT16
        inputs.append(helper.make_tensor_value_info(name, dtype, shape))
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT16, input_shapes["q"])

    node_attrs = dict(DEFAULT_ATTRS)
    if attributes:
        node_attrs.update(attributes)

    node = helper.make_node(
        "plugin_flash_attention",
        input_names,
        ["output"],
        **node_attrs,
        domain="my_custom_ops",
        version=1
    )
    graph = helper.make_graph([node], "plugin_flash_attention", inputs, [output])
    model = helper.make_model(graph)
    model.opset_import.append(helper.make_opsetid("my_custom_ops", 1))
    return model


def reference_attention(q, kc, vc, bt, q_lens, kv_lens, scale):
    """Independent fp32 paged causal attention, including explicit scale=0."""
    result = np.empty(q.shape, dtype=np.float32)
    offset = 0
    block_size = kc.shape[2]
    groups = q.shape[1] // kc.shape[1]
    for b, length in enumerate(q_lens):
        length, used = int(length), int(kv_lens[b])
        if length == 0:
            continue
        pages = bt[b, :(used + block_size - 1) // block_size]
        key = kc[pages].transpose(0, 2, 1, 3).reshape(-1, kc.shape[1], q.shape[2])[:used]
        value = vc[pages].transpose(0, 2, 1, 3).reshape(-1, vc.shape[1], q.shape[2])[:used]
        key = np.repeat(key.astype(np.float32), groups, axis=1).transpose(1, 2, 0)
        value = np.repeat(value.astype(np.float32), groups, axis=1).transpose(1, 0, 2)
        query = q[offset:offset + length].astype(np.float32).transpose(1, 0, 2)
        score = (query @ key) * scale
        visible = np.arange(used)[None, :] <= (used - length + np.arange(length))[:, None]
        score = np.where(visible[None], score, -np.inf)
        probability = np.exp(score - score.max(axis=-1, keepdims=True))
        probability /= probability.sum(axis=-1, keepdims=True)
        result[offset:offset + length] = (probability @ value).transpose(1, 0, 2)
        offset += length
    return result


def check_output(name, out, expected_shape, reference):
    """Assert the plugin produced a real, finite, non-degenerate result.

    The plugin Enqueue() discards the tecoopsFlashAttention() return status, so a
    NOT_SUPPORTED rejection (e.g. ALGO0 requires block_size == 32) would leave the
    output buffer uninitialised and the old shape-only check would still print OK.
    These fixtures therefore only use the supported block_size == 32, and this
    check verifies the values are finite and non-constant so a masked
    non-computation cannot pass as OK.
    """
    assert out.shape == expected_shape, f"{name}: shape mismatch: {out.shape}"
    values = np.asarray(out, dtype=np.float32)
    assert np.isfinite(values).all(), f"{name}: non-finite values in output"
    assert values.std() > 0.0, f"{name}: output is constant, kernel likely did not run"
    np.testing.assert_allclose(values, reference, atol=2e-3, rtol=1e-2, err_msg=name)
    print(f"OK (shape={out.shape}, max_error={np.max(np.abs(values - reference)):.6e})")


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
    bt = np.arange(blk, dtype=np.int32).reshape(1, blk)
    cu = np.array([L], dtype=np.int32)
    sq = np.array([S], dtype=np.int32)

    model = create_plugin_onnx_model(
        {"q": (L, H, HS), "k_cache": (blk, KV, BS, HS), "v_cache": (blk, KV, BS, HS),
         "block_table": (1, blk), "q_seq_lens": (1,), "seqused_k": (1,)},
        {"max_seqlen_q": L, "max_seqlen_k": S, "max_block_num": blk}
    )
    mod, params = tvm.relay.frontend.from_onnx(
        model, {"q": (L, H, HS), "k_cache": (blk, KV, BS, HS), "v_cache": (blk, KV, BS, HS),
                "block_table": (1, blk), "q_seq_lens": (1,), "seqused_k": (1,)})
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
    check_output("prefill", out, (L, H, HS),
                 reference_attention(q, kc, vc, bt, cu, sq, 1.0 / math.sqrt(HS)))


def test_decode():
    """decode: L=1, S=128"""
    # ALGO0 dispatch requires block_size == 32 (find_flash_attention.cpp); a
    # block_size of 128 used to be rejected with NOT_SUPPORTED while this smoke
    # test still printed OK (the plugin Enqueue() discards the C ABI status and
    # the output buffer was never written), so use the supported 32 here.
    print("  decode (L=1, S=128)...", end=" ")
    H, KV, HS, BS = 8, 4, 64, 32
    L, S = 1, 128
    blk = (S + BS - 1) // BS
    np.random.seed(43)

    q = np.random.randn(L, H, HS).astype(np.float16)
    kc = np.random.randn(blk, KV, BS, HS).astype(np.float16)
    vc = np.random.randn(blk, KV, BS, HS).astype(np.float16)
    bt = np.arange(blk, dtype=np.int32).reshape(1, blk)
    cu = np.array([L], dtype=np.int32)
    sq = np.array([S], dtype=np.int32)

    model = create_plugin_onnx_model(
        {"q": (L, H, HS), "k_cache": (blk, KV, BS, HS), "v_cache": (blk, KV, BS, HS),
         "block_table": (1, blk), "q_seq_lens": (1,), "seqused_k": (1,)},
        {"max_seqlen_q": L, "max_seqlen_k": S, "max_block_num": blk}
    )
    mod, params = tvm.relay.frontend.from_onnx(
        model, {"q": (L, H, HS), "k_cache": (blk, KV, BS, HS), "v_cache": (blk, KV, BS, HS),
                "block_table": (1, blk), "q_seq_lens": (1,), "seqused_k": (1,)})
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
    check_output("decode", out, (L, H, HS),
                 reference_attention(q, kc, vc, bt, cu, sq, 1.0 / math.sqrt(HS)))


def test_chunked_prefill():
    """chunked prefill: L=32, S=64"""
    # Same ALGO0 block_size == 32 constraint as test_decode
    # (see teco/ual/ops/flash_attention/find_flash_attention.cpp).
    print("  chunked prefill (L=32, S=64)...", end=" ")
    H, KV, HS, BS = 8, 4, 64, 32
    L, S = 32, 64
    blk = (S + BS - 1) // BS
    np.random.seed(44)

    q = np.random.randn(L, H, HS).astype(np.float16)
    kc = np.random.randn(blk, KV, BS, HS).astype(np.float16)
    vc = np.random.randn(blk, KV, BS, HS).astype(np.float16)
    bt = np.arange(blk, dtype=np.int32).reshape(1, blk)
    cu = np.array([L], dtype=np.int32)
    sq = np.array([S], dtype=np.int32)

    model = create_plugin_onnx_model(
        {"q": (L, H, HS), "k_cache": (blk, KV, BS, HS), "v_cache": (blk, KV, BS, HS),
         "block_table": (1, blk), "q_seq_lens": (1,), "seqused_k": (1,)},
        {"max_seqlen_q": L, "max_seqlen_k": S, "max_block_num": blk}
    )
    mod, params = tvm.relay.frontend.from_onnx(
        model, {"q": (L, H, HS), "k_cache": (blk, KV, BS, HS), "v_cache": (blk, KV, BS, HS),
                "block_table": (1, blk), "q_seq_lens": (1,), "seqused_k": (1,)})
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
    check_output("chunked prefill", out, (L, H, HS),
                 reference_attention(q, kc, vc, bt, cu, sq, 1.0 / math.sqrt(HS)))


def test_explicit_scale():
    """explicit softmax_scale must reach the kernel, not be silently dropped.

    scale=0 makes every valid QK score zero, so the attention weights become
    uniform over the visible KV entries.  The default path uses 1/sqrt(hd).
    Running the same tensors through both paths therefore has to change the
    output: a plugin that quietly ignored the attribute would return the exact
    same buffer and fail this assertion.
    """
    print("  explicit scale=0 vs default (L=S=64)...", end=" ")
    H, KV, HS, BS = 8, 4, 64, 32
    L, S = 64, 64
    blk = (S + BS - 1) // BS
    np.random.seed(45)

    q = np.random.randn(L, H, HS).astype(np.float16)
    kc = np.random.randn(blk, KV, BS, HS).astype(np.float16)
    vc = np.random.randn(blk, KV, BS, HS).astype(np.float16)
    bt = np.arange(blk, dtype=np.int32).reshape(1, blk)
    cu = np.array([L], dtype=np.int32)
    sq = np.array([S], dtype=np.int32)

    shapes = {"q": (L, H, HS), "k_cache": (blk, KV, BS, HS),
              "v_cache": (blk, KV, BS, HS), "block_table": (1, blk),
              "q_seq_lens": (1,), "seqused_k": (1,)}
    base = {"max_seqlen_q": L, "max_seqlen_k": S, "max_block_num": blk}

    def run(extra_attrs):
        attrs = dict(base, **extra_attrs)
        model = create_plugin_onnx_model(shapes, attrs)
        mod, params = tvm.relay.frontend.from_onnx(model, shapes)
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
        return np.asarray(ctx.get_output(0), dtype=np.float32)

    default_out = run({})
    scaled_out = run({"has_softmax_scale": 1, "softmax_scale": 0.0})
    assert np.isfinite(default_out).all() and np.isfinite(scaled_out).all()
    np.testing.assert_allclose(default_out,
        reference_attention(q, kc, vc, bt, cu, sq, 1.0 / math.sqrt(HS)),
        atol=2e-3, rtol=1e-2)
    np.testing.assert_allclose(scaled_out,
        reference_attention(q, kc, vc, bt, cu, sq, 0.0), atol=2e-3, rtol=1e-2)
    delta = float(np.abs(default_out - scaled_out).max())
    assert delta > 1e-2, f"explicit softmax_scale was ignored (max abs diff {delta})"
    print(f"OK (explicit-scale path active, max abs diff {delta:.4f})")


if __name__ == "__main__":
    print("=" * 60)
    print("plugin_flash_attention Test Suite")
    print("=" * 60)

    test_prefill()
    test_decode()
    test_chunked_prefill()
    test_explicit_scale()

    print("=" * 60)
    print("All plugin tests passed (shape + finite + non-constant + explicit-scale parity)")
    print("=" * 60)
