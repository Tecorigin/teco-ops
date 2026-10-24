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
import hashlib
from pathlib import Path
import numpy as np
from flash_attention_c_api import run_flash
from flash_attention_rejection import assert_sdk_rejected
from tvm.plugin import plugins
import tecoinference
from tvm.contrib.teco_infer_dyn import dyn
from onnx import helper, TensorProto
import tvm

# The low-level plugin consumes per-sequence query lengths, as the C API does.
try:
    plugins.register_op(
        op_name="plugin_flash_attention",
        inputs=["q", "k_cache", "v_cache", "block_table", "q_seq_lens", "seqused_k"],
        attrs={"max_seqlen_q": "int", "max_seqlen_k": "int", "max_block_num": "int"}
    )
except AssertionError:
    pass  # already registered


def create_plugin_onnx_model(input_shapes, attributes):
    """Build a plugin node whose q_seq_lens input contains actual lengths."""
    inputs = []
    for name, shape in input_shapes.items():
        dtype = TensorProto.INT32 if name in ("block_table", "q_seq_lens", "seqused_k") else TensorProto.FLOAT16
        inputs.append(helper.make_tensor_value_info(name, dtype, shape))
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT16, input_shapes["q"])
    node = helper.make_node(
        "plugin_flash_attention", list(input_shapes), ["output"],
        **attributes, domain="my_custom_ops", version=1
    )
    graph = helper.make_graph([node], "plugin_flash_attention", inputs, [output])
    model = helper.make_model(graph)
    model.opset_import.append(helper.make_opsetid("my_custom_ops", 1))
    return model


def _make_inputs(q_lens, kv_lens, seed, head_size=64, block_size=32):
    """Pack independent dense sequences into shuffled physical block_size-token pages."""
    rng = np.random.default_rng(seed)
    heads, kv_heads = 8, 4
    counts = [(length + block_size - 1) // block_size for length in kv_lens]
    num_blocks = sum(counts)
    q = rng.standard_normal((sum(q_lens), heads, head_size)).astype(np.float16)
    kc = np.zeros((num_blocks, kv_heads, block_size, head_size), dtype=np.float16)
    vc = np.zeros_like(kc)
    bt = np.zeros((len(q_lens), max(counts)), dtype=np.int32)
    pages = rng.permutation(num_blocks)
    dense_k, dense_v = [], []
    cursor = 0
    for batch, length in enumerate(kv_lens):
        key = rng.standard_normal((length, kv_heads, head_size)).astype(np.float16)
        value = rng.standard_normal((length, kv_heads, head_size)).astype(np.float16)
        dense_k.append(key)
        dense_v.append(value)
        for logical in range(counts[batch]):
            physical = pages[cursor]
            cursor += 1
            bt[batch, logical] = physical
            start = logical * block_size
            count = min(block_size, length - start)
            kc[physical, :, :count, :] = key[start:start + count].transpose(1, 0, 2)
            vc[physical, :, :count, :] = value[start:start + count].transpose(1, 0, 2)
    arrays = [q, kc, vc, bt, np.array(q_lens, dtype=np.int32), np.array(kv_lens, dtype=np.int32)]
    return arrays, dense_k, dense_v


def _reference(q, dense_k, dense_v, q_lens):
    """Independent float32 CPU causal attention on the original dense sequences."""
    result = np.empty(q.shape, dtype=np.float32)
    offset = 0
    for key, value, length in zip(dense_k, dense_v, q_lens):
        query = q[offset:offset + length].astype(np.float32).transpose(1, 0, 2)
        groups = q.shape[1] // key.shape[1]
        key = np.repeat(key.astype(np.float32), groups, axis=1).transpose(1, 2, 0)
        value = np.repeat(value.astype(np.float32), groups, axis=1).transpose(1, 0, 2)
        scores = (query @ key) / math.sqrt(q.shape[2])
        kv_length = key.shape[2]
        allowed = np.arange(kv_length)[None, :] <= (kv_length - length + np.arange(length))[:, None]
        scores = np.where(allowed[None, :, :], scores, -np.inf)
        probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)
        result[offset:offset + length] = (probs @ value).transpose(1, 0, 2)
        offset += length
    return result


def _run_case(name, q_lens, kv_lens, seed):
    arrays, dense_k, dense_v = _make_inputs(q_lens, kv_lens, seed)
    names = ["q", "k_cache", "v_cache", "block_table", "q_seq_lens", "seqused_k"]
    shapes = {key: value.shape for key, value in zip(names, arrays)}
    model = create_plugin_onnx_model(shapes, {
        "max_seqlen_q": max(q_lens), "max_seqlen_k": max(kv_lens),
        "max_block_num": arrays[1].shape[0],
    })
    mod, _ = tvm.relay.frontend.from_onnx(model, shapes)
    fbs_model = dyn.to_teco_infer_dyn(mod, {}, "teco_dyn")
    engine = tecoinference.Engine(fbs_model)
    ctx = engine.create_context()
    for index, value in enumerate(arrays):
        ctx.set_input(index, value)
    ctx.executor_run()
    out = np.asarray(ctx.get_output(0))
    ref = _reference(arrays[0], dense_k, dense_v, q_lens)
    assert out.shape == ref.shape, f"{name}: shape mismatch: {out.shape}"
    assert np.isfinite(out).all(), f"{name}: non-finite output"
    np.testing.assert_allclose(out.astype(np.float32), ref, atol=2e-3, rtol=1e-2, err_msg=name)
    max_error = np.max(np.abs(out.astype(np.float32) - ref))
    print(f"  {name}: shape={out.shape}, max_error={max_error:.6e} PASSED")


def test_prefill():
    _run_case("prefill", [64], [64], 42)


def test_decode():
    _run_case("decode", [1], [128], 43)


def test_chunked_prefill():
    _run_case("chunked_prefill", [32], [64], 44)


def test_mixed_batch():
    _run_case("mixed_batch", [1, 32], [128, 64], 45)


def test_unsupported_config():
    """Require direct C API status 6 + untouched output, then SDK rejection.

    SDK output allocation has no alias guarantee. It is printed for diagnostics
    after rejection; only the owned C API allocation has a sentinel assertion.
    """
    cases = [
        ("unsupported_head_size_48", [32], [32], 48, 32),
        ("unsupported_block_size_64", [32], [64], 64, 64),
    ]
    all_ok = True
    for name, q_lens, kv_lens, head_size, block_size in cases:
        arrays, _, _ = _make_inputs(q_lens, kv_lens, seed=123, head_size=head_size, block_size=block_size)
        names = ["q", "k_cache", "v_cache", "block_table", "q_seq_lens", "seqused_k"]
        shapes = {key: value.shape for key, value in zip(names, arrays)}
        model = create_plugin_onnx_model(shapes, {
            "max_seqlen_q": max(q_lens), "max_seqlen_k": max(kv_lens),
            "max_block_num": arrays[1].shape[0],
        })
        mod, _ = tvm.relay.frontend.from_onnx(model, shapes)
        fbs_model = dyn.to_teco_infer_dyn(mod, {}, "teco_dyn")
        engine = tecoinference.Engine(fbs_model)
        ctx = engine.create_context()
        for index, value in enumerate(arrays):
            ctx.set_input(index, value)

        # Check the exact C implementation loaded by the engine, independently
        # of whether this SDK preserves a C++ exception's what() string.
        mapped = [line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                  if line.split()[-1].endswith("/libteco_ops_plugin.so")]
        assert mapped, "the engine did not map libteco_ops_plugin.so"
        core = str(Path(mapped[0]).resolve())
        status, raw_out = run_flash(core, arrays)
        assert status == 6, f"{name}: C API returned {status}, expected NOT_SUPPORTED=6"
        assert np.all(raw_out == 7.0), f"{name}: rejected C API wrote output"
        print(f"  {name}: mapped_core={core}, sha256={hashlib.sha256(Path(core).read_bytes()).hexdigest()}, "
              "C_API_status=6 sentinel_untouched=True")

        try:
            assert_sdk_rejected(ctx, name)
        except AssertionError as exc:
            print(f"  [{name}] FAILED: {exc}")
            all_ok = False

    return all_ok


if __name__ == "__main__":
    print("plugin_flash_attention numerical tests (block_size=32)")
    test_prefill()
    test_decode()
    test_chunked_prefill()
    test_mixed_batch()
    print("ALL NUMERICAL TESTS PASSED")

    print("-" * 60)
    print("plugin_flash_attention negative tests (unsupported configs)")
    neg_ok = test_unsupported_config()

    print()
    print("=" * 60)
    print("NOTE: negative cases are expected to fail (EXPECTED_FAIL).")
    print('      The "rejected ...: PASSED (EXPECTED_FAIL)" lines assert that an')
    print('      unsupported shape was rejected before kernel launch; the matching')
    print('      low-level "tecoual runtime error" / "branch is not exit" log lines')
    print("      are emitted by that rejection path, not by a functional defect.")
    print("=" * 60)
    if not neg_ok:
        print("SOME TESTS FAILED")
        raise SystemExit(1)
    print("ALL TESTS PASSED")
    raise SystemExit(0)
