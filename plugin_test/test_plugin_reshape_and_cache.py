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
import numpy as np
from tvm.plugin import plugins
import tecoinference
from tvm.contrib.teco_infer_dyn import dyn
from onnx import helper, TensorProto
import tvm
from tvm import relay


plugins.register_op(
    op_name="plugin_reshape_and_cache",
    inputs=["key", "value", "slot_mapping"],
    attrs={"num_blocks": "int", "block_size": "int"}
)


def create_onnx_model(input_shapes, num_blocks, block_size):
    key_shape = input_shapes["key"]
    num_kv_heads = key_shape[1]
    head_size = key_shape[2]

    inputs = [
        helper.make_tensor_value_info("key", TensorProto.FLOAT16, key_shape),
        helper.make_tensor_value_info("value", TensorProto.FLOAT16, input_shapes["value"]),
        helper.make_tensor_value_info("slot_mapping", TensorProto.INT64, input_shapes["slot_mapping"]),
    ]

    cache_shape = [num_blocks * 2, num_kv_heads, block_size, head_size]
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT16, cache_shape)

    node = helper.make_node(
        "plugin_reshape_and_cache",
        ["key", "value", "slot_mapping"],
        ["output"],
        num_blocks=num_blocks,
        block_size=block_size,
        domain="my_custom_ops",
        version=1
    )

    graph = helper.make_graph([node], "reshape_and_cache", inputs, [output])
    model = helper.make_model(graph)
    model.opset_import.append(helper.make_opsetid("my_custom_ops", 1))
    return model


def reshape_and_cache_ref(key, value, num_blocks, block_size, slot_mapping):
    """CPU 参考: SDAA layout [num_blocks, num_kv_heads, block_size, head_size]"""
    num_tokens, num_kv_heads, head_size = key.shape
    kc = np.zeros((num_blocks, num_kv_heads, block_size, head_size), dtype=np.float16)
    vc = np.zeros((num_blocks, num_kv_heads, block_size, head_size), dtype=np.float16)

    for i in range(num_tokens):
        slot_idx = int(slot_mapping[i])
        if slot_idx < 0:
            continue
        blk = slot_idx // block_size
        off = slot_idx % block_size
        kc[blk, :, off, :] = key[i]
        vc[blk, :, off, :] = value[i]
    return kc, vc


def test_reshape_and_cache():
    print("Testing plugin_reshape_and_cache...")

    num_tokens, num_kv_heads, head_size = 4, 2, 8
    num_blocks, block_size = 4, 2

    key = np.random.randn(num_tokens, num_kv_heads, head_size).astype(np.float16)
    value = np.random.randn(num_tokens, num_kv_heads, head_size).astype(np.float16)
    slot_mapping = np.array([0, 1, 3, 6], dtype=np.int64)

    expected_kc, _ = reshape_and_cache_ref(key, value, num_blocks, block_size, slot_mapping)

    model = create_onnx_model(
        input_shapes={
            "key": (num_tokens, num_kv_heads, head_size),
            "value": (num_tokens, num_kv_heads, head_size),
            "slot_mapping": (num_tokens,),
        },
        num_blocks=num_blocks,
        block_size=block_size,
    )

    mod, _ = tvm.relay.frontend.from_onnx(model, {
        "key": (num_tokens, num_kv_heads, head_size),
        "value": (num_tokens, num_kv_heads, head_size),
        "slot_mapping": (num_tokens,),
    })
    print("IR:", mod)

    fbs_model = dyn.to_teco_infer_dyn(mod, {}, "teco_dyn")
    engine = tecoinference.Engine(fbs_model)
    ctx = engine.create_context()
    ctx.set_input(0, key)
    ctx.set_input(1, value)
    ctx.set_input(2, slot_mapping)
    ctx.executor_run()
    kc_out = ctx.get_output(0)
    # 前一半是 key_cache
    kc_out = kc_out[:num_blocks]

    ok = np.allclose(kc_out, expected_kc, atol=0.01)
    print(f"  key_cache: {'OK' if ok else 'MISMATCH'}")
    if not ok:
        diff = np.abs(kc_out.astype(np.float32) - expected_kc.astype(np.float32)).max()
        print(f"  max diff: {diff:.6f}")
    return ok


if __name__ == "__main__":
    print("=" * 60)
    print("plugin_reshape_and_cache Test")
    print("=" * 60)
    passed = test_reshape_and_cache()
    print("\n" + "=" * 60)
    print("All tests PASSED!" if passed else "FAILED!")
    print("=" * 60)
