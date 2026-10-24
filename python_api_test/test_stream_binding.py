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
# CONSEQUENTIAL DAMAGES (INCLUDING BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.
"""Smoke test: the python bindings launch on the caller's current SDAA stream.

Scope note: this is a *smoke* test, not a race detector.  It checks that

  1. running the delivered bindings inside a non-default torch.sdaa.Stream
     context yields bit-identical results to running them on the default
     stream (same inputs, same kernels), and
  2. a producer that runs on the non-default stream is correctly consumed by a
     bound operator on that same stream, with a CPU reference for the value.

A binding that silently used the handle's default stream can pass (1) on an
idle queue (the two streams do not overlap there); only a genuinely concurrent
producer/consumer schedule would expose the evidence deterministically, and
that schedule is a race test, which is out of scope here.  What this file does
guarantee is that the current-stream code path is exercised end to end and does
not change any value.
"""
import sys

import torch
import torch_sdaa  # noqa: F401  (registers the PrivateUse1 SDAA device)
import _bootstrap
import tecoops


RMS_HIDDEN = 1536
RMS_TOKENS = 64
EPS = 1e-5
RMS_ATOL = 5e-3

RAC_TOKENS = 128
RAC_KV_HEADS = 4
RAC_HEAD_DIM = 128
RAC_BLOCK_SIZE = 16
RAC_NUM_BLOCKS = 16


def rms_norm_ref(x_cpu, w_cpu, eps=EPS):
    """CPU reference: RMSNorm (fp32 math, fp16 result), same as test_rms_norm.py."""
    rstd = 1.0 / torch.sqrt(torch.mean(x_cpu.float() ** 2, dim=-1, keepdim=True) + eps)
    return (x_cpu.float() * rstd * w_cpu.float()).half()


def make_rms_inputs(seed=0, num_tokens=RMS_TOKENS, hidden=RMS_HIDDEN):
    gen = torch.Generator().manual_seed(seed)
    x_cpu = torch.randn(num_tokens, hidden, dtype=torch.half, generator=gen)
    w_cpu = torch.randn(hidden, dtype=torch.half, generator=gen)
    return x_cpu, w_cpu


def make_rac_inputs(seed=0):
    gen = torch.Generator().manual_seed(seed)
    num_tokens, kv_heads, head_dim = RAC_TOKENS, RAC_KV_HEADS, RAC_HEAD_DIM
    block_size, num_blocks = RAC_BLOCK_SIZE, RAC_NUM_BLOCKS
    key_cpu = torch.randn(num_tokens, kv_heads, head_dim, dtype=torch.half, generator=gen)
    val_cpu = torch.randn(num_tokens, kv_heads, head_dim, dtype=torch.half, generator=gen)
    slots = torch.randperm(num_blocks * block_size, generator=gen)[:num_tokens]
    slot_cpu = slots.to(torch.int64)
    return key_cpu, val_cpu, slot_cpu, num_blocks, block_size


def run_rms_norm(x_cpu, w_cpu):
    """Enqueue rms_norm on the current stream, return the device output tensor."""
    x = x_cpu.to("sdaa")
    weight = w_cpu.to("sdaa")
    out = torch.empty(x_cpu.shape, dtype=torch.half, device="sdaa")
    tecoops.rms_norm(x, weight, None, out, None, EPS)
    return out


def run_reshape_and_cache(key_cpu, val_cpu, slot_cpu, num_blocks, block_size):
    """Enqueue reshape_and_cache on the current stream, return the device caches."""
    key = key_cpu.to("sdaa")
    value = val_cpu.to("sdaa")
    slot_mapping = slot_cpu.to("sdaa")
    kc = torch.zeros(num_blocks, RAC_KV_HEADS, block_size, RAC_HEAD_DIM,
                     dtype=torch.half, device="sdaa")
    vc = torch.zeros_like(kc)
    tecoops.reshape_and_cache(key, value, slot_mapping, kc, vc)
    return kc, vc


def _require_stream_api():
    if not hasattr(torch.sdaa, "Stream") or not hasattr(torch.sdaa, "stream"):
        raise RuntimeError("torch.sdaa Stream API is required for stream coverage")
    if not torch.sdaa.is_available():
        raise RuntimeError("SDAA device is required for stream coverage")


def test_non_default_stream_parity():
    """Non-default stream must produce bit-identical outputs to the default stream."""
    print("=" * 60)
    print("Test: non-default stream parity (rms_norm + reshape_and_cache)")
    print("=" * 60)
    _require_stream_api()

    x_cpu, w_cpu = make_rms_inputs(seed=0)
    key_cpu, val_cpu, slot_cpu, num_blocks, block_size = make_rac_inputs(seed=0)

    # --- default stream ---
    out_def = run_rms_norm(x_cpu, w_cpu)
    kc_def, vc_def = run_reshape_and_cache(key_cpu, val_cpu, slot_cpu, num_blocks, block_size)
    out_def_cpu = out_def.cpu()
    kc_def_cpu = kc_def.cpu()
    vc_def_cpu = vc_def.cpu()

    # --- non-default stream ---
    stream = torch.sdaa.Stream()
    with torch.sdaa.stream(stream):
        out_nd = run_rms_norm(x_cpu, w_cpu)
        kc_nd, vc_nd = run_reshape_and_cache(key_cpu, val_cpu, slot_cpu, num_blocks, block_size)
    stream.synchronize()
    out_nd_cpu = out_nd.cpu()
    kc_nd_cpu = kc_nd.cpu()
    vc_nd_cpu = vc_nd.cpu()

    ok = True
    for name, got, ref in (
        ("rms_norm out", out_nd_cpu, out_def_cpu),
        ("reshape_and_cache key_cache", kc_nd_cpu, kc_def_cpu),
        ("reshape_and_cache value_cache", vc_nd_cpu, vc_def_cpu),
    ):
        equal = torch.equal(got, ref)
        max_diff = (got.float() - ref.float()).abs().max().item()
        ok = ok and equal
        print(f"  {name:30s} bit_equal={equal}  max_abs_diff={max_diff:.6e}  "
              f"{'PASSED' if equal else 'FAILED'}")
    return ok


def test_producer_consumer_same_stream():
    """Producer on the non-default stream, bound operator consuming on the same stream."""
    print("=" * 60)
    print("Test: producer on non-default stream -> bound op on the same stream")
    print("=" * 60)
    _require_stream_api()

    x_cpu, w_cpu = make_rms_inputs(seed=1)
    scale = 3.0

    stream = torch.sdaa.Stream()
    with torch.sdaa.stream(stream):
        x_sdaa = x_cpu.to("sdaa")          # H2D copy on the non-default stream
        x_prod = x_sdaa * scale            # producer kernel on the same stream
        weight = w_cpu.to("sdaa")
        out = torch.empty(x_cpu.shape, dtype=torch.half, device="sdaa")
        tecoops.rms_norm(x_prod, weight, None, out, None, EPS)  # consumer, must share the stream
    stream.synchronize()

    # Reference through the same fp16 producer arithmetic.
    x_prod_cpu = (x_cpu * scale).half()
    ref = rms_norm_ref(x_prod_cpu, w_cpu)
    got = out.cpu()
    max_err = (got.float() - ref.float()).abs().max().item()
    passed = max_err < RMS_ATOL
    print(f"  producer/consumer id={stream} max_error={max_err:.6e}  "
          f"{'PASSED' if passed else 'FAILED'}")
    return passed


if __name__ == "__main__":
    p1 = test_non_default_stream_parity()
    p2 = test_producer_consumer_same_stream()
    print()
    if p1 and p2:
        print("ALL PASSED")
    else:
        print("SOME FAILED")
        sys.exit(1)
