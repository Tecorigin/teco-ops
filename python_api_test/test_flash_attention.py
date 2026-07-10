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
import torch
import torch_sdaa
import tecoops
import numpy as np

def test_prefill():
    """prefill: L == S"""
    print("  prefill (L=S=64)...", end=" ")
    B, H, KV, HS, BS = 1, 8, 4, 128, 32
    L = S = 64
    blk = (S + BS - 1) // BS
    bt = torch.zeros(1, blk, dtype=torch.int32, device='sdaa')
    q = torch.randn(L, H, HS, dtype=torch.float16, device='sdaa')
    kc = torch.randn(blk, KV, BS, HS, dtype=torch.float16, device='sdaa')
    vc = torch.randn(blk, KV, BS, HS, dtype=torch.float16, device='sdaa')
    q_seq_lens = torch.tensor([L], dtype=torch.int32, device='sdaa')  # [32]
    cu_seqlens_q = torch.zeros(len(q_seq_lens) + 1, dtype=torch.int32, device='sdaa')
    cu_seqlens_q[0] = 0
    for i in range(len(q_seq_lens)):
        cu_seqlens_q[i + 1] = cu_seqlens_q[i] + q_seq_lens[i]
    sk = torch.tensor([S], dtype=torch.int32, device='sdaa')
    out = torch.empty_like(q)
    print(f"  shapes: q={list(q.shape)} kc={list(kc.shape)} vc={list(vc.shape)} "
          f"bt={list(bt.shape)} cu={list(cu_seqlens_q.shape)} sk={list(sk.shape)}")

    tecoops.flash_attn_varlen_func(
        q, kc, vc, L, cu_seqlens_q, S,
        torch.Tensor(), sk, 0.0, True, torch.Tensor(), bt, False, out)


def test_decode():
    """decode: L=1, S large"""
    print("  decode (L=1, S=128)...", end=" ")
    B, H, KV, HS, BS = 1, 8, 4, 128, 32
    L, S = 1, 128
    blk = (S + BS - 1) // BS
    bt = torch.zeros(1, blk, dtype=torch.int32, device='sdaa')
    q = torch.randn(L, H, HS, dtype=torch.float16, device='sdaa')
    kc = torch.randn(blk, KV, BS, HS, dtype=torch.float16, device='sdaa')
    vc = torch.randn(blk, KV, BS, HS, dtype=torch.float16, device='sdaa')
    q_seq_lens = torch.tensor([L], dtype=torch.int32, device='sdaa')  # [32]
    cu_seqlens_q = torch.zeros(len(q_seq_lens) + 1, dtype=torch.int32, device='sdaa')
    cu_seqlens_q[0] = 0
    for i in range(len(q_seq_lens)):
        cu_seqlens_q[i + 1] = cu_seqlens_q[i] + q_seq_lens[i]
    sk = torch.tensor([S], dtype=torch.int32, device='sdaa')
    out = torch.empty_like(q)

    tecoops.flash_attn_varlen_func(
        q, kc, vc, L, cu_seqlens_q, S,
        torch.Tensor(), sk, 0.0, True, torch.Tensor(), bt, False, out)


def test_chunked_prefill():
    """chunked prefill: L < S"""
    print("  chunked prefill (L=32, S=64)...", end=" ")
    B, H, KV, HS, BS = 1, 8, 4, 128, 32
    L, S = 32, 64
    blk = (S + BS - 1) // BS
    bt = torch.zeros(1, blk, dtype=torch.int32, device='sdaa')
    q = torch.randn(L, H, HS, dtype=torch.float16, device='sdaa')
    kc = torch.randn(blk, KV, BS, HS, dtype=torch.float16, device='sdaa')
    vc = torch.randn(blk, KV, BS, HS, dtype=torch.float16, device='sdaa')
    q_seq_lens = torch.tensor([L], dtype=torch.int32, device='sdaa')  # [32]
    cu_seqlens_q = torch.zeros(len(q_seq_lens) + 1, dtype=torch.int32, device='sdaa')
    cu_seqlens_q[0] = 0
    for i in range(len(q_seq_lens)):
        cu_seqlens_q[i + 1] = cu_seqlens_q[i] + q_seq_lens[i]
    sk = torch.tensor([S], dtype=torch.int32, device='sdaa')
    out = torch.empty_like(q)

    tecoops.flash_attn_varlen_func(
        q, kc, vc, L, cu_seqlens_q, S,
        torch.Tensor(), sk, 0.0, True, torch.Tensor(), bt, False, out)


def test_multi_batch():
    """two batches: decode + prefill"""
    print("  multi-batch (decode+prefill)...", end=" ")
    H, KV, HS, BS = 8, 4, 128, 32
    seqs = [(1, 64), (32, 32)]
    total_q = sum(s[0] for s in seqs)
    max_k = max(s[1] for s in seqs)
    blk = (max_k + BS - 1) // BS
    bt = torch.zeros(len(seqs), blk, dtype=torch.int32, device='sdaa')
    q = torch.randn(total_q, H, HS, dtype=torch.float16, device='sdaa')
    kc = torch.randn(blk, KV, BS, HS, dtype=torch.float16, device='sdaa')
    vc = torch.randn(blk, KV, BS, HS, dtype=torch.float16, device='sdaa')
    cu_seqlens_q = torch.tensor([0, 1, 33], dtype=torch.int32, device='sdaa')
    sk = torch.tensor([64, 32], dtype=torch.int32, device='sdaa')
    out = torch.empty_like(q)

    tecoops.flash_attn_varlen_func(
        q, kc, vc, 32, cu_seqlens_q, 64,
        torch.Tensor(), sk, 0.0, True, torch.Tensor(), bt, False, out)


if __name__ == "__main__":
    print("=" * 60)
    print("flash_attn_varlen_func Test Suite")
    print("=" * 60)

    if not torch.sdaa.is_available():
        print("Warning: SDAA is not available, tests may fail")

    test_prefill()
    test_decode()
    test_chunked_prefill()
    test_multi_batch()

    print("=" * 60)
    print("All smoke tests passed (no crash = OK)")
    print("=" * 60)
