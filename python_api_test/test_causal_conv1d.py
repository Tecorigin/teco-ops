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

import torch
import torch.nn.functional as F
import _bootstrap
import tecoops


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


def _test_one_case(dim, total_seqlen, width, num_entries,
                   seq_lens, cache_indices, has_init, pad_slot_id=-1):
    """单个 causal_conv1d 测试"""
    batch = len(seq_lens)

    # 构造 query_start_loc
    qsl = [0]
    for sl in seq_lens:
        qsl.append(qsl[-1] + sl)
    query_start_loc = torch.tensor(qsl, dtype=torch.int32)

    cache_indices_t = torch.tensor(cache_indices, dtype=torch.int32)
    has_init_t = torch.tensor(has_init, dtype=torch.int8)

    # CPU 输入
    x_cpu = torch.randn(dim, total_seqlen, dtype=torch.half)
    w_cpu = torch.randn(width, dim, dtype=torch.half)
    conv_cpu = torch.randn(num_entries, width - 1, dim, dtype=torch.half)

    # SDAA 输入
    x = x_cpu.to("sdaa")
    weight = w_cpu.to("sdaa")
    conv_states = conv_cpu.clone().to("sdaa")
    qsl_sdaa = query_start_loc.to("sdaa")
    idx_sdaa = cache_indices_t.to("sdaa")
    init_sdaa = has_init_t.to("sdaa")
    out = torch.zeros(dim, total_seqlen, dtype=torch.half, device="sdaa")

    # 调用 SDAA 算子
    tecoops.causal_conv1d_fn_torch(
        out, conv_states, x, weight,
        qsl_sdaa, idx_sdaa, init_sdaa, pad_slot_id)

    # CPU 参考
    out_ref, conv_ref = causal_conv1d_ref(
        x_cpu.clone(), w_cpu, query_start_loc,
        cache_indices_t, has_init_t, conv_cpu.clone(), pad_slot_id)

    max_err_out = (out.cpu().float() - out_ref.float()).abs().max().item()
    max_err_conv = (conv_states.cpu().float() - conv_ref.float()).abs().max().item()
    return max_err_out, max_err_conv


def test_causal_conv1d():
    """基本 causal_conv1d 精度测试"""
    print("=" * 60)
    print("Test: causal_conv1d")
    all_passed = True

    cases = [
        # (dim, total_seqlen, width, num_entries, [seq_lens], cache_indices, [has_init])
        (2048,   8, 4, 1, [8],         [0],   [0]),     # case 0: 单batch, 无初态
        (1280, 512, 4, 2, [256, 256], [0, 1], [0, 0]),  # case 1: 双batch, 无初态
        (2048,   8, 4, 1, [8],         [0],   [1]),     # case 2: 单batch, 有初态
    ]

    for dim, total_seqlen, width, num_entries, seq_lens, cache_indices, has_init in cases:
        max_err_out, max_err_conv = _test_one_case(
            dim, total_seqlen, width, num_entries, seq_lens, cache_indices, has_init)

        passed = max_err_out < 2e-2 and max_err_conv < 2e-2
        all_passed = all_passed and passed
        pad_str = f"dim={dim:4d} seq={total_seqlen:4d} w={width} batch={len(seq_lens)}"
        print(f"  {pad_str}  out_err={max_err_out:.6e}  conv_err={max_err_conv:.6e}  atol=2.000e-02  (reference: CPU fp32 conv1d+SiLU via torch.nn.functional, fp16 inputs)  {'PASSED' if passed else 'FAILED'}")

    return all_passed


def test_causal_conv1d_pad():
    """测试 pad_slot_id 跳过逻辑"""
    print("=" * 60)
    print("Test: causal_conv1d pad skip")
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
        print(f"  dim={dim:4d} seq={total_seqlen:4d} w={width} batch={len(seq_lens)} pad_skip"
              f"  out_err={max_err_out:.6e}  conv_err={max_err_conv:.6e}  atol=2.000e-02  (reference: CPU fp32 conv1d+SiLU via torch.nn.functional, fp16 inputs)  {'PASSED' if passed else 'FAILED'}")

    return all_passed


def test_causal_conv1d_misaligned():
    """非 16 对齐 shape 必须在参数层 fail-closed，而不是在 kernel 内 abort

    16-lane 向量主循环以 32 B 对齐的 simd_load/simd_store 访问
    {t*x_seq_stride, t*out_seq_stride, k*dim, w*dim} + 16*blk，因此要求 dim 与
    x_seq_stride / out_seq_stride / conv_state_stride 全部是 16 half（32 B）的倍数。
    在加这道守卫之前，dim=2047 会在 warmup 阶段 abort 整个进程
    （SDAA_ERROR_MISALIGNED_ADDRESS，rc=134）；现在应由 C 入口返回
    TECOOPS_STATUS_BAD_PARAM，绑定层转成 RuntimeError，且**不写任何输出**。
    所以这里同时断言输出与 conv_state 的哨兵值未被改动：失败必须是"拒绝"，
    不能是"跑了但写坏"。
    """
    print("=" * 60)
    print("Test: causal_conv1d non-16-aligned rejection (fail-closed)")
    dim, total_seqlen, width, num_entries = 2047, 16, 4, 2
    sentinel = 7.0

    qsl = torch.tensor([0, 6, 16], dtype=torch.int32, device="sdaa")
    cache_indices = torch.tensor([0, 1], dtype=torch.int32, device="sdaa")
    has_init = torch.tensor([0, 0], dtype=torch.int8, device="sdaa")

    x = torch.randn(dim, total_seqlen, dtype=torch.half, device="sdaa")
    weight = torch.randn(width, dim, dtype=torch.half, device="sdaa")
    conv_states = torch.randn(num_entries, width - 1, dim, dtype=torch.half, device="sdaa")
    conv_before = conv_states.clone()
    out = torch.full((dim, total_seqlen), sentinel, dtype=torch.half, device="sdaa")

    raised = False
    try:
        tecoops.causal_conv1d_fn_torch(
            out, conv_states, x, weight, qsl, cache_indices, has_init, -1)
    except RuntimeError as err:
        raised = True
        print(f"  dim={dim} (dim % 16 != 0) -> RuntimeError: {err}")

    out_kept = bool(torch.all(out == sentinel).item())
    state_kept = bool(torch.equal(conv_states, conv_before))
    passed = raised and out_kept and state_kept
    print(f"  raised={raised}  out_sentinel_kept={out_kept}  conv_state_kept={state_kept}"
          f"  {'PASSED' if passed else 'FAILED'}")
    return passed


if __name__ == "__main__":
    r1 = test_causal_conv1d()
    r2 = test_causal_conv1d_pad()
    r3 = test_causal_conv1d_misaligned()
    print()
    print("ALL PASSED" if (r1 and r2 and r3) else "SOME FAILED")
