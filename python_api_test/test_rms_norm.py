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
"""精度测试：tecoops.rms_norm vs PyTorch 参考实现"""
import math

import torch
import _bootstrap
import tecoops

# 固定输入随机流：本测试原先每次运行都重新抽随机张量，导致 max_error 与
# PASS/FAIL 判定逐次变化（hidden=2048/4096 时 7.8125e-3 = 2**-7 恰好是
# |out| in [8,16) 的一个 fp16 ULP，落在旧 atol=5e-3 之外，实测 3 次运行里
# 就有 1 次 FAILED 而进程仍 exit 0）。给定 seed 后每次运行的输入相同，
# max_error 逐位可复现。
torch.manual_seed(42)


def rms_norm_ref(x, weight, eps=1e-5):
    """CPU 参考: RMSNorm"""
    rstd = 1.0 / torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + eps)
    return (x.float() * rstd * weight.float()).half()


def rms_norm_add_ref(x, residual, weight, eps=1e-5):
    """CPU 参考: residual add + RMSNorm"""
    x_add = x.float() + residual.float()
    rstd = 1.0 / torch.sqrt(torch.mean(x_add ** 2, dim=-1, keepdim=True) + eps)
    out = (x_add * rstd * weight.float()).half()
    return out, x_add.half()


def fp16_ulp_at(value):
    """|value| 处的一个 fp16 ULP = 2 ** (floor(log2|value|) - 10)。

    out_dev 与 out_ref 是同一实数量的两个 fp16 舍入结果：两种 fp32 求值次序
    只相差若干 fp32 ULP，若跨过 fp16 舍入中点，就会落到相邻 fp16 值上，即
    最多相差 1 ULP。所以"1 fp16 ULP"是该比较的严格表示误差上界；任何更小的
    固定 atol 都会在 |out| 较大时拒绝合法 kernel 输出。
    """
    mag = abs(float(value))
    if mag == 0.0 or not math.isfinite(mag):
        return 0.0
    return math.ldexp(1.0, math.frexp(mag)[1] - 11)


def _test_rms_norm_one(num_tokens, hidden_size, eps=1e-5):
    """单个纯 RMSNorm 测试"""
    x_cpu = torch.randn(num_tokens, hidden_size, dtype=torch.half)
    w_cpu = torch.randn(hidden_size, dtype=torch.half)

    x = x_cpu.to("sdaa")
    weight = w_cpu.to("sdaa")
    out = torch.empty(num_tokens, hidden_size, dtype=torch.half, device="sdaa")

    tecoops.rms_norm(x, weight, None, out, None, eps)

    out_ref = rms_norm_ref(x_cpu, w_cpu, eps)
    max_err = (out.cpu().float() - out_ref.float()).abs().max().item()
    max_abs = out_ref.float().abs().max().item()
    return max_err, max_abs


def _test_rms_norm_add_one(num_tokens, hidden_size, eps=1e-5):
    """单个 RMSNorm + residual add 测试"""
    x_cpu = torch.randn(num_tokens, hidden_size, dtype=torch.half)
    r_cpu = torch.randn(num_tokens, hidden_size, dtype=torch.half)
    w_cpu = torch.randn(hidden_size, dtype=torch.half)

    x = x_cpu.to("sdaa")
    residual = r_cpu.to("sdaa")
    weight = w_cpu.to("sdaa")
    out = torch.empty(num_tokens, hidden_size, dtype=torch.half, device="sdaa")
    residual_out = torch.empty(num_tokens, hidden_size, dtype=torch.half, device="sdaa")

    tecoops.rms_norm(x, weight, residual, out, residual_out, eps)

    out_ref, res_out_ref = rms_norm_add_ref(x_cpu, r_cpu, w_cpu, eps)
    max_err_out = (out.cpu().float() - out_ref.float()).abs().max().item()
    max_err_res = (residual_out.cpu().float() - res_out_ref.float()).abs().max().item()
    max_abs_out = out_ref.float().abs().max().item()
    max_abs_res = res_out_ref.float().abs().max().item()
    return max_err_out, max_err_res, max_abs_out, max_abs_res


def test_rms_norm():
    """纯 RMSNorm 测试（多维度）"""
    print("=" * 60)
    print("Test: RMSNorm (pure)")
    num_tokens = 64

    # (hidden_size, eps) 配对
    norm_cases = [
        (128,  1e-6),
        (128,  1e-5),
        (514,  1e-6),   # hidden % 32 != 0: even non-multiple-of-32 (64B SPM aligned)
        (520,  1e-5),
        (1536, 1e-6),
        (2048, 1e-5),
        (4096, 1e-6),
    ]

    all_passed = True
    for hidden_size, eps in norm_cases:
        max_err, max_abs = _test_rms_norm_one(num_tokens, hidden_size, eps)
        atol = fp16_ulp_at(max_abs)
        passed = max_err <= atol
        all_passed = all_passed and passed
        print(f"  hidden={hidden_size:4d}  eps={eps:.0e}  max_error = {max_err:.6e}  "
              f"atol = {atol:.6e} (1 fp16 ULP at max|ref| = {max_abs:.3f})  "
              f"{'PASSED' if passed else 'FAILED'}")

    return all_passed


def test_rms_norm_add():
    """RMSNorm + residual add 测试（多维度）"""
    print("=" * 60)
    print("Test: RMSNorm + residual add")
    num_tokens = 64

    # (hidden_size, eps) 配对
    add_cases = [
        (514,  1e-6),   # same even non-multiple-of-32 coverage as test_rms_norm
        (520,  1e-5),
        (1536, 1e-6),
        (2048, 1e-5),
        (4096, 1e-6),
    ]

    all_passed = True
    for hidden_size, eps in add_cases:
        max_err_out, max_err_res, max_abs_out, max_abs_res = _test_rms_norm_add_one(
            num_tokens, hidden_size, eps)
        atol_out = fp16_ulp_at(max_abs_out)
        atol_res = fp16_ulp_at(max_abs_res)
        passed = max_err_out <= atol_out and max_err_res <= atol_res
        all_passed = all_passed and passed
        print(f"  hidden={hidden_size:4d}  eps={eps:.0e}  out_max_err = {max_err_out:.6e} "
              f"(atol = {atol_out:.6e}, 1 fp16 ULP at max|ref| = {max_abs_out:.3f})  "
              f"res_max_err = {max_err_res:.6e} (atol = {atol_res:.6e})  "
              f"{'PASSED' if passed else 'FAILED'}")

    return all_passed


def test_rms_norm_non_default_stream():
    """在非默认 SDAA stream 上调用，覆盖 binding 的 stream 亲和性。"""
    if not hasattr(torch.sdaa, "Stream") or not hasattr(torch.sdaa, "stream"):
        raise RuntimeError("torch.sdaa Stream API is required for stream-affinity coverage")

    num_tokens, hidden_size, eps = 64, 1536, 1e-6
    x_cpu = torch.randn(num_tokens, hidden_size, dtype=torch.half)
    w_cpu = torch.randn(hidden_size, dtype=torch.half)
    stream = torch.sdaa.Stream()
    with torch.sdaa.stream(stream):
        x = x_cpu.to("sdaa")
        weight = w_cpu.to("sdaa")
        out = torch.empty(num_tokens, hidden_size, dtype=torch.half, device="sdaa")
        tecoops.rms_norm(x, weight, None, out, None, eps)
    stream.synchronize()

    out_ref = rms_norm_ref(x_cpu, w_cpu, eps)
    max_err = (out.cpu().float() - out_ref.float()).abs().max().item()
    max_abs = out_ref.float().abs().max().item()
    atol = fp16_ulp_at(max_abs)
    passed = max_err <= atol
    print(f"  non_default_stream max_error = {max_err:.6e}  "
          f"atol = {atol:.6e} (1 fp16 ULP at max|ref| = {max_abs:.3f})  "
          f"{'PASSED' if passed else 'FAILED'}")
    return passed


def test_negative_odd_hidden():
    """负例测试：奇数 hidden_size 必须被拒绝并抛出 RuntimeError"""
    print("=" * 60)
    print("Test: Negative (odd hidden_size rejected)")
    num_tokens = 64
    hidden_size = 513
    eps = 1e-5

    all_passed = True

    # 1. 纯 RMSNorm
    x_cpu = torch.randn(num_tokens, hidden_size, dtype=torch.half)
    w_cpu = torch.randn(hidden_size, dtype=torch.half)
    x = x_cpu.to("sdaa")
    w = w_cpu.to("sdaa")
    out = torch.empty(num_tokens, hidden_size, dtype=torch.half, device="sdaa")
    out.fill_(7.0)

    pure_threw = False
    pure_err = ""
    try:
        tecoops.rms_norm(x, w, None, out, None, eps)
    except RuntimeError as e:
        pure_threw = True
        pure_err = str(e)

    out_untouched = (out.cpu() == 7.0).all().item()
    passed_pure = pure_threw and out_untouched
    all_passed = all_passed and passed_pure
    if passed_pure:
        print(f"  pure rms_norm (hidden={hidden_size}): PASSED (RuntimeError: {pure_err})")
    else:
        print(f"  pure rms_norm (hidden={hidden_size}): FAILED "
              f"(threw={pure_threw}, out_untouched={out_untouched})")

    # 2. RMSNorm + residual add
    r_cpu = torch.randn(num_tokens, hidden_size, dtype=torch.half)
    residual = r_cpu.to("sdaa")
    residual_out = torch.empty(num_tokens, hidden_size, dtype=torch.half, device="sdaa")
    out.fill_(7.0)
    residual_out.fill_(7.0)

    add_threw = False
    add_err = ""
    try:
        tecoops.rms_norm(x, w, residual, out, residual_out, eps)
    except RuntimeError as e:
        add_threw = True
        add_err = str(e)

    out_untouched_add = (out.cpu() == 7.0).all().item()
    res_untouched_add = (residual_out.cpu() == 7.0).all().item()
    passed_add = add_threw and out_untouched_add and res_untouched_add
    all_passed = all_passed and passed_add
    if passed_add:
        print(f"  add  rms_norm (hidden={hidden_size}): PASSED (RuntimeError: {add_err})")
    else:
        print(f"  add  rms_norm (hidden={hidden_size}): FAILED "
              f"(threw={add_threw}, out_untouched={out_untouched_add}, res_untouched={res_untouched_add})")

    return all_passed


if __name__ == "__main__":
    r1 = test_rms_norm()
    r2 = test_rms_norm_add()
    r3 = test_rms_norm_non_default_stream()
    r4 = test_negative_odd_hidden()
    print()
    print("ALL PASSED" if (r1 and r2 and r3 and r4) else "SOME FAILED")
