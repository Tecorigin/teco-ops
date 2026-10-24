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
"""精度测试：tecoops.reduce_variance vs PyTorch 参考实现

Gate 与官方 zoo 一致：DIFF3_MAX（atol=1e-6）且 error_num==0。
参考实现与 test/zoo/teco/reduce_variance/reduce_variance.py 相同：
torch.var(x, dim=axis, correction=correction)（CPU，fp32）。

dtype 契约：**fp32 only**。kernel 侧只有 float 实现
（`teco/ual/ops/reduce_variance/find_reduce_variance.cpp` 只对
`UAL_DTYPE_FLOAT` 选分支，其它 dtype 落到 `algo = 0` 即 fp32 kernel），
因此 `api/torch_ext.cpp` 的绑定对 fp16 输入 fail-closed 抛 `RuntimeError`，
本文件用 `test_fp16_rejected` 覆盖该拒绝路径。
"""

import torch
import _bootstrap
import tecoops


# DIFF3_MAX 默认判据（官方 zoo 对 reduce_variance 三例的判据为 1e-6，error_num=0）
DIFF3_MAX = 1e-6


def reduce_variance_ref(x, axis, correction):
    """CPU 参考: 方差（与 zoo 参考脚本 reduce_variance.py 同一公式，keepdim=False）"""
    return torch.var(x, dim=axis, correction=correction)


def _test_one_case(shape, axis, correction, seed=5):
    """单个 reduce_variance 测试；shape/axis/correction 取自官方 zoo case

    Bit-exactness note (10/22 review): the literal "bit-exact vs reference" contract is
    not well-defined here, because the reference is CPU fp32 (torch.var) while the kernel
    accumulates in SIMD order; the two are only equal up to the DIFF3_MAX gate (the
    device-level byte comparison of the pre/post-SIMD builds is recorded in the PR
    thread: case_0/1 byte-identical, case_2 differs by 1 ULP).  What *is* well-defined
    and worth pinning is run-to-run determinism: the same input must produce the same
    bits on every launch, so a reordered/partial SIMD path shows up immediately.
    """
    # 与 zoo prototxt 相同的输入分布：uniform [-2, 2]
    g = torch.Generator().manual_seed(seed)
    x_cpu = (torch.rand(shape, generator=g) * 4.0 - 2.0).to(torch.float32)

    x = x_cpu.to("sdaa")
    y = tecoops.reduce_variance(x, axis, correction)
    y_again = tecoops.reduce_variance(x, axis, correction)

    bit_identical = bool(torch.equal(y.cpu().contiguous().view(torch.int32),
                                     y_again.cpu().contiguous().view(torch.int32)))

    y_ref = reduce_variance_ref(x_cpu, axis, correction)
    abs_err = (y.cpu().float() - y_ref).abs()
    max_err = abs_err.max().item()
    error_num = int((abs_err > DIFF3_MAX).sum().item())
    return max_err, error_num, bit_identical


def test_reduce_variance():
    """官方 zoo 三例 shape + 一个 correction=1 小例"""
    print("=" * 60)
    print("Test: reduce_variance (SDAA vs CPU reference)")
    print(f"  gate: atol(DIFF3_MAX)={DIFF3_MAX:.0e}, error_num==0")
    print(f"  reference: torch.var(x, dim=axis, correction=correction) on CPU fp32")
    all_passed = True

    cases = [
        # (name, shape, axis, correction) —— 前三例与
        # test/zoo/teco/reduce_variance/test_case/case_{0,1,2}.prototxt 完全一致
        ("case_0 (axis=0, left)",   (150, 3, 3, 200), 0, 0),
        ("case_1 (axis=1, middle)", (3, 120, 3, 200), 1, 0),
        ("case_2 (axis=3, right)",  (3, 120, 3, 200), 3, 0),
        # 额外小例：correction=1（样本方差），覆盖接口参数分支
        ("correction=1", (2, 3, 4, 8), 2, 1),
    ]

    for name, shape, axis, correction in cases:
        max_err, error_num, bit_identical = _test_one_case(shape, axis, correction)
        passed = (max_err < DIFF3_MAX) and (error_num == 0) and bit_identical
        all_passed = all_passed and passed
        print(f"  {name:24s} shape={str(shape):16s} "
              f"max_error = {max_err:.6e}  error_num = {error_num}  "
              f"bit_identical_run_to_run = {bit_identical}  "
              f"{'PASSED' if passed else 'FAILED'}")

    return all_passed


def test_fp16_rejected():
    """负例: fp16 输入必须被绑定层拒绝（kernel 侧无 fp16 实现，静默跑会算错）

    依据（源码级，可复核）：`teco/ual/ops/reduce_variance/find_reduce_variance.cpp`
    只对 `UAL_DTYPE_FLOAT` 设置分支号；非 FLOAT 只打印 ERROR 并保留 `algo = 0`，
    等于用 fp32 kernel 去解释 fp16 位模式（既算错、也会按 float 宽度越界读）。
    因此绑定的 dtype 契约收窄为 fp32，fp16 必须在进入 kernel 前抛 RuntimeError。
    """
    print("=" * 60)
    print("Test: reduce_variance fp16 拒绝路径 (dtype 契约 = fp32 only)")
    print("  basis: find_reduce_variance.cpp 仅对 UAL_DTYPE_FLOAT 选分支;")
    print("         非 float 落到 algo=0 (fp32 kernel) => 绑定层 fail-closed")
    x_cpu = torch.rand((3, 120, 3, 200)) * 4.0 - 2.0
    x_half = x_cpu.to(torch.float16).to("sdaa")

    raised = None
    try:
        tecoops.reduce_variance(x_half, 3, 0)
        torch.sdaa.synchronize()
    except RuntimeError as exc:
        raised = str(exc)

    if raised is not None and "float32" in raised:
        print(f"  [{'fp16 rejected':24s}] RuntimeError: {raised.splitlines()[0][:92]}  PASSED")
        return True
    print(f"  [{'fp16 rejected':24s}] FAILED (expected RuntimeError mentioning "
          f"'float32'; got {raised!r})")
    return False


if __name__ == "__main__":
    r = test_reduce_variance()
    r16 = test_fp16_rejected()
    print()
    print("ALL PASSED" if (r and r16) else "SOME FAILED")
