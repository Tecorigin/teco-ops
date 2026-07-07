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
import torch
import tecoops


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


def test_rms_norm():
    """纯 RMSNorm 测试"""
    print("=" * 60)
    print("Test: RMSNorm (pure)")
    
    num_tokens, hidden_size = 64, 4096
    eps = 1e-5
    
    x_cpu = torch.randn(num_tokens, hidden_size, dtype=torch.half)
    w_cpu = torch.randn(hidden_size, dtype=torch.half)
    
    # SDAA tensors
    x = x_cpu.to("sdaa")
    weight = w_cpu.to("sdaa")
    out = torch.empty(num_tokens, hidden_size, dtype=torch.half, device="sdaa")
    
    # SDAA compute
    tecoops.rms_norm(x, weight, None, out, None, eps)
    
    # CPU reference
    out_ref = rms_norm_ref(x_cpu, w_cpu, eps)
    
    max_err = (out.cpu().float() - out_ref.float()).abs().max().item()
    passed = max_err < 5e-3
    print(f"  max_error = {max_err:.6e}  {'PASSED' if passed else 'FAILED'}")
    return passed


def test_rms_norm_add():
    """RMSNorm + residual add 测试"""
    print("=" * 60)
    print("Test: RMSNorm + residual add")
    
    num_tokens, hidden_size = 64, 4096
    eps = 1e-5
    
    x_cpu = torch.randn(num_tokens, hidden_size, dtype=torch.half)
    r_cpu = torch.randn(num_tokens, hidden_size, dtype=torch.half)
    w_cpu = torch.randn(hidden_size, dtype=torch.half)
    
    # SDAA tensors
    x = x_cpu.to("sdaa")
    residual = r_cpu.to("sdaa")
    weight = w_cpu.to("sdaa")
    out = torch.empty(num_tokens, hidden_size, dtype=torch.half, device="sdaa")
    residual_out = torch.empty(num_tokens, hidden_size, dtype=torch.half, device="sdaa")
    
    # SDAA compute
    tecoops.rms_norm(x, weight, residual, out, residual_out, eps)
    
    # CPU reference
    out_ref, res_out_ref = rms_norm_add_ref(x_cpu, r_cpu, w_cpu, eps)
    
    max_err_out = (out.cpu().float() - out_ref.float()).abs().max().item()
    max_err_res = (residual_out.cpu().float() - res_out_ref.float()).abs().max().item()
    passed = max_err_out < 1e-2 and max_err_res < 1e-2
    print(f"  output       max_error = {max_err_out:.6e}")
    print(f"  residual_out max_error = {max_err_res:.6e}")
    print(f"  {'PASSED' if passed else 'FAILED'}")
    return passed


if __name__ == "__main__":
    r1 = test_rms_norm()
    r2 = test_rms_norm_add()
    print()
    print("ALL PASSED" if (r1 and r2) else "SOME FAILED")
