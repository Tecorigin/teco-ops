#!/usr/bin/env python3
"""
Test script for AXPY function.

AXPY computes: y = alpha * x + y
This is a Level 1 BLAS operation (vector-vector operation).
"""

import torch
import torch_sdaa
import numpy as np

# 使用 load_inline 直接内嵌 SDAA 内核代码
try:
    from torch_sdaa.utils.cpp_extension import load_inline

    # SDAA 内核代码（直接内嵌，无需外部文件）
    SDAA_SRC = r'''
__global__ void axpy_kernel(float* x, float* y, float alpha, int total_num) {
    // 将所有数据切分为threadDim块，每块含有tile_num个数据，分给不同的SPE去计算
    uint32_t tile_num = total_num / threadDim;
    // 计算出tile_num个数据占用多少字节的数据空间
    uint32_t spm_size = tile_num * sizeof(float);
    // malloc spm
    float *spm_x = (float *)malloc(spm_size);
    float *spm_y = (float *)malloc(spm_size);
    // load
    memcpy(spm_x, (x + threadIdx * tile_num), spm_size);
    memcpy(spm_y, (y + threadIdx * tile_num), spm_size);

    // 每个计算核心SPE相隔tile_num个float数据，且循环tile_num次，进行AXPY计算: y = alpha*x + y
    for (uint32_t i = 0; i < tile_num; i++) {
        // compute: y[i] = alpha * x[i] + y[i]
        spm_y[i] = alpha * spm_x[i] + spm_y[i];
    }
    // store
    memcpy((y + threadIdx * tile_num), spm_y, spm_size);
}

void my_axpy(torch::Tensor& x, torch::Tensor& y, float alpha) {
    axpy_kernel<<<0>>>(x.data_ptr<float>(), y.data_ptr<float>(), alpha, x.numel());
}
'''

    # C++ 函数声明
    CPP_SRC = "void my_axpy(torch::Tensor& x, torch::Tensor& y, float alpha);"

    # 编译并加载扩展
    axpy_module = load_inline(
        name="axpy_test",
        cpp_sources=CPP_SRC,
        sdaa_sources=SDAA_SRC,
        functions=["my_axpy"],
        verbose=False,
    )
    my_axpy = axpy_module.my_axpy

except Exception as e:
    print(f"加载 AXPY 内核失败: {e}")
    my_axpy = None


def test_axpy_basic():
    """Test AXPY with basic example."""
    print("Testing AXPY basic...")

    # 设置参数
    alpha = 2.5
    n = 1024

    # 准备输入数据 (CPU)
    x_cpu = torch.rand([n], dtype=torch.float32)
    y_cpu = torch.rand([n], dtype=torch.float32)

    # CPU 参考计算
    y_ref = alpha * x_cpu + y_cpu

    # 将数据移动到 SDAA 设备
    x_sdaa = x_cpu.sdaa()
    y_sdaa = y_cpu.sdaa()

    # 调用 TECO AXPY 算子 (in-place 修改 y_sdaa)
    my_axpy(x_sdaa, y_sdaa, alpha)

    # 验证结果
    y_result = y_sdaa.cpu()

    if torch.allclose(y_ref, y_result, rtol=1e-5, atol=1e-5):
        print("✓ Basic test PASSED!")
        return True
    else:
        print("✗ Basic test FAILED!")
        max_diff = (y_ref - y_result).abs().max()
        print(f"  Max difference: {max_diff}")
        return False


def test_axpy_different_sizes():
    """Test AXPY with different vector sizes."""
    print("\nTesting AXPY with different sizes...")

    sizes = [256, 512, 1024, 2048, 4096]
    alpha = 1.5

    all_passed = True
    for n in sizes:
        x = torch.rand([n], dtype=torch.float32)
        y = torch.rand([n], dtype=torch.float32)

        # CPU reference
        y_ref = alpha * x + y

        # SDAA computation
        x_sdaa = x.sdaa()
        y_sdaa = y.sdaa()
        my_axpy(x_sdaa, y_sdaa, alpha)
        y_result = y_sdaa.cpu()

        if torch.allclose(y_ref, y_result, rtol=1e-5, atol=1e-5):
            print(f"  Size {n:5d}: PASSED")
        else:
            print(f"  Size {n:5d}: FAILED")
            all_passed = False

    return all_passed


def test_axpy_different_alpha():
    """Test AXPY with different alpha values."""
    print("\nTesting AXPY with different alpha values...")

    n = 1024
    alphas = [0.0, 0.5, 1.0, 2.0, -1.0, 3.14159]

    all_passed = True
    for alpha in alphas:
        x = torch.rand([n], dtype=torch.float32)
        y = torch.rand([n], dtype=torch.float32)

        # CPU reference
        y_ref = alpha * x + y

        # SDAA computation
        x_sdaa = x.sdaa()
        y_sdaa = y.sdaa()
        my_axpy(x_sdaa, y_sdaa, alpha)
        y_result = y_sdaa.cpu()

        if torch.allclose(y_ref, y_result, rtol=1e-5, atol=1e-5):
            print(f"  Alpha {alpha:8.4f}: PASSED")
        else:
            print(f"  Alpha {alpha:8.4f}: FAILED")
            all_passed = False

    return all_passed


def test_axpy_in_place():
    """Test that AXPY correctly modifies y in-place."""
    print("\nTesting AXPY in-place modification...")

    n = 512
    alpha = 2.0

    x = torch.rand([n], dtype=torch.float32)
    y_original = torch.rand([n], dtype=torch.float32)
    y_copy = y_original.clone()

    # Move to SDAA
    x_sdaa = x.sdaa()
    y_sdaa = y_original.sdaa()

    # Store original y value
    y_before = y_sdaa.cpu().clone()

    # Run AXPY
    my_axpy(x_sdaa, y_sdaa, alpha)

    # Get result
    y_after = y_sdaa.cpu()

    # Verify y was modified
    if not torch.equal(y_before, y_after):
        print("✓ In-place modification test PASSED!")
        return True
    else:
        print("✗ In-place modification test FAILED!")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("AXPY Test Suite")
    print("=" * 60)

    # Check if my_axpy is available
    if my_axpy is None:
        print("错误: my_axpy 不可用。请确保已安装 PyTorch 和 torch_sdaa。")
        exit(1)

    # Check if SDAA is available
    if not torch.sdaa.is_available():
        print("Warning: SDAA is not available, tests may fail")

    # Run tests
    test1_passed = test_axpy_basic()
    test2_passed = test_axpy_different_sizes()
    test3_passed = test_axpy_different_alpha()
    test4_passed = test_axpy_in_place()

    print("\n" + "=" * 60)
    if all([test1_passed, test2_passed, test3_passed, test4_passed]):
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)
