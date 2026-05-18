"""
AXPY (y = alpha * x + y) 使用示例

需要安装 PyTorch 和 torch_sdaa:
    pip install torch torch_sdaa
"""
import torch
import torch_sdaa

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
        name="axpy_inline",
        cpp_sources=CPP_SRC,
        sdaa_sources=SDAA_SRC,
        functions=["my_axpy"],
        verbose=False,
    )
    my_axpy = axpy_module.my_axpy
    print("AXPY 内核编译加载成功！")

except Exception as e:
    print(f"加载 AXPY 内核失败: {e}")
    my_axpy = None


def axpy_example():
    """AXPY 计算示例: y = alpha * x + y"""

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
    print("Running AXPY on SDAA...")
    my_axpy(x_sdaa, y_sdaa, alpha)

    # 验证结果
    y_result = y_sdaa.cpu()

    if torch.allclose(y_ref, y_result, rtol=1e-5, atol=1e-5):
        print("✓ AXPY 计算正确!")
        print(f"  输入 x: {x_cpu[:5]}")
        print(f"  输入 y: {y_cpu[:5]}")
        print(f"  alpha: {alpha}")
        print(f"  结果 y = {alpha}*x + y: {y_result[:5]}")
        return True
    else:
        print("✗ AXPY 计算结果不匹配!")
        print(f"  参考结果: {y_ref[:5]}")
        print(f"  SDAA 结果: {y_result[:5]}")
        return False


if __name__ == "__main__":
    # 检查 my_axpy 是否可用
    if my_axpy is None:
        print("错误: my_axpy 不可用。请确保已安装 PyTorch 和 torch_sdaa。")
        exit(1)

    # 运行示例
    axpy_example()
