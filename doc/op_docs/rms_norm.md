# tecoopsRmsNorm 设计文档

## 计算原理

RMS Normalization（RMSNorm）是一种用于神经网络训练的归一化方法。与 LayerNorm 不同，RMSNorm 不计算均值，仅使用均方根（Root Mean Square）进行归一化，计算量更小。

本算子同时支持 **纯 RMSNorm** 和 **融合 residual add 的 RMSNorm** 两种模式。

**纯 RMSNorm 计算公式：**

$$
\begin{aligned}
\text{rms}(x) &= \sqrt{\frac{1}{D}\sum_{i=1}^{D} x_i^2 + \epsilon} \\
\text{rstd} &= \frac{1}{\text{rms}(x)} \\
y_i &= x_i \cdot \text{rstd} \cdot w_i
\end{aligned}
$$

**融合 residual add 的 RMSNorm 计算公式：**

$$
\begin{aligned}
x'_i &= x_i + r_i \quad (\text{residual add}) \\
\text{rms}(x') &= \sqrt{\frac{1}{D}\sum_{i=1}^{D} x_i'^2 + \epsilon} \\
\text{rstd} &= \frac{1}{\text{rms}(x')} \\
y_i &= x_i' \cdot \text{rstd} \cdot w_i
\end{aligned}
$$

**参数解释：**

- $x$：输入张量，形状 `[num_tokens, hidden_size]`
- $w$：权重张量，形状 `[hidden_size]`
- $r$：残差输入（可选），形状 `[num_tokens, hidden_size]`
- $\epsilon$：数值稳定性参数，典型值 `1e-5` 或 `1e-6`
- $y$：输出张量，形状 `[num_tokens, hidden_size]`
- $r_{out}$：残差输出（可选，当有 residual 时输出 $x'$），形状 `[num_tokens, hidden_size]`

## 功能实现

### 接口设计

参考 PyTorch `RMSNorm` 实现及 LLM 推理中 fused residual add 的需求，设计 userAPI 接口：

```c++
tecoopsStatus_t tecoopsRmsNorm(
    tecoopsHandle_t handle,
    const void *input,
    const void *weight,
    const void *residual,
    void *output,
    void *residual_out,
    int num_tokens,
    int hidden_size,
    float eps);
```

### 参数信息

其中，各参数含义如下：

| 参数         | 输入/输出 | 主机端/设备端 | 说明                                        |
| ------------ | --------- | ------------- | ------------------------------------------- |
| handle       | 输入      | 主机端        | Teco-Ops 句柄，管理设备上下文               |
| input        | 输入      | 设备端        | 输入张量，形状`[num_tokens, hidden_size]` |
| weight       | 输入      | 设备端        | 权重张量，形状`[hidden_size]`             |
| residual     | 输入      | 设备端        | 残差输入（可为`nullptr`，表示纯 RMSNorm） |
| output       | 输出      | 设备端        | 输出张量，形状`[num_tokens, hidden_size]` |
| residual_out | 输出      | 设备端        | 残差输出（`residual != nullptr` 时有效）  |
| num_tokens   | 输入      | 主机端        | token 数量                                  |
| hidden_size  | 输入      | 主机端        | 隐藏层维度                                  |
| eps          | 输入      | 主机端        | 数值稳定性参数                              |

### 类型限制

当前计算分支，主要完成以下功能实现，其余情况暂不支持。

| 参数         | 数据类型 | 维度信息                                     | 存储格式 |
| ------------ | -------- | -------------------------------------------- | -------- |
| input        | fp16     | `[num_tokens, hidden_size]`                | NCHW     |
| weight       | fp16     | `[hidden_size]`                            | Array    |
| residual     | fp16     | `[num_tokens, hidden_size]` 或 `nullptr` | NCHW     |
| output       | fp16     | `[num_tokens, hidden_size]`                | NCHW     |
| residual_out | fp16     | `[num_tokens, hidden_size]` 或 `nullptr` | NCHW     |
| num_tokens   | int      | 标量，`> 0`                                | -        |
| hidden_size  | int      | 标量，`> 0`                                | -        |
| eps          | float    | 标量，`> 0`                                | -        |

## 性能优化

### 多核并行划分

使用 Hal Tile `R1C32_CR` 模式进行行并行划分，将 `num_tokens` 行均匀分配到各 SPE 核心。每个核心处理 `my_rows` 行，通过 `tile.compute_linear_index()` 计算 HBM 偏移量。

### 双缓冲流水设计

采用双缓冲流水线，使用 `MemcpyHandle` 实现 Load / Compute / Store 之间的重叠：

```
SPM 布局:
  in0 / in1  — 双缓冲输入行（交替使用）
  out0 / out1 — 双缓冲输出行（交替使用）
  weight     — 权重（只加载一次）
```

流水循环流程：

1. 发起下一行 Load（`memcpy_async` + `get_handle`），与计算重叠
2. 等待当前行 Load 完成（`memcpy_wait(get_handle)`）
3. 等待上一轮 Store 完成（`memcpy_wait(put_handle)`），释放输出缓冲区
4. 计算当前行（sum_sq + rstd + element-wise mul）
5. 发起当前行 Store（`memcpy_async` + `put_handle`）
6. 交换双缓冲标志

### 性能数据

| 测例 | 配置 | eps | 状态 |
|---|---|---|---|
| case_0 | 64×4096, 纯 norm | 1e-6 | OK |
| case_1 | 64×4096, add norm | 1e-6 | OK |
| case_2 | 64×2048, 纯 norm | 1e-5 | OK |
| case_3 | 64×2048, add norm | 1e-5 | OK |
| case_4 | 64×1536, 纯 norm | 1e-6 | OK |
| case_5 | 64×1536, add norm | 1e-6 | OK |
| case_6 | 64×128, 纯 norm | 1e-6 | OK |
| case_7 | 64×128, 纯 norm | 1e-5 | OK |

## 分支派发

| 算法取值                 | 计算分支                         | 含义说明                           |
| ------------------------ | -------------------------------- | ---------------------------------- |
| `branch=0`（自动派发） | `teco_slave_rms_norm_fp16`     | 纯 RMSNorm，双缓冲流水             |
| `branch=1`（自动派发） | `teco_slave_rms_norm_fp16_add` | RMSNorm + residual add，双缓冲流水 |

> 注：本算子不通过 `tecoopsAlgo_t` 参数选择分支，而是根据 `residual` 是否为 `nullptr` 自动派发。`residual == nullptr` → pure RMSNorm，`residual != nullptr` → add 模式。

## 文件结构

```
teco/
├── interface/
│   ├── include/tecoops.h              # userAPI 声明
│   └── ops/rms_norm.cpp               # 接口实现（参数组装 + RUN_OP 分发）
├── ual/
│   ├── args/rms_norm_args.h           # 参数结构体（RmsNormArgs / RmsNormPatchArgs）
│   ├── ops/rms_norm/
│   │   ├── rms_norm.hpp               # Op 类定义（RmsNormOp + RmsNormAlgos）
│   │   ├── find_rms_norm.cpp          # 分支选择（根据 has_residual 派发）
│   │   └── find_rms_norm.h
│   └── kernel/rms_norm/
│       ├── rms_norm.h                 # kernel 声明
│       └── rms_norm_fp16.scpp         # fp16 kernel 实现（双缓冲流水）
├── plugin/
│   └── pluginRmsNorm/
│       └── plugin_rms_norm.cc         # Plugin 自定义算子（Teco-Inference 推理框架）
test/
├── test_proto/
│   ├── tecokernel/rms_norm.proto      # Proto 参数定义
│   └── tecokernel.proto               # 注册 TecokernelParam
└── zoo/teco/rms_norm/
    ├── rms_norm.h                     # 测试类声明
    ├── rms_norm.cpp                   # 测试实现 + CPU baseline
    └── test_case/
        ├── case_0.prototxt            # 64×4096, 纯 norm, eps=1e-6
        ├── case_1.prototxt            # 64×4096, add norm, eps=1e-6
        ├── case_2.prototxt            # 64×2048, 纯 norm, eps=1e-5
        ├── case_3.prototxt            # 64×2048, add norm, eps=1e-5
        ├── case_4.prototxt            # 64×1536, 纯 norm, eps=1e-6
        ├── case_5.prototxt            # 64×1536, add norm, eps=1e-6
        ├── case_6.prototxt            # 64×128, 纯 norm, eps=1e-6
        └── case_7.prototxt            # 64×128, 纯 norm, eps=1e-5
api/
├── torch_ext.cpp                      # PyTorch 绑定
└── tecoops/__init__.py                # Python 导出
python_api_test/
└── test_rms_norm.py                   # Python API 精度测试
plugin_test/
└── test_plugin_rms_norm.py            # Plugin 推理精度测试
```

## 使用示例

```python
import torch
import tecoops

# 纯 RMSNorm
x = torch.randn(64, 4096, dtype=torch.half, device='sdaa')
w = torch.randn(4096, dtype=torch.half, device='sdaa')
out = torch.empty(64, 4096, dtype=torch.half, device='sdaa')
tecoops.rms_norm(x, w, None, out, None, eps=1e-6)

# RMSNorm + residual add
residual = torch.randn(64, 4096, dtype=torch.half, device='sdaa')
res_out = torch.empty(64, 4096, dtype=torch.half, device='sdaa')
tecoops.rms_norm(x, w, residual, out, res_out, eps=1e-6)
```
