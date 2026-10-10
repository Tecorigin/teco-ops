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

### PyTorch 当前流绑定

PyTorch 绑定在每次调用前将 Teco-Ops 句柄绑定到调用方当前的 SDAA stream，
使 RMSNorm 与同一 stream 上的输入生产和后续消费保持顺序，避免句柄默认流
引入的隐式串行化或跨流重排。该改动不改变 kernel 数学、输入输出布局或 API
签名；构建后的 C++/Python 正确性和同口径性能门禁仍需在厂商环境执行。

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

## FP16 SIMD epilogue 与 plain 输出 DMA

plain/add 的第二遍输出循环使用 `floatv16` 主循环和标量尾循环，保留
FP32 规约与逐元素数学顺序。PyTorch 入口绑定调用方当前 SDAA stream。

plain 输出 DMA 的 source buffer 持有到原有的同 buffer 复用 wait；
删除紧接 store 的 wait，在释放 SPM 前 drain 两个输出 handle。输入和
输出 buffer 独立，空闲 handle 初始化为 reply/counter=0/0。add 路径
保持原来的 wait，因为 residual prefetch 会复用 store source。

### 各模型独立证据

以下固定 commit 链接保留完整 shape、原始三次计时与 median、正确性、
源码/DSO 哈希和模型收据；每项只证明对应模型和对应源码。

| 模型 / 机制 | 测试范围与结果 | 独立证明 |
| --- | --- | --- |
| InternVL / E29 SIMD | hidden4096 与 Q/K128，原 oracle/stream/tail/输入不变；TP2 text/image greedy32，318.024065s 稳态。旧 Q/K 全局 head 行数属于边界压力测试；实际 TP2 每 rank shape 见下一行 | [E29 proof](https://github.com/Tecorigin/tecovllm-modelzoo/blob/3b8bf1a28fa38374e027dbe2e785cdebd4e06e1e/model_adaptations/InternVL3_5SCUdoudui/validation/rms_epilogue_simd_20261007.json) |
| InternVL / plain DMA | 48/48 focused 与 A/B/A-after bitwise 一致；per-rank Q/K 行数28976/7244和69632/17408；TP2 greedy32，317.156770s 稳态。长 hidden 同步调用延迟下降1.70%–1.75%，Q/K下降12.20%–13.19% | [DMA proof](https://github.com/Tecorigin/tecovllm-modelzoo/blob/3b8bf1a28fa38374e027dbe2e785cdebd4e06e1e/model_adaptations/InternVL3_5SCUdoudui/validation/rms_plain_writeback_20261007.json) |
| Hy-MT2 / E29 SIMD | 自有 hidden2048/QK128、eps1e-5，36 focused 与原0.002容差；TP1 greedy32和129模块/映射/peak。设备 kernel 延迟下降约32%–65%，host wall计时含噪声/异常值；本模型未验证新的 plain DMA | [Hy proof](https://github.com/Tecorigin/tecovllm-modelzoo/blob/2c4378bf0a0f5794b0b3d1fd5a35d2d5e947ce95/model_adaptations/HyMT2SCUdoudui/validation/rms_epilogue_simd_20261007.json) |
| MiniCPM / E29 SIMD | 自有N1/N7、D1536、eps1e-6，原plain0.005/add和residual0.01；A/B/A-after focused、公开8/8、TP1 greedy32，301.928123s/110稳态请求。同步调用延迟下降24.72%–48.18%；本模型 plain DMA 候选为NO-GO | [Mini proof](https://github.com/Tecorigin/tecovllm-modelzoo/blob/0d0c5d4a1ce4361768f8a7e29fa5de0906a918d8/model_adaptations/MiniCPM5SCUdoudui/validation/rms_epilogue_simd_20261008.json) |

E29 是 `e29b53c256f366e6eee538d656bfbb56e867cefc`，kernel SHA256 为
`41a517c23ae849f0d36cbdb2746ea2b75e21e5f2efe4dbaf44b6ed99a62193be`。
InternVL plain DMA 的 kernel SHA256 为
`93e8e80d08d401fffaa3060bf8e6e24bd989ab59fd8cb09c079228710f66d5b7`。
Hy/Mini 的 E29 模型结果不验证新的 DMA kernel。

三次同步 host operator-call 与 profiler device 时间按各证明的计时范围
解释，不相互替代。短 shape 的噪声、负向 median、异常值和模型计时漂移
保留在原证明中；以上各项均不宣称稳定端到端模型加速。

复现脚本、epilogue-only.patch 与 plain-writeback-only.patch 位于上述
各模型公开目录。[官方定向 CI](https://github.com/Tecorigin/teco-ops/pull/37#issuecomment-6041962254)
通过的 head 是 `2949f7061a05af0f4cb2b88fb3344380e3dfc4e0`，早于 plain
DMA 提交；当前 DMA head 的官方 CI/完整 wheel、官方模型环境和组委会
完整精度尚未验证。原始收据中的限制仍按其实际源码与构建范围保留。
