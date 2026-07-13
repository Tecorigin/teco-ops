# tecoopsCausalConv1d 设计文档

## 计算原理

1D Causal Convolution（因果卷积）是一种序列建模操作，在 t 时刻的输出仅依赖于 t 时刻及之前的输入，不会"看到"未来的信息。本算子同时维护一个跨 step 的历史状态（convState），支持变长序列 batch 处理和缓存索引映射。

**计算公式：**

对于 batch `b` 中的位置 `t`：

$$
\begin{aligned}
\mathbf{x}_{in} &= [\text{state}[t], \mathbf{x}[t]] \quad \text{(拼接历史状态与当前输入)} \\
y[t] &= \sum_{i=0}^{W-1} \mathbf{x}_{in}[t - W + 1 + i] \cdot \mathbf{w}[i] \\
\text{out}[t] &= \text{SiLU}(y[t])
\end{aligned}
$$

其中 `W` 为卷积核宽度（width），`state[t]` 为上一轮保留的 `W-1` 个历史值。

**状态更新：**

每次处理后，将最后 `W-1` 个输入写回 `convState`，供下一轮推理使用。

**参数解释：**

- $\mathbf{x}$：输入序列，形状 `[totalSeqLen, dim]`
- $\mathbf{w}$：卷积权重，形状 `[width, dim]`（对每个 dim 通道独立卷积）
- $\text{state}$：卷积历史状态，形状 `[num_entries, width-1, dim]`
- $y$：卷积输出（SiLU 激活前）
- $\text{out}$：最终输出，形状 `[totalSeqLen, dim]`
- $\text{padSlotId}$：无效 cache slot ID（等于该值的 entry 被跳过）
- $\text{api\_mode}$：状态使用模式（`0`=按 `hasInitialState` 标记，`1`=始终使用）

## 功能实现

### 接口设计

参考 Mamba 系列模型中 causal conv1d 的实现，设计 userAPI 如下：

```c++
tecoopsStatus_t tecoopsCausalConv1d(
    tecoopsHandle_t handle,
    int batch,
    int totalSeqLen,
    int dim,
    int width,
    int convStateStride,
    int xSeqStride,
    int outSeqStride,
    int padSlotId,
    int api_mode,
    int actMode,
    const void *x,
    void *convState,
    const void *weight,
    void *out,
    const int *queryStartLoc,
    const int *convStateIndices,
    const int8_t *hasInitialState);
```

### 参数信息

各参数含义如下：

| 参数             | 输入/输出 | 主机端/设备端 | 说明                                                |
| ---------------- | --------- | ------------- | --------------------------------------------------- |
| handle           | 输入      | 主机端        | Teco-Ops 句柄，管理设备上下文                       |
| batch            | 输入      | 主机端        | batch 数量                                          |
| totalSeqLen      | 输入      | 主机端        | 总序列长度（所有 batch 之和）                       |
| dim              | 输入      | 主机端        | 特征维度                                            |
| width            | 输入      | 主机端        | 卷积核宽度                                          |
| convStateStride  | 输入      | 主机端        | convState 中 entry 间跨步（元素数）                 |
| xSeqStride       | 输入      | 主机端        | x 中行间跨步（元素数）                              |
| outSeqStride     | 输入      | 主机端        | out 中行间跨步（元素数）                            |
| padSlotId        | 输入      | 主机端        | 无效 slot ID，匹配此值的 entry 跳过                 |
| api_mode         | 输入      | 主机端        | 状态使用模式                                        |
| actMode          | 输入      | 主机端        | 激活函数模式（预留）                                |
| x                | 输入      | 设备端        | 输入序列`[totalSeqLen, dim]`                      |
| convState        | 输入/输出 | 设备端        | 卷积状态`[num_entries, width-1, dim]`（原地更新） |
| weight           | 输入      | 设备端        | 卷积权重`[width, dim]`                            |
| out              | 输出      | 设备端        | 输出序列`[totalSeqLen, dim]`                      |
| queryStartLoc    | 输入      | 设备端        | 各 batch 起始位置`[batch+1]`，int32               |
| convStateIndices | 输入      | 设备端        | 各 batch 对应 cache entry 索引`[batch]`，int32    |
| hasInitialState  | 输入      | 设备端        | 各 batch 是否有初态标记`[batch]`，int8            |

### 类型限制

当前计算分支支持以下数据类型，其余情况暂不支持。

| 参数             | 数据类型 | 维度信息                        | 存储格式 |
| ---------------- | -------- | ------------------------------- | -------- |
| x                | fp16     | `[totalSeqLen, dim]`          | NCHW     |
| convState        | fp16     | `[num_entries, width-1, dim]` | NCHW     |
| weight           | fp16     | `[width, dim]`                | NCHW     |
| out              | fp16     | `[totalSeqLen, dim]`          | NCHW     |
| queryStartLoc    | int32    | `[batch+1]`                   | Array    |
| convStateIndices | int32    | `[batch]`                     | Array    |
| hasInitialState  | int8     | `[batch]`                     | Array    |
| batch            | int      | 标量，`>= 1`                  | -        |
| totalSeqLen      | int      | 标量，`>= 1`                  | -        |
| dim              | int      | 标量，`>= 1`                  | -        |
| width            | int      | 标量，`= 4`                   | -        |

## 性能优化

### 计算流程

当前 kernel 在单个 SPE 核心上以 `threadIdx == 0` 串行执行（单线程版本），流程如下：

```
for b in batch:
    cache_idx = convStateIndices[b]
    if cache_idx == padSlotId: continue   ← 跳过无效 slot
  
    current_state = convState[cache_idx * convStateStride]
    use_state = hasInitialState[b]
  
    for d in dim:
        x0, x1, x2 = current_state (或 0)   ← 加载历史状态
        for t in seq_len:
            x3 = x[pos * xSeqStride + d]
            sum = x0*w0 + x1*w1 + x2*w2 + x3*w3
            out[pos * outSeqStride + d] = SiLU(sum)
            shift: x0=x1, x1=x2, x2=x3
        current_state[0..2] = x1, x2, x3   ← 更新状态
```

### 性能数据

| 测例   | 配置                                                   | 状态 |
| ------ | ------------------------------------------------------ | ---- |
| case_0 | batch=1, totalSeqLen=8, dim=2048, width=4              | OK   |
| case_1 | batch=2, totalSeqLen=16, dim=256, width=4              | OK   |
| case_2 | batch=2, totalSeqLen=1024, dim=1280, width=4, pad_skip | OK   |

## 分支派发

| 算法取值                 | 计算分支                          | 含义说明                   |
| ------------------------ | --------------------------------- | -------------------------- |
| `branch=0`（自动派发） | `teco_slave_causal_conv1d_half` | fp16 1D causal conv + SiLU |

当前仅支持 fp16 一个分支，根据 `data_type = UAL_DTYPE_HALF` 自动派发。

## 文件结构

```
teco/
├── interface/
│   ├── include/tecoops.h                    # userAPI 声明
│   └── ops/causal_conv1d.cpp                # 接口实现
├── ual/
│   ├── args/causal_conv1d_args.h            # 参数结构体
│   ├── ops/causal_conv1d/
│   │   ├── causal_conv1d.hpp                # Op 类定义
│   │   ├── find_causal_conv1d.cpp           # 分支选择
│   │   └── find_causal_conv1d.h
│   └── kernel/causal_conv1d/
│       ├── causal_conv1d.h                  # kernel 声明
│       └── causal_conv1d.scpp               # kernel 实现
├── plugin/
│   └── pluginCausalConv1d/
│       └── plugin_causal_conv1d.cc          # Plugin 自定义算子
test/
├── test_proto/
│   ├── tecokernel/causal_conv1d.proto       # Proto 参数定义
│   └── tecokernel.proto                     # 注册 TecokernelParam
└── zoo/teco/causal_conv1d/
    ├── causal_conv1d.h                      # 测试类声明
    ├── causal_conv1d.cpp                    # 测试实现
    ├── causal_conv1d.py                     # PyTorch baseline
    └── test_case/
        ├── case_0.prototxt                  # batch=1, totalSeqLen=8, dim=2048
        ├── case_1.prototxt                  # batch=2, totalSeqLen=16, dim=256
        └── case_2.prototxt                  # batch=2, totalSeqLen=1024, dim=1280, pad_skip
api/
├── torch_ext.cpp                            # PyTorch 绑定
└── tecoops/__init__.py                      # Python 导出
python_api_test/
└── test_causal_conv1d.py                    # Python API 精度测试
plugin_test/
└── test_plugin_causal_conv1d.py             # Plugin 推理精度测试
```

## 使用示例

```python
import torch
import tecoops

dim, total_seqlen, width, batch = 2048, 8, 4, 1
num_entries = 4

x = torch.randn(dim, total_seqlen, dtype=torch.half, device='sdaa')
weight = torch.randn(width, dim, dtype=torch.half, device='sdaa')
conv_states = torch.randn(num_entries, width - 1, dim, dtype=torch.half, device='sdaa')
query_start_loc = torch.tensor([0, total_seqlen], dtype=torch.int32, device='sdaa')
cache_indices = torch.tensor([0], dtype=torch.int32, device='sdaa')
has_init = torch.tensor([0], dtype=torch.int8, device='sdaa')
out = torch.empty(dim, total_seqlen, dtype=torch.half, device='sdaa')

tecoops.causal_conv1d_fn_torch(
    out, conv_states, x, weight,
    query_start_loc, cache_indices, has_init,
    pad_slot_id=-1)
```
