# tecoopsFlashAttention 设计文档

## 计算原理

Flash Attention 通过分块（tiling）和在线 softmax（online softmax）技术，在不牺牲精度的情况下减少 HBM 访问量。计算过程为：

```
O = Softmax(Q @ K^T * scale) @ V
```

**分块策略：**

- Q 分块大小默认 `BM = 128`，head_size=256/512 时分别使用独立 `BM = 64/32` 入口；K/V 分块大小 `BN = 32`
- 对于每个 Q block，依次加载所有的 KV block 进行分块计算
- 每个 KV block 的中间结果通过 **online softmax** 累积到 float 累加器中

**Online Softmax 原理：**

```
对于每个 Q block:
  初始化 o_accum = 0, l = 0, m = -inf
  
  对于每个 KV block:
    1. S = Q_block @ K_block^T * scale              ← gemm 计算分数
    2. 对每行 i:
         m_new[i] = max(m[i], max(S[i]), 0)
         scale[i] = exp(m[i] - m_new[i])            ← 旧累积的重缩放因子
         o_accum[i] *= scale[i]                     ← 重缩放旧累积
         S[i] = exp(S[i] - m_new[i])                ← 未归一化概率
         l_new[i] = l[i] * scale[i] + sum(S[i])     ← 新累积和
         m[i] = m_new[i], l[i] = l_new[i]
    3. o_accum += S @ V_block                       ← 累加当前块贡献
  
  输出: obuf = o_accum / l                           ← 归一化
```

**GQA (Grouped Query Attention) 支持：**

- 每个 KV head 服务 `num_heads / num_kv_heads` 个 query head
- `kv_group = tid * num_kv_heads / num_heads`

## 功能实现

### 接口设计

```c++
tecoopsStatus_t tecoopsFlashAttention(
    tecoopsHandle_t handle,
    int max_seqlen_q,
    int max_seqlen_k,
    int max_block_num,
    const int *q_seq_lens,           // [batch_size]
    const int *kv_seq_lens,          // [batch_size]
    const tecoopsTensorDescriptor_t blockTableDesc,
    const void *blockTable,          // [batch_size, block_table_dim] int32
    const tecoopsTensorDescriptor_t qDataDesc,
    const void *qData,               // [total_q, num_heads, head_size] half
    const tecoopsTensorDescriptor_t kCacheDesc,
    const void *kCache,              // [max_block, num_kv_heads, block_size, head_size] half
    const tecoopsTensorDescriptor_t vCacheDesc,
    const void *vCache,              // [max_block, num_kv_heads, block_size, head_size] half
    const tecoopsTensorDescriptor_t oDataDesc,
    void *oData,                     // [total_q, num_heads, head_size] half
    void *workspace);
```

Python API：

```python
tecoops.flash_attn_varlen_func(
    q,              # [total_q, num_heads, head_size]
    k,              # [num_blocks, num_kv_heads, block_size, head_size]
    v,              # [num_blocks, num_kv_heads, block_size, head_size]
    max_seqlen_q,
    cu_seqlens_q,   # [batch_size + 1] 累积和
    max_seqlen_k,
    cu_seqlens_k=None,
    seqused_k=None, # [batch_size] 每 batch 实际 KV 数
    softmax_scale=None,
    causal=False,
    window_size=None,
    block_table=None, # [batch_size, block_table_dim]
    return_softmax_lse=False,
    out=None,
)
```

### 显式 scale 与兼容性

`tecoopsFlashAttentionWithScale` 在原 C 接口 `max_block_num` 后新增 `float softmax_scale`，其余参数和布局一致。
原 `tecoopsFlashAttention` 符号及默认 `1/sqrt(head_size)` 语义保持不变。
Python `flash_attn_varlen_func` 的已有 `softmax_scale` 参数现在传到 kernel，有限零值和负值也按公式计算；非有限值拒绝执行。
`softmax_scale=0` 会使所有有效 QK 分数变为 0，因此 softmax 在当前 query 可见的 KV 项上得到均匀 attention 权重；
`softmax_scale=1` 则使用未缩放的 QK 内积作为分数。
Python pybind 的完整位置参数 ABI 不变，调用方仍应显式提供 scale。
该修复不改变 causal/window 支持或 kernel 数学实现。Gemma 的 scale=1 需要这项修复；D512 容量支持另见 PR41，本 PR 未做模型接入或性能声明。

### 参数信息

tecoopsFlashAttention参数信息

| 参数           | 输入/输出 | 主机端/设备端 | 说明                                                               |
| -------------- | --------- | ------------- | ------------------------------------------------------------------ |
| handle         | 输入      | 主机端        | Teco-Ops 句柄，管理设备上下文                                      |
| max_seqlen_q   | 输入      | 主机端        | 最大 query 序列长度                                                |
| max_seqlen_k   | 输入      | 主机端        | 最大 KV 序列长度                                                   |
| max_block_num  | 输入      | 主机端        | KV cache 中最大 block 数量                                         |
| q_seq_lens     | 输入      | 设备端        | 每 batch 的 query 长度，`[batch_size]`                           |
| kv_seq_lens    | 输入      | 设备端        | 每 batch 的 KV 长度，`[batch_size]`                              |
| blockTableDesc | 输入      | 主机端        | block table 描述符                                                 |
| blockTable     | 输入      | 设备端        | block id 映射表，`[batch_size, block_table_dim]` int32           |
| qDataDesc      | 输入      | 主机端        | Q 数据描述符                                                       |
| qData          | 输入      | 设备端        | Q 矩阵，`[total_q, num_heads, head_size]` half                   |
| kCacheDesc     | 输入      | 主机端        | K cache 描述符                                                     |
| kCache         | 输入      | 设备端        | K cache，`[max_block, num_kv_heads, block_size, head_size]` half |
| vCacheDesc     | 输入      | 主机端        | V cache 描述符                                                     |
| vCache         | 输入      | 设备端        | V cache，`[max_block, num_kv_heads, block_size, head_size]` half |
| oDataDesc      | 输入      | 主机端        | 输出描述符                                                         |
| oData          | 输出      | 设备端        | 输出矩阵，`[total_q, num_heads, head_size]` half                 |
| workspace      | 输入      | 设备端        | 工作空间（当前未使用）                                             |

### 输出初始化契约

`tecoopsFlashAttention` 会覆盖全部 `[total_q, num_heads, head_size]` 输出元素。PyTorch
绑定在 `out` 未提供时使用 `empty_like`，在 `out` 已提供时直接写入，不再为每次调用执行整块
设备清零。调用方必须传入与 `q` 同形状、可写的输出张量；API focused test 用 `7.0` 哨兵预填
输出并与参考结果逐元素比较，以便暴露任何未写入区域。

### 类型限制

| 参数          | 数据类型 | 维度信息                                             | 存储格式 |
| ------------- | -------- | ---------------------------------------------------- | -------- |
| max_seqlen_q  | int32    | 标量                                                 | -        |
| max_seqlen_k  | int32    | 标量                                                 | -        |
| max_block_num | int32    | 标量                                                 | -        |
| softmax_scale | float    | 标量                                                 | -        |
| qData         | float16  | `[total_q, num_heads, head_size]`                  | Array    |
| kCache        | float16  | `[max_block, num_kv_heads, block_size, head_size]` | Array    |
| vCache        | float16  | `[max_block, num_kv_heads, block_size, head_size]` | Array    |
| blockTable    | int32    | `[batch_size, block_table_dim]`                    | Array    |
| q_seq_lens    | int32    | `[batch_size]`                                     | Array    |
| kv_seq_lens   | int32    | `[batch_size]`                                     | Array    |
| oData         | float16  | `[total_q, num_heads, head_size]`                  | Array    |
| workspace     | void*    | 标量                                                 | -        |

### PyTorch 绑定的变长元数据

SDAA kernel 直接读取设备端的 `q_seq_lens` 与 `kv_seq_lens` 指针。PyTorch
绑定从设备端 `cu_seqlens_q` 的相邻元素差分生成连续的 `q_seq_lens`，不把
累积长度拷回 CPU，也不再把派生数组复制回设备；`seqused_k` 直接作为
`kv_seq_lens` 使用。绑定在派发前把句柄绑定到调用方当前 SDAA stream，保证
元数据差分与 attention kernel 的顺序。这样保持 C API 的 `[batch_size]` ABI
不变，同时避免每次变长调用的主机同步。

## 性能优化

### 当前实现的计算分支

支持以下场景：

| 场景            | q_seq_len    | kv_seq_len  | causal | 说明          |
| --------------- | ------------ | ----------- | ------ | ------------- |
| prefill         | = kv_seq_len | = q_seq_len | 是     | 首轮全量计算  |
| decode          | 1            | 任意        | 是     | 逐 token 推理 |
| chunked prefill | < kv_seq_len | > q_seq_len | 是     | 分块预填充    |

### 优化设计

**1. 数据分块 (Tiling)**

- Q 默认按 `BM=128` 分块，D256/D512 分别按 `BM=64/32` 分块；K/V 按 `BN=32` 分块
- 每块独立加载到 SPM，减少 HBM 访问
- 每个 Q block 遍历所有 KV block 后一次性写出结果

**2. Online Softmax 分块累积**

- 使用 float32 累加器（`o_accum`）跨 KV block 累积
- 每步记录 `m`（row max）和 `l`（row sum），更新时重缩放旧累积
- 所有 KV block 处理完后归一化：`obuf = o_accum / l` → half

**3. SIMD 向量化**

- 计算 `exp(S - m_new)`、`l_block`、float→half 转换均使用 `floatv16` / `halfv16` 向量化
- 每批次处理 16 个元素，64 字节对齐

**4. GQA (Grouped Query Attention)**

每个 query head 映射到对应的 KV head：`kv_group = tid * num_kv_heads / num_heads`

### 性能数据

| 测试环境 | 测例 | 配置与 Shape | 状态 |
| --- | --- | --- | --- |
| 太初 SDAA 3.2.0 | 0.prototxt | Q: [256, 32, 128], KV_cache: [8, 8, 32, 128] | OK |
| 太初 SDAA 3.2.0 | 1.prototxt | Q: [256, 32, 128], KV_cache: [8, 8, 32, 128] | OK |
| 太初 SDAA 3.2.0 | 2.prototxt | Q: [256, 32, 128], KV_cache: [8, 8, 32, 128] | OK |
| 太初 SDAA 3.2.0 | 3.prototxt | Q: [256, 32, 128], KV_cache: [8, 8, 32, 128] | OK |
| 太初 SDAA 3.2.0 | 4.prototxt | Q: [256, 32, 128], KV_cache: [8, 8, 32, 128] | OK |
| 太初 SDAA 3.2.0 | 5.prototxt | Q: [256, 32, 128], KV_cache: [8, 8, 32, 128] | OK |

## 分支派发

| 算法取值           | 计算分支                            | 含义说明                            |
| ------------------ | ----------------------------------- | ----------------------------------- |
| `TECOOPS_ALGO_0` | `teco_slave_flash_attention_half` | 基础实现，half 精度，单 SPE 单 head |
| `TECOOPS_ALGO_1` | `teco_slave_flash_attention_half_d256` | head_size=256 的 BM64 编译期特化 |
| `TECOOPS_ALGO_2` | `teco_slave_flash_attention_half_d512` | head_size=512 的 BM32 编译期特化 |

### D256/D512 SPM 预算与范围

当 `size_per_head == 256` 时，外层按 head size 内部选择算法索引 1（BM=64）；
当 `size_per_head == 512` 时，内部选择算法索引 2（BM=32）。当前 C/Python 接口没有公开算法选择参数，
调用方不能为 D256/D512 请求 BM128。三个入口共享同一模板实现；
BM 是编译期常量；D64/D128 保留 BM128，设备热路径不增加后端选择。
HAL 的 `M128_N32` 接口允许 `M2 <= 128, N2 = 32`，两次 GEMM 的配置不变。

每个 SPE 的显式缓冲区预算（bytes）为：

`8*BM*D + 4*BN*D + 6*BM*BN + 20*BM + 4*block_table_dim`。

| 维度与分块 | 缓冲区字节数（另加 block table） | 说明 |
| --- | ---: | --- |
| D256 / BM128 / BN32 | 322048 | 超过仓库 240512-byte SPM 上限；o_accum 单独需 131072 bytes |
| D256 / BM64 / BN32 | 177408 | 距上限余 63104 bytes，需容纳 block table、分配对齐和运行时开销 |
| D512 / BM32 / BN32 | 203392 | 距上限余 37120 bytes，需容纳 block table、分配对齐和运行时开销 |

该预算不包含栈和 HAL 运行时开销，实际正确性以设备测例为准。FP32 累加器、online
softmax、GQA、右对齐 causal mask、KV cache 布局及 Python ABI 均保持原语义。
D512 分支沿用通用 paged causal 计算，不代表模型已接入此分支。当前 BN 固定为 block_size=32；Python 的 `window_size`
尚未传入底层 kernel，因此不支持 sliding-window mask，也不能据此声称 Gemma
超过 1024 tokens 的 sliding attention 正确。本改动不声明性能提升。

## 文件结构

```
teco/
├── interface/
│   ├── include/tecoops.h                    # userAPI 声明
│   └── ops/flash_attention.cpp              # 接口实现
├── ual/
│   ├── args/flash_attention_args.h         # 参数结构体
│   ├── ops/flash_attention/
│   │   ├── flash_attention.hpp             # Op 类定义
│   │   ├── find_flash_attention.cpp        # 分支选择
│   │   └── find_flash_attention.h
│   └── kernel/flash_attention/
│       ├── flash_attention.h              # kernel 声明
│       └── flash_attention.scpp            # kernel 实现
├── plugin/
│   └── pluginFlashAttention/               # teco-inference 插件
│       └── plugin_flash_attention.cc
test/
├── zoo/teco/flash_attention/
│   ├── flash_attention.cpp                 # 测试 executor
│   ├── flash_attention.h
│   ├── flash_attention.py                  # Python reference
│   └── test_case/
│       ├── 0.prototxt ~ 7.prototxt         # 测试用例
api/
├── torch_ext.cpp                           # PyTorch 扩展绑定
└── tecoops/
    └── __init__.py                         # Python API
```

## 使用示例

```python
import torch
import torch_sdaa
import tecoops

# 构造输入 (prefill: L=S=256)
batch_size, num_heads, num_kv_heads, head_size, block_size = 1, 32, 8, 128, 32
total_q, total_kv = 256, 256
block_table_dim = (total_kv + block_size - 1) // block_size

q = torch.randn(total_q, num_heads, head_size, dtype=torch.float16, device='sdaa')
k_cache = torch.randn(block_table_dim, num_kv_heads, block_size, head_size, dtype=torch.float16, device='sdaa')
v_cache = torch.randn(block_table_dim, num_kv_heads, block_size, head_size, dtype=torch.float16, device='sdaa')
block_table = torch.zeros(batch_size, block_table_dim, dtype=torch.int32, device='sdaa')
cu_seqlens_q = torch.tensor([0, total_q], dtype=torch.int32, device='sdaa')
seqused_k = torch.tensor([total_kv], dtype=torch.int32, device='sdaa')
out = torch.empty_like(q)

tecoops.flash_attn_varlen_func(
    q, k_cache, v_cache, total_q, cu_seqlens_q, total_kv,
    seqused_k=seqused_k, causal=True, block_table=block_table, out=out,
)
```

## D512 SIMD 机制与独立验证

BM32/D512 的 Step B 用 `floatv16` 逐元素重缩放 FP32 累加器，Step E 用
`floatv16` 累加 SV。两项机制分别验证；逐元素数学顺序、softmax、FP16
packing、DMA、SPM 分配和 ABI 保持原实现。BM64/BM128 保留标量路径，
特化在编译期选择。

Gemma 独立测试使用 FP16 Q[N,8,512]、KV[blocks,1,32,512]、scale=1、
N=1/4/8、KV=33..4352。每项均通过 80 个 paired decode 和 8 个 paired
prefill 用例：与各自基线 bitwise 相同，CPU 容差保持 0.02，输入/cache
不变；默认/非默认 stream、4352 边界和毒化尾部均通过。各自的 TP2/FP16
固定 greedy32 回归与超过 300 秒稳态测试也通过。

同输入、warmup5、10 calls/trial、连续三轮的同步 operator-call 计时中，
长 KV 的 Step E 延迟下降 3.11%–3.32%，Step B 再下降 3.74%–3.84%。
全部三次原值、A-after、median、shape、构建哈希和 worker 收据分别保存在
[Step E 证明](https://github.com/Tecorigin/tecovllm-modelzoo/blob/3746150cb6d78e7154de5046d56f0189bf3dcb82/model_adaptations/Gemma4SCUdoudui/validation/d512_accumulate_simd_20261007.json)
和 [Step B 证明](https://github.com/Tecorigin/tecovllm-modelzoo/blob/3746150cb6d78e7154de5046d56f0189bf3dcb82/model_adaptations/Gemma4SCUdoudui/validation/d512_rescale_simd_20261007.json)。
短 KV 计时范围重叠，模型顺序计时存在漂移，均不作为稳定端到端模型加速结论。

实测 isolated Gemma 构建还包含 PR36/42 的兼容绑定；它与本 PR 的独立
kernel 差异分别留证。复现入口和固定官方 main 基线见
[Gemma README](https://github.com/Tecorigin/tecovllm-modelzoo/blob/3746150cb6d78e7154de5046d56f0189bf3dcb82/model_adaptations/Gemma4SCUdoudui/README.md)，
完整补丁见 [official_combined.patch](https://github.com/Tecorigin/tecovllm-modelzoo/blob/3746150cb6d78e7154de5046d56f0189bf3dcb82/model_adaptations/Gemma4SCUdoudui/op_learning/attention/gemma-d512-model/official_combined.patch)。
[官方定向 CI](https://github.com/Tecorigin/teco-ops/pull/41#issuecomment-6041987041)
通过的 head 是 `cb7e7732b81b01eb5c5330459d3ab5c130185ceb`；后续文档
提交保留该 kernel。完整组合 wheel、官方模型环境与组委会完整精度仍需验证。
