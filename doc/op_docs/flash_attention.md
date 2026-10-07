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

### 参数信息

tecoopsFlashAttention参数信息

| 参数           | 输入/输出 | 主机端/设备端 | 说明                                                               |
| -------------- | --------- | ------------- | ------------------------------------------------------------------ |
| handle         | 输入      | 主机端        | Teco-Ops 句柄，管理设备上下文                                      |
| max_seqlen_q   | 输入      | 主机端        | 最大 query 序列长度                                                |
| max_seqlen_k   | 输入      | 主机端        | 最大 KV 序列长度                                                   |
| max_block_num  | 输入      | 主机端        | KV cache 中最大 block 数量                                         |
| q_seq_lens     | 输入      | 主机端        | 每 batch 的 query 长度，`[batch_size]`                           |
| kv_seq_lens    | 输入      | 主机端        | 每 batch 的 KV 长度，`[batch_size]`                              |
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

| 测试环境    | 测例                 | 硬件时间 (us) |
| ----------- | -------------------- | ------------- |
| CI 测试环境 | test_case_0.prototxt | 待补充        |

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

### D512 SV accumulation SIMD verification (2026-10-07)

Only the BM32 specialization replaces the scalar Step E FP32 additions with
two `floatv16` additions per 32-column SV chunk. Per-element addition order,
rescale, softmax, probability packing, DMA, ABI and the BM64/BM128 paths stay
the same. Selection is compile-time; no forward-time backend switch is added.

The paired measurement baseline is the combined PR36/41/42 Gemma source on
official main `de27305efed0a17ae926d21d5415d8b915614649`. This PR's new kernel
delta is independently applicable to PR41 head `5584d46`; the combined source
is disclosed because the model also needs the other compatibility fixes.
Vendor Python resolves to `/usr/local/python/bin/python3.12`; SDK/runtime3.2.0,
Torch2.12.0a0+0d62256 and Torch-SDAA20260623.8.51+d942f23 were used.

FP16 Q[N,8,512], KV[blocks,1,32,512], scale1, one logical device2, seed20261007,
warmup5 and10 calls per trial. Each bracket contains all three ms/call values.

| N / KV lengths | Baseline ms | Candidate ms | Median baseline → candidate ms |
| --- | --- | --- | --- |
| 1 /33 | [0.242017402,0.243326405,0.240724400] | [0.240233395,0.238871400,0.239240401] | 0.242017402 →0.239240401 |
| 1 /4352 | [10.584034101,10.588059103,10.587743099] | [10.239770997,10.245721001,10.289950797] | 10.587743099 →10.245721001 |
| 4 /33,65,1025,4352 | [13.557968498,13.639971300,13.569197600] | [13.118165702,13.117788697,13.125886599] | 13.569197600 →13.118165702 |
| 8 /33,65,1025,4097,4320,4352,65,4097 | [44.135197904,44.132023799,44.195023697] | [42.701129499,42.771772301,42.764305399] | 44.135197904 →42.764305399 |

The80 paired decode rows and8 paired prefill rows are bitwise equal to their
independent baseline. Prefill Q2/33/64/128 spans KV33/65/1025/4352; original
reference tolerance .02, default/nondefault streams and cache preservation
are retained. The public opaque/fullgraph-eager,4352-boundary and poisoned-tail
focused checks also pass. Candidate core SHA256 is
`e88dd5c34e6cee2bec005dfb5b8696a555e48d83c6c68be2dffc25e3e37f47a8`.

Own Gemma TP2/FP16/context4352 fixed greedy32 matches on83 candidate requests,
including65 steady requests over318.317824s. A→B→A model raw seconds are
[13.562570464,13.465909750,13.543346152] /
[12.748440575,12.935509947,12.962619586] /
[13.004446402,12.935581907,12.969123281]. The final baseline overlaps the
candidate, so no end-to-end speedup is claimed. This is not full official
wheel/CI,official py311 or final committee accuracy evidence. The source and
worker/peak evidence are published in the Gemma model PR's
`validation/d512_accumulate_simd_20261007.json`.

Public SIMD source reproduction: use the Gemma model PR public package at
model_adaptations/Gemma4SCUdoudui. In a fresh checkout of official main
de27305efed0a17ae926d21d5415d8b915614649, apply its archived
op_learning/attention/gemma-d512-model/official_combined.patch with
git apply --check --unidiff-zero and then git apply --unidiff-zero.
The public build helper produces local library provenance for that exact
source; this patch includes the disclosed PR36/41/42 combination.
Its checksum is 5308055b25e87d217574561f4e9545f36d2e779092d83c7d3bcc305176995ca7.
The isolated validated build is tied to this source and its own worker
receipts; a newly built or installed vendor wheel requires its own gates.

### D512 accumulator rescale SIMD verification (2026-10-07)

The additional atomic mechanism changes only BM32 Step B: each FP32 accumulator
element is multiplied by the same row scale using floatv16 load/multiply/store.
Earlier Step E additions, softmax/rounding order, packing, DMA, tile sizes, SPM
allocation and ABI retain their previous behavior. BM64/BM128 use the previous
scalar rescale. Selection is compile-time.

Independent hardware baseline is the prior accepted Step E combination on
official main `de27305efed0a17ae926d21d5415d8b915614649`, core e88dd5c34; this
new mechanism applies exactly to PR41 head5b8f65f. The model's isolated combined
build also contains PR36/42 compatibility bindings. Fresh candidate source
SHA256 `2630b68c14df8285bad0d1dd4f1bcc14fb77c124fc32c7d5dc3ce2e74d74b352` and core SHA256
`d61208eeba2362714f9aab67d74d8f3dd9e57121991c90f0b5a17f62ee0fe4ef` are recorded separately from the unchanged
Torch extension c727f57f. Vendor Python realpath is
`/usr/local/python/bin/python3.12`; SDK/runtime3.2.0, Torch2.12.0a0+0d62256,
Torch-SDAA20260623.8.51+d942f23.

Own paired decode80 and prefill8 rows are bitwise equal to baseline, with
original .02 reference tolerance, exact input/cache preservation, default and
nondefault streams. Ordinary/fullgraph-eager,4352-boundary and poisoned-tail
gates pass. FP16 Q[N,8,512], KV[blocks,1,32,512], scale1, device2, seed20261007,
warmup5 and10 calls/trial are fixed across three micro runs:

| N / KV lengths | Baseline three ms/call values | Candidate three ms/call values | Median baseline → candidate ms |
| --- | --- | --- | --- |
| 1 / 33 | [0.243227498, 0.244031404, 0.245578500] | [0.243547495, 0.243287499, 0.241812400] | 0.244031404 → 0.243287499 |
| 1 / 4352 | [10.242068500, 10.248963395, 10.316291201] | [9.855969297, 9.855175199, 9.856668301] | 10.248963395 → 9.855969297 |
| 4 / 33,65,1025,4352 | [13.125246100, 13.113395998, 13.116348104] | [12.621673202, 12.612573203, 12.612844200] | 13.116348104 → 12.612844200 |
| 8 / 33,65,1025,4097,4320,4352,65,4097 | [42.709624901, 42.696079897, 42.764923704] | [41.114962404, 41.076609504, 41.112411401] | 42.709624901 → 41.112411401 |

Long-KV micro latency decreases3.74–3.84%; the post-candidate baseline retains
medians10.241912602 /13.111219398 /42.709663801ms for the three long cases.
The short KV33 triples overlap, so no short-case improvement is claimed.
The complete A-after triples and all focused cases are in the model proof.

Own public TP2/FP16/context4352/chunk512 configuration matches32-token IDs
on83 candidate requests, including65 requests over
313.616906s. Actual workers retain the selected DSO
mapping, eight D512 layer bindings and memory peaks. Baseline/candidate model raw seconds: [13.565518133, 13.524816754, 13.467264207] / [13.376095781, 13.331312942, 13.374056167]; medians 13.524816754 / 13.374056167. Ordered model runs are regression evidence; no end-to-end speedup is claimed.

Provenance tests6/6, vendor-Python syntax, shell syntax and diff-check pass.
Published U0 combined patch `9756790105110f8fd4ed610d8e12c9269d2992a331fb8a0a530134204b3b7600` reconstructs
byte-identical tested source. Source/proof: Gemma model PR3,
`validation/d512_rescale_simd_20261007.json`. Current-head official CI/full
wheel, official py311, final committee accuracy and new long-input model tests
were not executed. Earlier CI/review and accuracy records retain their original
head/build scope.
