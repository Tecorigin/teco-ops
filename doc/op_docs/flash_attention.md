# tecoopsFlashAttention 设计文档

## 计算原理

Flash Attention 通过分块（tiling）和在线 softmax（online softmax）技术，在不牺牲精度的情况下减少 HBM 访问量。计算过程为：

```
O = Softmax(Q @ K^T * scale) @ V
```

**分块策略：**

- Q 分块大小 `BM = 128`，K/V 分块大小 `BN = 32`
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

当前 ALGO0 要求 `size_per_head % 32 == 0`、`block_size == 32`。
不满足时，UAL 返回 `Status::NOT_IMPLEMENTED`，C API 映射为
`TECOOPS_STATUS_NOT_SUPPORTED`（6），Python API 抛出
`RuntimeError`，不会返回未写入的输出。`head_dim=48` 虽然是 16 的倍数，仍不受支持。
Plugin 也遵守同一分页约束：长度为 64/128 的 KV 序列应映射为 2/4 个 32-token
页，不能把序列长度当作 `block_size`。`block_table` 必须给出每条序列实际使用的物理页号。
此分支只注册 ALGO0；D256/D512 专属算法不属于本次守卫修复。
ALGO0 至少分配 `1152 * size_per_head + 27136` 字节 SPM（另加 block table），
分发时用 64 位整数累计此值与按 64 字节向上对齐的 `4 * block_table_dim`，
超过 235 KiB 时拒绝；`size_per_head` 和 `block_table_dim` 必须为正。
因此从 192 起的 32 对齐维度已超过预算，统一拒绝，
包括 D256/D512；本 PR 不声称支持这些维度。

分发判定表（实现见 `teco/ual/ops/flash_attention/find_flash_attention.cpp`，
逐项与代码返回值核对，不满足即 fail-closed）：

| 输入条件 | 支持 | 分发结果 |
| --- | --- | --- |
| `size_per_head % 32 == 0` 且 `block_size == 32` 且 SPM 预算内 | ✅ | 走 ALGO0 |
| `size_per_head % 32 != 0`（含 `head_dim=48`） | ❌ | `NOT_IMPLEMENTED` → C API `TECOOPS_STATUS_NOT_SUPPORTED`(6) |
| `block_size != 32`（含 KV 页长 64/128） | ❌ | 同上（6） |
| `size_per_head >= 192`（SPM > 235 KiB，含 D256/D512） | ❌ | 同上（6） |
| `size_per_head <= 0` 或 `block_table_dim <= 0` | ❌ | 同上（6） |
| 非 half / 非连续 / 描述符维度不符 | ❌ | `TECOOPS_STATUS_BAD_PARAM`(3)，参数校验层拒绝 |

本分支的 `findFlashAttentionBranch()` 只返回 `algo = 0`，因此不存在对算法表
的越界索引；若后续重新接入 D256/D512，必须**在同一个提交内**同时补齐 kernel
声明、算法表条目与分发下标，并让分发下标在取表前做 `[0, count)` 一致性检查，
避免出现“表项缺失但下标有效”的静默越界。

Plugin 的第五个输入现在明确为 `q_seq_lens: int32[batch_size]`，例如两条 query
长度为 1 和 32 时传 `[1, 32]`。旧名称 `cu_seqlens_q` 与实现所需的实际长度不符，
原实现把累计值直接交给 kernel，未做差分，不能正确执行旧累计格式的图。
旧图调用方需先将累计长度差分为实际长度，并改接 `q_seq_lens`；此修复不宣称旧图兼容。
PyTorch API 仍接收 `cu_seqlens_q`，由其已有绑定负责转换。

### 破坏性行为变更（Release Note 措辞）

> **Breaking change (flash_attention ALGO0 dispatch)**: shapes that only passed the
> old `head_dim % 16 == 0` guard — most notably `head_dim=48` — were accepted and
> computed, but produced **wrong results** (measured: `head_dim=48` yields an output
> whose only non-finite elements are NaNs while the CPU reference is finite; the
> official DIFF1 harness aborts with `TEST_STATUS_RESULT_NAN_ERROR`). The dispatch
> guard is now fail-closed on the real kernel preconditions (`head_dim % 32 == 0`,
> `block_size == 32`, positive `head_dim`/`block_table_dim`, ALGO0 SPM budget
> ≤ 235 KiB — the budget additionally rejects configurations such as `head_dim=160`
> with `block_table_dim=8192`): those shapes are now **hard-rejected** before any
> kernel launch instead of returning silently wrong values. Callers that previously
> depended on the old behaviour will now see an error instead of a NaN/garbage
> output. No dispatch threshold was relaxed; only shapes that already computed
> incorrectly (or exceeded the ALGO0 SPM budget) changed from "runs with wrong
> results" to "fails explicitly".

中文要点：`head_dim=48/160` 等形状此前可以运行但**结果错误**（NaN），本 PR 起改为
派发期直接拒绝（fail-closed）。这是有意的破坏性行为变更——把"算错"变成"明确报错"，
不改变任何已正确形状的行为。若第三方（如 vLLM 侧）依赖旧行为，会从"结果错"变成
"直接报错"，请在升级说明中显式标注。

#### 受影响的形状集合（release note 素材，来自本 PR 的 8 个负例）

规则（三条互相独立的 fail-closed 前置条件，均在 `find_flash_attention.cpp` 派发期判定）：

1. `head_dim % 32 != 0` → 拒绝（旧实现只校验 `% 16`，`head_dim=48` 会运行但输出 NaN）；
2. `head_dim % 32 == 0` 但 ALGO0 的 SPM 预算 `1152*head_dim + 27136 + align64(4*block_table_dim) > 235 KiB` → 拒绝（从 192 起的 32 对齐维度全部超预算，含本 PR 不提供的 D256/D512）；
3. `block_size != 32` → 拒绝（ALGO0 硬编码 `BN == block_size == 32`，其他值会静默读错页）。

| 类别 | 具体形状（本 PR 负例原样） | 旧行为 | 现行为 |
| --- | --- | --- | --- |
| head_dim 非 32 倍数 | `head_dim=48`（16 的倍数）、`head_dim=160` | 运行但输出 NaN / 错值 | `NOT_SUPPORTED`（6），不启动 kernel |
| 32 倍数但超 ALGO0 SPM 预算 | `head_dim=192`、`256`、`512` | 误派发 ALGO0、超预算 SPM 分配 | `NOT_SUPPORTED`（6） |
| block_size 非 32 | `block_size=16`、`64`、`128`（与 `head_dim=128` 组合） | 静默按错误页步长读取 KV | `NOT_SUPPORTED`（6） |
| 预算与表宽组合 | `head_dim=160` + `block_table_dim=8192` | 超预算 SPM 分配 | `NOT_SUPPORTED`（6） |

仍受支持的正向形状：`head_dim ∈ {32, 64, 96, 128, 160}` 且 `block_size == 32` 且 `block_table_dim` 使上表预算 ≤ 235 KiB（例如 `head_dim=160` 时 `block_table_dim ≲ 7296`）。

### `NOT_IMPLEMENTED` 与 `BAD_PARAM` 的区分（返回码语义）

本 PR 只在 **UAL 派发层**返回 `-1 → Status::NOT_IMPLEMENTED`（UAL 枚举值 4），
`api/torch_ext.cpp` 与 C 接口的返回码**未改动**。不同层的映射与触发条件：

| 层 | 返回码 | 触发条件 | 调用方语义 |
| --- | --- | --- | --- |
| C API 参数检查（`flashAttentionCheckArgs`） | `TECOOPS_STATUS_BAD_PARAM`（3） | `k_cache`/`v_cache` dtype 不一致；dtype 非 half；两者 shape 不一致 | **参数形状/dtype 非法**：换一组合法参数重试即可 |
| C API 派发拒绝（经 `Convert::toStatus`） | `TECOOPS_STATUS_NOT_SUPPORTED`（6）（UAL `NOT_IMPLEMENTED`/`NOT_SUPPORTED`/`NO_IMPLEMENTATION` 均映射到 6） | `head_dim <= 0` 或 `block_table_dim <= 0`；`head_dim % 32 != 0`（如 48）；ALGO0 SPM 预算超 235 KiB（含所有 `head_dim >= 192` 的 32 对齐维度 192/256/512，以及 `head_dim=160` 配超大 `block_table_dim=8192` 的组合）；`block_size != 32`（如 16/64/128） | **该形状没有对应实现**（fail-closed）：换 shape（32 对齐 head_dim + block_size=32 + SPM 预算内）或换算子重试 |
| Python `flash_attn_varlen_func` | `RuntimeError("flash_attn_varlen_func rejected parameters (status <n>)")` | 上述任一非 SUCCESS 状态 | 状态码 `<n>` 为 C API 值（3 = BAD_PARAM，6 = NOT_SUPPORTED） |

即：**`BAD_PARAM`（3）= 参数本身非法**（dtype/shape 不匹配），调用方应修正参数；
**`NOT_SUPPORTED`（6，源于 UAL `NOT_IMPLEMENTED`）= 参数合法但该形状/配置没有实现**，
调用方应更换形状或算子。当前所有"形状非法/未实现"情形（负例覆盖：`head_dim` 48/192/256/512、
`block_size` 16/64/128、`table_dim` 8192）都走第 2 行，即统一为 6；本轮**不**把非法形状
改判为 `BAD_PARAM`（review 建议选项二，为避免改变已发布的返回码契约，仅文档区分）。

### Head Dimension (D) 与 Page Size (block_size) 支持与拒绝规格表

依据 `teco/interface/ops/flash_attention.cpp`、`teco/ual/ops/flash_attention/find_flash_attention.cpp` 与 `teco/plugin/pluginFlashAttention/plugin_flash_attention.cc` 的代码实现，将各维度与 page size 的支持状态、派发路径及拒绝返回码整理如下表：

| 配置类别 | 具体输入条件与形状示例 | 支持状态 | 算法分支 | UAL 返回码 | C API 返回码 | 拒绝行为与返回码判定依据 |
| --- | --- | :---: | --- | --- | --- | --- |
| **标准支持配置** | `head_size % 32 == 0` 且 `block_size == 32` 且 SPM 预算内（如 `D ∈ {32, 64, 96, 128, 160}`，`block_size == 32`） | ✅ 支持 | 走 ALGO0（`teco_slave_flash_attention_half`） | `SUCCESS`（0） | `TECOOPS_STATUS_SUCCESS`（0） | 正常派发并启动 kernel，写出有效输出 |
| **head_size 非 32 对齐** | `head_size % 32 != 0`（如 `D=48, 80, 112`；`D=48` 虽满足 `% 16 == 0` 但不满足 32 对齐） | ❌ 拒绝 | 无（派发层 fail-closed 拦截） | `Status::NOT_IMPLEMENTED`（4） | `TECOOPS_STATUS_NOT_SUPPORTED`（6） | **非 BAD_PARAM**；ALGO0 的 PV GEMM 按 32 列步进，非 32 对齐导致尾列未累加且 SIMD 重缩放越界。旧版输出 NaN，现 fail-closed 抛 `RuntimeError(status 6)`，输出张量未被写入（哨兵保留） |
| **block_size 非 32** | `block_size != 32`（如 `block_size=16, 64, 128`） | ❌ 拒绝 | 无（派发层 fail-closed 拦截） | `Status::NOT_IMPLEMENTED`（4） | `TECOOPS_STATUS_NOT_SUPPORTED`（6） | **非 BAD_PARAM**；ALGO0 硬编码 `BN == block_size == 32`，非 32 会静默读取错误物理页。现 fail-closed 抛 `RuntimeError(status 6)`，输出张量未被写入（哨兵保留） |
| **大头维度 D256 / D512** | `D=256` 或 `D=512`（即使 `block_size == 32`） | ❌ 当前分支拒绝 | 本分支仅注册 ALGO0，拒绝派发；走 ALGO1/2 需满足前置条件（见下） | `Status::NOT_IMPLEMENTED`（4） | `TECOOPS_STATUS_NOT_SUPPORTED`（6） | **非 BAD_PARAM**；ALGO0 的单核 SPM 需求公式（`1152*D + 27136 > 235 KiB`）超出硬件限制。当前分支 fail-closed 抛 `RuntimeError(status 6)`，输出未被写入 |
| **超 SPM 预算配置** | `head_size % 32 == 0` 但 `1152*D + 27136 + align64(4*table_dim) > 235 KiB`（如 `D=192`，或 `D=160` + `table_dim=8192`） | ❌ 拒绝 | 无（派发层 fail-closed 拦截） | `Status::NOT_IMPLEMENTED`（4） | `TECOOPS_STATUS_NOT_SUPPORTED`（6） | **非 BAD_PARAM**；超出硬件 SPM 235 KiB 容量限制，强行分配会导致设备内存越界。现 fail-closed 抛 `RuntimeError(status 6)`，输出未被写入 |
| **非正数维度** | `head_dim <= 0` 或 `block_table_dim <= 0` | ❌ 拒绝 | 无（派发层 fail-closed 拦截） | `Status::NOT_IMPLEMENTED`（4） | `TECOOPS_STATUS_NOT_SUPPORTED`（6） | **非 BAD_PARAM**；参数非法，派发层校验拦截 |
| **非法数据类型 / 描述符** | `k_cache`/`v_cache` dtype 非 half，或描述符维度不一致 | ❌ 拒绝 | 无（参数层拦截） | N/A（在 C API 层拦截） | `TECOOPS_STATUS_BAD_PARAM`（3） | **是 BAD_PARAM**；参数本身非法，在 `flashAttentionCheckArgs` 中校验失败直接返回 3，未进入 UAL 派发 |

#### D256 / D512 走 ALGO1 / ALGO2 的前置条件

当前 PR 分支仅包含 ALGO0 基础实现，未引入大头维度特化 kernel。若后续分支要接入并支持 D256 / D512，必须**在同一提交内**满足以下全部前置条件，方可开放分发：
1. **特化 Kernel 源码**：引入并编译独立的 D256 与 D512 slave kernel（如 `teco_slave_flash_attention_d256_half` 与 `teco_slave_flash_attention_d512_half`），重构其 SPM 分块（通过缩小 Q block 行数 `BM`、调整计算步进与异步双缓冲结构，使单核内存占用严格受控在 256 KiB SPM 硬限制内）；
2. **算子类算法表注册**：在 `teco/ual/ops/flash_attention/flash_attention.hpp`（`FlashAttentionOp`）的算法数组中分别注册 ALGO1（D256）与 ALGO2（D512）条目；
3. **分发函数匹配与越界保护**：在 `find_flash_attention.cpp` 中建立 `head_dim == 256 -> return 1` 与 `head_dim == 512 -> return 2` 的映射规则，并在取表前增加算法下标 `[0, count)` 的一致性界限检查，防止表项缺失导致的非法访问；
4. **分页与流契约一致**：保证特化 kernel 同样遵守 `block_size == 32` 分页契约与当前 stream 绑定规则。

在上述前置条件完备前，D256/D512 在本分支严格按未实现（`NOT_IMPLEMENTED` → `TECOOPS_STATUS_NOT_SUPPORTED` 6）fail-closed 拒绝。

#### 插件调用路径（Plugin）的拒绝契约与输出观测边界

`Enqueue()` 调用 C API；不支持的派发配置返回
`TECOOPS_STATUS_NOT_SUPPORTED`（6），插件释放 descriptor 后抛出
`std::runtime_error("plugin_flash_attention rejected parameters (status 6)")`。

负例测试分别验证两层，不能把它们的输出缓冲语义混用：

1. **C API 权威断言**：从 `/proc/self/maps` 找到引擎实际映射的
   `libteco_ops_plugin.so`，记录路径和 SHA256，直接调用该 C API。
   探针独立分配设备输出并预填 7.0；必须返回 status 6，且同一分配读回
   仍全为 7.0。任一条件不满足即失败。
2. **SDK 拒绝断言**：`executor_run()` 必须抛出拒绝异常；异常可见的
   status 必须为 6。部分 SDK 仅透传 `std::exception`，此时仅在上述
   mapped C API 两项硬断言先通过后接受该传输限制，不视为 SDK 已透传状态码。
   未抛错、可读状态非 6、无关异常均失败。
3. **SDK 输出仅作诊断**：`set_output()` 和 `get_output()` 没有已验证的
   buffer alias 保证，测试不使用 SDK 哨兵断言。失败后输出若可读，打印
   `min/max/mean/std/first8`；不可读则明确记录原因。SDK 输出的数值不能
   单独证明 kernel 写入、fallback 或缓冲重分配，也不证明 SDK sentinel untouched。

独立 C API CLI、SDK 测试和无设备 mock 契约命令见
[`README_PLUGIN.md`](../README_PLUGIN.md#flash-attention-拒绝路径复核)。
数值正例的 CPU 参考、finite 检查及 `atol=2e-3, rtol=1e-2` 保持原样。
本次测试修订不修改派发、kernel、C API 或 Python binding，不作性能主张。

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

- Q 按 `BM=128` 分块，K/V 按 `BN=32` 分块
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
│       ├── 0.prototxt ~ 6.prototxt         # 测试用例
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
