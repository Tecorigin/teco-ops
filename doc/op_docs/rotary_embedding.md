# rotary_embedding

## 定位

`rotary_embedding` 是本队贡献的**融合 NeoX RoPE kernel**：对 Q 与 K 同时施加旋转位置编码，
支持 GQA（`num_q_heads != num_kv_heads`）与任意行步长（Q/K 可以来自 fused-QKV 非连续视图，
只要 head 维连续）。`cos_sin_cache` 由调用方预先算好，kernel 只做查表与旋转，不做三角函数。

## 语义（NeoX / half-split）

对每个 token `t`、head `h`、以及 `d ∈ [0, head_dim/2)`：

```
c = cos_sin_cache[pos[t]][d]        # 前半为 cos
s = cos_sin_cache[pos[t]][d + D/2]  # 后半为 sin
x1 = x[t][h][d]
x2 = x[t][h][d + D/2]
out1 = fp16( fp32(x1) * c - fp32(x2) * s )
out2 = fp16( fp32(x2) * c + fp32(x1) * s )
```

`is_neox` 必须为 `True`：本 kernel **只实现 half-split（NeoX）配对**，GPT-J 交错的
`rotate_half` 配对会旋转错误的元素，因此绑定层直接拒绝（fail closed），而不是静默算错。

## 接口

| 层 | 入口 |
| --- | --- |
| Python | `tecoops.rotary_embedding(q, k, cos_sin_cache, positions, q_out, k_out, is_neox=True)` |
| C ABI | `tecoopsRotaryEmbedding(handle, positions, q, k, cos_sin_cache, q_out, k_out, num_tokens, num_q_heads, num_kv_heads, head_dim, q_stride, k_stride, q_out_stride, k_out_stride, cache_stride, is_neox)` |
| Python 测试 | `python_api_test/test_rotary_embedding.py`（含 Hy-MT2-1.8B 真实形状用例） |
| Zoo | `test/zoo/teco/rotary_embedding/`（`test_case/` 通用用例 + `test_case_hy/` Hy-MT2-1.8B 真实形状） |

## 形状 / dtype 约束

- `q`、`k`：FP16，`[num_tokens, heads * head_dim]`（2-D；行 stride 可以大于 head 宽度，
  即允许 fused-QKV 切出来的视图，只要**最后一维（head 维）连续**）。
- `cos_sin_cache`：FP16，`[max_position, head_dim]`，前半 cos / 后半 sin，最后一维连续。
- `q_out`、`k_out`：FP16，形状与 `q`/`k` 一致；**允许与输入别名**（原地旋转）。
- `positions`：`int64`，`[num_tokens]`；绑定层对非 int64 / 非连续输入自动 `.to(kLong)` / `.contiguous()`。
- `head_dim` 必须为正且为偶数，并且整除 Q/K 的行宽（`q.size(1) % head_dim == 0`）。
- `num_q_heads` 与 `num_kv_heads` 分别由 `q.size(1)/head_dim` 与 `k.size(1)/head_dim` 推出，二者可不等（GQA）。
- 算子绑定调用方**当前 SDAA stream**；热路径无分支、无 `getenv`。

## 与参考实现的差异

- 只实现 NeoX half-split 配对；非 NeoX 由调用方拒绝（见上）。
- `cos_sin_cache` 的生成（`inv_freq`、`pos` 组合、cos/sin 拼接顺序）由调用方负责，本算子不做
  `concat` 或 `apply_rotary_pos_emb` 的组合逻辑，因此与具体模型的 cache 布局解耦。
- 旋转在 fp32 中计算、结果按 fp16 舍入；这与参考链的 `x*cos + rotate_half(x)*sin`
  在 fp32 下同序。

## 验证与边界

- 算子级：`python_api_test/test_rotary_embedding.py` 覆盖真实形状（Hy-MT2-1.8B 几何）、
  非默认 stream、原地/非原地输出与拒绝路径；zoo 用例走官方固定命令形态。
- **未覆盖**：plugin 层（`teco/plugin/pluginRotaryEmbedding*` 不存在，推理侧零覆盖）；
  非 NeoX 布局（故意不支持）；`torch.compile`/FakeTensor（无证据，不应视为已支持）。

## 设备门 denormal 掩码（判定依据）

设备门（`gate=device`）的判定不是纯容差断言，而是两层：

1. **先做逐位比对**：`bit_mismatch = 输出与 CPU 参考链逐位不一致的元素数`；
2. **再用掩码分类**：其中落在 denormal 掩码内的部分记为
   `bit_mismatch_in_denormal_mask`，掩码外的部分记为 `bit_mismatch_unmasked`——
   **后者必须为 0**（`unmasked=0`），否则判 FAILED。

**掩码定义**（`denormal_involved`，与 PR #45 rms_norm_compat 同源的方法）：对每个输出
元素，沿参考链舍入序逐中间量检查——`t1..t4 = fp16(x*cos / x*sin / x2*cos / x1*sin)`
四个首次舍入量，以及一次舍入后的 `o1 = t1 - t2`、`o2 = t3 + t4`；只要其中任一中间量
**非零且幅值 < `SUBNORMAL = 2^-14`**（IEEE-754 binary16 最小正规数阈值），该元素的
两个输出半（`[d, D/2)` 与 `[D/2, D)` 成对）即被掩码覆盖。

**为什么这些位模式被允许**：SDAA 对 fp16 次正规数做 flush-to-zero，CPU IEEE 模型
不舍入；两者只在「某操作数或中间舍入结果落入次正规数区间」的元素上可能分叉（表现为
不超过 thresh 的 `max_error` 和数量极少的 `bit_mismatch`）。因此掩码内差异是**环境
语义差异**，不是 kernel 错误；掩码同样只圈「可能受影响的元素」，是保守
超集——`bit_mismatch=4, in_denormal_mask=4, max_error=6.1e-05`（CI 实测）即全部
差异都落在掩码内、且数值在 DIFF1 门限内。判失效只看 `bit_mismatch_unmasked != 0`。

## 模型级接入状态 / 待办

- **已接入**：Hy-MT2-1.8B 线在模型侧通过 `tecoops.rotary_embedding(...)` 调用本算子
  （`hunyuan_v1.py` rope 层，fused-QKV 视图直接传入，行 stride 3072/输出 dense），并
  按该模型 README 记录接入点与 32-token 验证状态。`test_case_hy/` 两个 case 即其
  真实几何。
- **算子级已验证**：官方固定形态 5 个 zoo case `TEST PASSED`；python 测试 CPU 门
  逐位一致（`max_error=0`、`bit_mismatch=0`），设备门 `bit_mismatch_unmasked=0`
  （见上）。
- **未完成 / 待办**：
  - plugin 层不存在（`teco/plugin/pluginRotaryEmbedding*`），推理侧零覆盖；需要经
    Teco-Inference 使用时另补插件与 `test_plugin_*`。
  - 非 NeoX（GPT-J 交错）布局故意不支持，调用方需自行保证 `is_neox=True` 语义。
  - `torch.compile`/FakeTensor 无证据，不应视为已支持。
  - 本 PR 未提交模型级端到端性能主张；模型级数值以模型交付 README 的 32-token
    gate 记录为准，不要按本算子级数据外推。


## 集成待办（plugin / vLLM，当前 N/A）

本算子目前只有 **C API / UAL / Python（`tecoops`）** 三个可执行入口；推理侧的 plugin 与
vLLM 集成**均未实现**，属**显式待办**而不是已覆盖（review 2026-10-20 要求登记）：

| 待办项 | 现状 | 完成标志 |
| --- | --- | --- |
| plugin 层（`teco/plugin/pluginRotaryEmbedding*`） | 不存在 | plugin TU + descriptor 注册齐备（与 `tecoopsRotaryEmbedding` 同一 C ABI） |
| `plugin_test/test_plugin_rotary_embedding.py` | 不存在 | 与其它 op-scoped plugin 用例同形：finite + 非恒定 + 与参考对拍，且打印判定 |
| vLLM 接入（Hot path dispatch） | Hy-MT2-1.8B 线走 Python 绑定（`hunyuan_v1.py` rope 层），未走 vLLM 的 custom-op dispatch | vLLM 侧在**初始化期**一次性绑定 `tecoops.rotary_embedding`（热路径零分支、零 `getenv`），并有 32-token 逐位一致记录 |
| CI 覆盖 | 官方 CI 对 `AFFECT_OPS=rotary_embedding` 的 plugin 阶段为 N/A（无 plugin 文件） | 上述 plugin 落地后该阶段自动可见 |
| FakeTensor / `torch.compile` | 无证据，不应视为已支持 | 有 meta/fake 实现与 graph 测试后再登记 |

约束：本 PR 不改变上述现状，也不把 plugin / vLLM 记为"已支持"。
