# rms_norm_compat

## 定位

`rms_norm_compat` 是 **参考精确（reference-exact）的 RMSNorm epilogue**，用于对齐
Gemma-4 推理链（`vllm.ir.ops.rms_norm` / `RMSNorm.forward_static`）在 FP16 上的
逐位数值行为。与 `rms_norm` 的关键区别：**它不重算行统计量**——每行的 `rstd`
由调用方（参考链本身）作为输入传入，因此该算子只负责 epilogue：缩放 + 权重乘 +
（可选）残差加。

## 语义（与设备 kernel 的注释逐条对应）

- 纯 epilogue（3 输入 / 1 输出）：
  `s = fp16(fp32(x) * rstd)`；`y = fp16(fp32(s) * fp32(w))`
  ——即 **两次舍入**：`x * rstd` 先按 fp16 RNE 舍入，再与 fp16 权重做 fp16 乘。
- 融合残差（4 输入 / 2 输出）：
  `xr = fp32(x) + fp32(res)`；`residual_out = fp16(xr)`；
  `s = fp16(xr * rstd)`；`y = fp16(fp32(s) * fp32(w))`
  ——`residual_out` 的舍入就是参考链里残差分支的那次舍入。

正因为这两次显式舍入与"先算完再舍入一次"不同，本算子才能与参考链逐位一致；
这也是它与 `rms_norm`（单次舍入、内部自行求 `rstd`）不可互换的原因。

## 接口

| 层 | 入口 |
| --- | --- |
| Python | `tecoops.rms_norm_compat(input, weight, rstd, residual, output, residual_out)` |
| C ABI | `tecoopsRmsNormCompat(handle, input, weight, rstd, residual, output, residual_out, num_tokens, hidden_size)` |
| UAL | `teco/ual/ops/rms_norm/find_rms_norm_compat.cpp` → `teco_slave_rms_norm_fp16_compat[_add]` |
| Zoo | `test/zoo/teco/rms_norm_compat/`（case_0 纯 epilogue，case_1 融合残差） |
| Python 测试 | `python_api_test/test_rms_norm_compat.py` |

## 形状 / dtype 约束

- `input`、`residual`、`output`、`residual_out`：FP16，`[N, D]`，连续。
- `weight`：FP16，`[D]`。
- `rstd`：**FP32**，`[N]`（不是 FP16；kernel 按 `const float *` 读取）。
- `D` 必须为 32 的倍数：epilogue 以 16 宽 SIMD 前进，且 tile 宽度为 32
  （`head_dim % 32 != 0` 时尾列不会被写）。
- `residual` 与 `residual_out` 必须同时给出或同时为 `nullptr`，否则 C API 返回
  `TECOOPS_STATUS_BAD_PARAM`；`input/weight/rstd/output` 为 `nullptr` 同样拒绝。
- 算子绑定调用方**当前 SDAA stream**（保证它排在产出 `rstd` 的统计 kernel 之后），
  热路径无分支、无 `getenv`。

## denormal 掩码（测试判定依据，自解释用）

Python 测试有两层门禁，第二层就是 denormal 掩码：

1. **主门禁**：与「按文档舍入序、用设备上的 torch 语义求值的参考 epilogue」**逐位一致**
   （`max_ulp = 0`，`bit_identical=True`）。
2. **次门禁（`vs CPU-IEEE`）**：与「硬件无关的 IEEE fp16 CPU 模型」（同一舍入序，但
   不做 flush-to-zero）逐元素对比；任何不一致位都必须落在 **denormal 掩码**内
   （`unexplained=0`）。

**掩码定义**（`denormal_involved_mask` / `denormal_mask_residual_out`）：对每个元素，
沿参考链的舍入序逐中间量检查——`x`、`w`、（残差时）`residual` 与 `xr = x + res` 本身的
幅值，以及中间舍入量 `scaled = fp16(x * rstd)` 和最终乘积 `scaled * w` 的幅值——只要其中
**任一中间量的幅值（非零时）落在 `SUB_SAFE = 4 × 2^-14`以内**，该元素即被掩码覆盖。
`2^-14` 是 IEEE-754 binary16 的最小正规数阈值，`SUB_SAFE` 是把该区域**放大 4 倍**的
保守包络（对本硬件 fp16 flush-to-zero 可能作用于操作数或舍入结果的区域取超集）。

**为什么这些位模式被允许**：SDAA 设备会把 fp16 次正规数（subnormal）flush 到零，
而 IEEE CPU 模型不舍入 flush；两者只在「某操作数或某舍入中间结果是次正规数」的元素上
可能分叉。掩码圈出的正是这些元素，因此「掩码内不一致」是**环境语义差异**而非实现错误。
于是日志行形如 `vs CPU-IEEE: ndiff=1 (1 in denormal mask, mask covers 1)  unexplained=0
PASSED` 的含义是：与 IEEE 模型有 1 个元素不同、它恰在掩码内、掩码共覆盖 1 个元素、
即 0 个无法解释的差异。反过来，**`mask covers N` 大于 `ndiff` 也完全正常**——掩码是
保守超集，覆盖 N 个元素只说明这 N 个元素*可能*受 flush 影响，不代表它们实际分叉；
判据只看 `unexplained != 0`（掩码外的不一致）必为 0。

## 与 `rms_norm` 的差异（不可混用）

`rms_norm_compat` 与 `rms_norm` 是同族但不可互换的两个入口。区别来源：Gemma 参考链
（`vllm.ir.ops.rms_norm` / `RMSNorm.forward_static`）要求 `rstd` **由统计 kernel 产出后
被 epilogue 原样消费**，且 `x*rstd` 与*w* 的乘法**各做一次 fp16 舍入**；通用
`rms_norm` 在算子内部重算统计量、单次舍入，数值路径与这条链不可能逐位一致。

| 项 | `rms_norm` | `rms_norm_compat` |
| --- | --- | --- |
| 行统计量 | 算子内计算 `sum(x^2)` | **调用方传入 `rstd`（fp32）** |
| `eps` | 有 | 无 |
| 舍入 | `x * rstd * w` 后一次舍入 | `x * rstd` 先舍入到 fp16，再与 `w` 做 fp16 乘 |
| 残差 | 可选（同 `rstd` 语义） | 可选（`residual_out` 为参考链的舍入结果） |
| 适用 | 常规 decode/prefill 的行归一化；调用方不需要与 Gemma 数值链逐位一致 | Gemma 参考链的 epilogue；需要与参考 `token_ids` 逐位对齐（例如后续 softmax/采样对单 ULP 敏感时） |

选型一句：需要「reference-exact 逐位一致」才用 `rms_norm_compat`（并接受多一个
`rstd` 输入的调用序）；否则用更常规、少一个输入的 `rms_norm`。两者输出可以逐位
不同（如上表），测试中 `rms_norm_compat` 用 `rstd` 喂入以证明不内部重算。

## 验证状态与边界

- 官方固定形态 zoo：`./build/demo --gid=0 --perf_repeat=50 --warm_repeat=3 --gtest_repeat=1 --case_path=zoo/teco/rms_norm_compat/test_case/<c>.prototxt`；
  参考实现按上文两次舍入写，与设备输出逐例对照（DIFF1 门限 1e-2，实测见 PR #45 评论）。
- Python 侧：`python_api_test/test_rms_norm_compat.py` 覆盖绑定/导出契约、非默认
  stream、`rstd` 被原样消费（喂入被缩放过的 `rstd` 以排除"内部重算"）、拒绝路径，
  以及与 `tecoops.rms_norm` 残差变体的对照断言（`residual_out` 逐位一致，`out` 在 1 ULP 理论容限内）。
- **未覆盖**：plugin 层（`teco/plugin/pluginRmsNormCompat*` 不存在，推理侧零覆盖）、
  FakeTensor/meta 与 `torch.compile`（无证据，不应视为已支持）、多流并发。

## `SUB_SAFE = 4 × 2^-14` 的可复现测量方法

`SUB_SAFE` 是**保守包络**而不是精确阈值，来源可以完全复算：

- `2^-14` 是 IEEE-754 binary16 的最小正规数。设备若把幅值 **< 2^-14** 的 fp16 操作数/舍入
  结果 flush 到零，则任何"真实幅值 < 4 × 2^-14"的中间量都可能受影响，故掩码取 4 倍超集
  （`2^-12`）。
- 掩码只决定"哪些差异**允许**"；判定仍然是 `unexplained == 0`。掩码一旦变松，唯一后果是
  "allowed"集合变大，因此需要下面的测量步骤在环境变化后重新核对，而不是相信常数本身。

**记录环境**（本库 CI / 队内 SDAA 服务器，c44–c47 窗口）：SDAA Driver 3.2.0、SDAA Runtime 3.2.0、
TecoDNN 3.2.0、Torch-SDAA `20260623.8.51`（`+gitd942f23`）、PyTorch `2.12.0a0+git0d62256`，
4 个逻辑设备。**driver / TecoDNN 版本变化时必须重跑本测量**——`SUB_SAFE` 只对可复现出
"flush 区不超出 IEEE 次正规区"的环境族有效。

**测量步骤（不依赖本仓库构建，只需要 `torch_sdaa`）**：

```python
import torch
import torch_sdaa  # noqa: F401  (registers the sdaa backend)

cpu, dev = torch.float16, "sdaa"
rows = []
for e in range(-30, -9):                     # 覆盖次正规区 ~ 正规区
    x = torch.tensor([2.0 ** e], dtype=cpu)  # 精确可表示的幅值
    w = torch.tensor([1.5], dtype=cpu)       # 乘积 = 1.5 * 2^e
    got = (x.to(dev) * w.to(dev)).cpu()      # 设备 fp16 乘
    want = x * w                             # CPU IEEE fp16 乘（不舍入 flush）
    same = bool(got.view(torch.int16) == want.view(torch.int16))
    rows.append((e, float(want[0]), float(got[0]), same))
    print(e, float(want[0]), float(got[0]), same)
```

**判据**：

1. 取**设备与 CPU 首次不一致的最大指数** `e*`（设备把该幅值 flush 成 0、CPU 保留次正规数）；
2. 若 `e* <= -14`（设备只 flush IEEE 次正规区）⇒ 当前 `4 × 2^-14` 包络成立；
3. 若 `e* > -14`（设备 flush 到正规区）⇒ 必须把 `SUB_SAFE` 提高到 `>= 4 × 2^e*`，
   并重跑 `python_api_test/test_rms_norm_compat.py`，要求 `unexplained == 0` 且
   `mask covers` 仍远小于元素数（本次 CI 实测 `mask covers` 为 0–1 / 3840，见 PR #45 评论）；
4. 记录设备型号、driver/TecoDNN 版本与测量日期，替换本节"记录环境"。

**现状**：该常数是按上面的解析推导引入的；本轮（cycle-47）**没有**在设备上重跑这条探针（当时
无法从历史记录确认是否曾单独测过）。历史 device 门禁一直以 `unexplained=0` 为判据，且 CI 运行的
`mask covers` 仅 0–1 个元素，说明包络并未在承担掩盖作用。本步骤供任意一台待验证机器复核。

## 集成待办（plugin / vLLM，当前 N/A）

本算子目前只有 **C API / UAL / Python（`tecoops`）** 三个入口；推理侧集成**未实现**，是
**显式待办**而不是已覆盖（review 2026-10-20 要求登记）：

| 待办项 | 现状 | 完成标志 |
| --- | --- | --- |
| plugin 层（`teco/plugin/pluginRmsNormCompat*`） | 不存在 | plugin TU + descriptor 注册齐备，且 `plugin_test/test_plugin_rms_norm_compat.py` 存在 |
| plugin smoke / 数值门禁 | 无 | 与其它 op-scoped plugin 用例同形（finite + 非恒定 + 与参考对拍） |
| vLLM 接入（Gemma 推理链） | 尚未接入 vLLM 的 custom-op dispatch；当前 Gemma 线仍走原生路径 | vLLM 初始化期一次性绑定 `tecoops.rms_norm_compat`（热路径零分支、零 `getenv`），并有 32-token 逐位一致性记录 |
| CI 覆盖 | `AFFECT_OPS=rms_norm_compat` 的 plugin 阶段为 N/A（无 plugin 文件） | 上述 plugin 落地后自动可见 |
| FakeTensor / `torch.compile` | 无证据，不应视为已支持 | 有 meta/fake 实现与 graph 测试后再登记 |

约束：本 PR 不改变上述现状，也不把 plugin / vLLM 记为"已支持"。

## 与 #37（rms_norm 性能线）的合入顺序

本 PR 与 #37 共用 `teco/ual/ops/rms_norm/`、`teco/ual/kernel/rms_norm/` 目录：#37 重写
`rms_norm_fp16.scpp` 的 current-stream + DMA overlap 路径，本 PR 新增 `rms_norm_compat` 的
参考精确 epilogue（两者文件不重叠）。合并前需在同一 fixture 上交叉确认两点：

1. `n % 16` 与 fp16 epilogue 的数值口径一致：同一输入下 `rms_norm` 与 `rms_norm_compat`
   的差异只应来自 Gemma 的 `1 + weight` 参考语义，不应出现额外 ULP 漂移；
2. `python_api_test/test_rms_norm_compat.py` 的 `unexplained == 0` 在 #37 的 kernel 改动之上
   仍成立（兼容层不依赖被 #37 改写的中间缓冲布局）。

建议合入顺序：**#37 → 本 PR**，随后分别重跑 `AFFECT_OPS=rms_norm` 与
`AFFECT_OPS=rms_norm_compat` 的 python/plugin 阶段。

## 来源与追溯（Provenance）

- 团队来源仓：`teco-ops-SCUdoudui` main `459992a29aadd070b4d987d99c958de40911cae8`
- 来源提交：`b374322`
- 官方 PR 落地提交：`8d5e44b`
- 以上与 PR 描述一致，仅用于审计追溯；数值口径与验证结论以本文上方各节为准。
## 真实 shape 覆盖（zoo）

模型侧真实调用形状取自 Gemma-4-12B 的 T1 逐位证据（2026-10-12，
`op_learning/norm/gemma-rmsnorm-compat-t1-20261011/evidence/numeric-t1-85shapes.json`：85 个 case，
`max_ulp = 0` 且 `bit_identical = true`）：

| 真实 case | shape | has_weight | residual | 覆盖位置 |
| --- | --- | --- | --- | --- |
| hidden2d（final norm） | `[1, 3840]` | 是 | 否 | python 测试 |
| q_full / k_full / v_full | `[1, 8, 512]` / `[1, 1, 512]` | 是 / 否 | 否 | `case_2`（D=512，epilogue-only） |
| 残差 add 路径 | `[1, 8, 512]` 等 | 是 | 是 | `case_3`（D=512 + residual） |
| prefill 行数扫描 | `[2..64, 3840]` | 是 | 否 | python 测试 |
| 原 zoo 几何 | `[64, 4096]` / `[32, 4096]` | 是 | 否 / 是 | `case_0` / `case_1` |

zoo 原来的 2 个 case 只覆盖 D=4096；新增 `case_2` / `case_3` 把 **D=512**（Gemma full-attention
投影宽度）与 **D=512 + residual** 补进来，与真实调用宽度对齐；D=3840 的 final-norm 宽度已由
python 测试的 85 个 case 逐位覆盖，不重复搬进 zoo。
