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

## SIMD RMSNorm output epilogue (2026-10-07)

The plain and fused-add FP16 kernels vectorize only their second-pass output
loops: widen16 FP16 input/residual and weight elements, multiply by rstd and
weight in the same FP32 order, then narrow16 FP16 outputs. Scalar tails remain.
Sum-of-squares order, residual half rounding, buffers, DMA, row partition and
public ABI remain unchanged. Inverting these two replacements restores the
entire baseline source byte-for-byte. No runtime backend switch is introduced.

Independent evidence is from model/internvl3_5-8b, baseline model commit
041903e99f6f9cd7625ad62f96ff806e12f502de and PR37 baseline
8f896f2a9103cc9c684eb4488c9f885a38f15eb6 on official main
de27305efed0a17ae926d21d5415d8b915614649. The candidate archive differs from
that complete559-file source archive in one file only. Kernel source SHA256 is
41a517c23ae849f0d36cbdb2746ea2b75e21e5f2efe4dbaf44b6ed99a62193be.
The selected baseline/candidate core SHA256 values are respectively
86576a574ef6c3cd52521245d5e048c9981a8444e6e4ba36822b9bb0f3a04d95 and
d26791a5c1f9d902f2449baf62c58490aafa455cab357f8b93bbc029f525da97.
Both builds use the recorded WITH_TORCH=ON/plugin=OFF recipe, vendor SDK and teco-hal0.0.2 dependency. Candidate flags retain -O3/-msimd/-flto and the unchanged CMake link contract. The prior baseline build cache/compiler binary hash was not retained; matching historical flags are inferred from its saved recipe and identical build-source files, rather than a fresh baseline rebuild.
Binding source/ABI is unchanged; independently built extension binaries have
different recorded digests. Vendor Python resolves to
/usr/local/python/bin/python3.12, SDK/runtime3.2.0, Torch2.12.0a0+0d62256 and
Torch-SDAA20260623.8.51+d942f23.

All32 paired cases pass original plain maxabs<.005 and add/residual<.01
reference thresholds. Candidate output and residual bits equal their own
baseline. Real hidden4096 and q/k128 shapes, default/nondefault streams,
preserved inputs, signed/subnormal inputs and legal non-vector tails30/126/130
are covered. Baseline HAL half-DMA cannot handle the probed odd-D31 alignment;
this existing limitation and CPU/SDAA residual rounding differences are kept
in raw evidence. CPU residual bit identity is not the reference gate.
The full model branch overlay CPU suite, syntax and diff checks are retained
with that branch's receipt.

The same accepted public launcher uses TP2/FP16/context4352, unchanged fixed
text/image inputs, greedy32, seed0, two warmup sets and three timed sets. Each
worker proves145 actual TecoopsRMSNorm modules, nonzero calls and its unique
mapped core. Candidate steady318.024065s/16rounds/48requests matches all
baseline token IDs. Both variants peak11.713GiB allocated/13.039GiB reserved
per worker, and owned services release to0MB.

Model raw seconds A→B:
text0 [6.555083294,6.560870825,6.556684889] →
[6.462610320,6.463274888,6.464103465], med6.556684889 →6.463274888;
text1 [6.555170664,6.576936734,6.556852598] →
[6.468733341,6.463693367,6.461631934], med6.556852598 →6.463693367;
image [6.875671206,6.877197781,6.873915271] →
[6.747882526,6.762659439,6.771143701], med6.875671206 →6.762659439.
These single-order observations do not establish causal end-to-end speedup.

The following independent operator measurements use FP16, visible device1,
seed20261007+shape index, CPU threads4, warmup5 and10 calls/trial. Three raw
values and their median are shown for every actual shape/mode. Long hidden
plain cases improve3.20–3.21x, fused-add1.95x, long q/k plain2.59–2.63x,
fused-add1.79–1.80x. Short D128 results have overlapping noise, including
<1% negative median fluctuations; no blanket gain is claimed.

Current-head official CI/full vendor wheel, official py311 and committee
accuracy were not run. Historical current-stream evidence remains tied to its
old builds. Hy's independent validation of this epilogue is recorded below;
MiniCPM's historical RMSNorm evidence does not validate the new epilogue.
InternVL's independent proof and reproducible minimal source patch are
published in the InternVL model PR validation folder.



Historical table scope correction (2026-10-08): q/k rows57952/14488 and139264/34816 use global head counts. They are boundary/stress shapes, not actual TP2 per-rank inputs. Hidden4096 rows are unaffected. Original measurements remain unchanged; current per-rank shapes are independently measured in the plain-DMA section below.

| Shape /mode | Baseline ms/call (three raw values) | Candidate ms/call (three raw values) | Median baseline →candidate |
| --- | --- | --- | --- |
| [1,4096] /plain | [0.066113798,0.067154894,0.067022804] | [0.025885901,0.025573000,0.025548902] | 0.067022804 →0.025573000 |
| [1,4096] /add | [0.091906800,0.091627805,0.091569702] | [0.050649897,0.052085798,0.052256905] | 0.091627805 →0.052085798 |
| [8,4096] /plain | [0.067773898,0.068025803,0.067013805] | [0.027551898,0.025728001,0.025627902] | 0.067773898 →0.025728001 |
| [8,4096] /add | [0.092133798,0.091591699,0.091493805] | [0.052483904,0.052559801,0.052486901] | 0.091591699 →0.052486901 |
| [32,4096] /plain | [0.068838795,0.068959797,0.068967801] | [0.027511897,0.027746899,0.027516001] | 0.068959797 →0.027516001 |
| [32,4096] /add | [0.095633703,0.095440802,0.095334800] | [0.056449801,0.054128899,0.054208899] | 0.095440802 →0.054208899 |
| [128,4096] /plain | [0.247005397,0.248339301,0.246915402] | [0.084424805,0.084232801,0.084340700] | 0.247005397 →0.084340700 |
| [128,4096] /add | [0.349174201,0.349210197,0.347794103] | [0.185250601,0.185871602,0.185631501] | 0.349174201 →0.185631501 |
| [1811,4096] /plain | [3.397346399,3.397505300,3.398384398] | [1.059695304,1.059824298,1.060448296] | 3.397505300 →1.059824298 |
| [1811,4096] /add | [4.800431797,4.798293801,4.797526804] | [2.465498698,2.463450696,2.463764796] | 4.798293801 →2.463764796 |
| [4352,4096] /plain | [8.094398497,8.094457397,8.092970395] | [2.518436598,2.518525603,2.518871595] | 8.094398497 →2.518525603 |
| [4352,4096] /add | [11.440060998,11.438460002,11.438516900] | [5.864734098,5.862259102,5.863305100] | 11.438516900 →5.863305100 |
| [57952,128] /plain | [3.708628600,3.708148503,3.707611602] | [1.413057401,1.413464395,1.414377399] | 3.708148503 →1.413464395 |
| [57952,128] /add | [5.160705902,5.161911901,5.160196900] | [2.868243697,2.871029702,2.869622700] | 5.160705902 →2.869622700 |
| [14488,128] /plain | [0.932800700,0.931265595,0.930370600] | [0.359712099,0.358779100,0.358977099] | 0.931265595 →0.358977099 |
| [14488,128] /add | [1.293770695,1.293881703,1.295365702] | [0.722641102,0.723788200,0.722667202] | 1.293881703 →0.722667202 |
| [139264,128] /plain | [8.903905400,8.903784299,8.901546401] | [3.387093404,3.386448399,3.387227404] | 8.903784299 →3.387093404 |
| [139264,128] /add | [12.391595502,12.392107502,12.401278497] | [6.883817504,6.881178502,6.883541500] | 12.392107502 →6.883541500 |
| [34816,128] /plain | [2.232089400,2.233631298,2.235715301] | [0.853396801,0.852543901,0.852858799] | 2.233631298 →0.852858799 |
| [34816,128] /add | [3.106193099,3.104600101,3.104725096] | [1.726694603,1.727896597,1.726452599] | 3.104725096 →1.726694603 |
| [32,128] /plain | [0.023057900,0.023155997,0.023175898] | [0.020839996,0.020640896,0.020774000] | 0.023155997 →0.020774000 |
| [32,128] /add | [0.023417897,0.023027998,0.023428898] | [0.024035899,0.021677901,0.023622997] | 0.023417897 →0.023622997 |
| [8,128] /plain | [0.020782999,0.021244900,0.023766002] | [0.024345901,0.021030998,0.021158898] | 0.021244900 →0.021158898 |
| [8,128] /add | [0.023196999,0.023198902,0.021142000] | [0.022432901,0.023656996,0.023321901] | 0.023196999 →0.023321901 |

Public proof source: InternVL model PR6 commit
4f01e16ad891dea1ce88aa3b313735fdd2e09fda stores
model_adaptations/InternVL3_5SCUdoudui/validation/rms_epilogue_simd_20261007.json
and validation/epilogue-only.patch. Its README pins the isolated source
build to operator code commit e29b53c256f366e6eee538d656bfbb56e867cefc.
This later documentation only identifies that public proof; the tested
kernel bytes and their source checksum remain unchanged.

## Hy-MT2 independent FP16 epilogue measurements (2026-10-07)

This model branch independently tests PR37 SIMD source e29b53c256f366e6eee538d656bfbb56e867cefc (kernel SHA256 41a517c23ae849f0d36cbdb2746ea2b75e21e5f2efe4dbaf44b6ed99a62193be). The geometry comes from its own hidden2048 and Q/K128 profile; the eight measurements use FP16, epsilon1e-5, seed20261007, warmup25, wall100 iterations and profiler30 iterations per run, on physical device1 / local sdaa:0. All three raw values are retained below, in ms/call. These are separate Hy results; the InternVL results above are not reused as its validation.

| Hy shape/mode | Baseline device raw | SIMD device raw | Median baseline / SIMD |
| --- | --- | --- | --- |
| hidden_decode [1,2048] | 0.031774, 0.031734, 0.031776 | 0.011307, 0.011267, 0.011273 | 0.031774 / 0.011273 |
| hidden_decode_residual [1,2048] | 0.044166, 0.044159, 0.044144 | 0.024502, 0.024474, 0.024478 | 0.044159 / 0.024478 |
| hidden_prefill [30,2048] | 0.03243, 0.03241, 0.032451 | 0.01198, 0.011955, 0.011948 | 0.032430 / 0.011955 |
| hidden_prefill_residual [30,2048] | 0.045586, 0.045647, 0.045586 | 0.025911, 0.025857, 0.025884 | 0.045586 / 0.025884 |
| q_decode [16,128] | 0.003811, 0.003832, 0.003814 | 0.002612, 0.002583, 0.002583 | 0.003814 / 0.002583 |
| k_decode [4,128] | 0.003761, 0.003764, 0.003772 | 0.002538, 0.002529, 0.002524 | 0.003764 / 0.002529 |
| q_prefill [480,128] | 0.032549, 0.032525, 0.032537 | 0.013541, 0.013546, 0.013537 | 0.032537 / 0.013541 |
| k_prefill [120,128] | 0.01001, 0.009998, 0.010078 | 0.004972, 0.004992, 0.004977 | 0.010010 / 0.004977 |

| Hy shape/mode | Baseline wall raw | SIMD wall raw | Median baseline / SIMD |
| --- | --- | --- | --- |
| hidden_decode [1,2048] | 0.062152, 0.062351, 0.062223 | 0.062266, 0.062562, 0.061947 | 0.062223 / 0.062266 |
| hidden_decode_residual [1,2048] | 0.07448, 0.074606, 0.074794 | 0.074172, 0.07433, 0.074303 | 0.074606 / 0.074303 |
| hidden_prefill [30,2048] | 0.062885, 0.063306, 0.06605 | 0.061374, 0.060719, 0.060869 | 0.063306 / 0.060869 |
| hidden_prefill_residual [30,2048] | 0.073059, 0.073555, 0.073142 | 0.073195, 0.126163, 0.07235 | 0.073142 / 0.073195 |
| q_decode [16,128] | 0.062829, 0.062723, 0.062826 | 0.060086, 0.059596, 0.059731 | 0.062826 / 0.059731 |
| k_decode [4,128] | 0.062454, 0.062992, 0.062629 | 0.059936, 0.059633, 0.059698 | 0.062629 / 0.059698 |
| q_prefill [480,128] | 0.061715, 0.061502, 0.061615 | 0.062499, 0.062401, 0.062312 | 0.061615 / 0.062401 |
| k_prefill [120,128] | 0.061522, 0.061763, 0.061448 | 0.060515, 0.060371, 0.060122 | 0.061522 / 0.060371 |

Only device kernel time shows a consistent reduction. Wall time is mostly dominated by dispatch, includes the 0.126163ms candidate outlier and the slightly negative Q-prefill median, and does not establish a stable wall or model speedup. The own FP16 wrapper focused suite passes36 cases/four graphs plus Fake/CPU rejection with its unchanged0.002 tolerances and baseline/candidate bitwise equality. Native BF16 capability is a separate deferred attempt because this public profile is FP16; it is not part of these measurements.

A separate controlled Hy model supplement uses the accepted public run.sh,
TP1, FP16, context8192 and the default compiler. Diagnostic worker RPC is
enabled with VLLM_SERVER_DEV_MODE=1 and a worker extension. Both arms select
129 TecoopsRMSNorm modules with SDAA FP16 weights and epsilon1e-5. Before/after
worker receipts retain the actual unique mapped extension/core paths and
SHA256 values for each isolated package:

- Baseline core: 86576a574ef6c3cd52521245d5e048c9981a8444e6e4ba36822b9bb0f3a04d95;
  extension: 6e10da7c805d24d431da546d75488695f7abcae1d90c2e7e767f0fc5fde521d6.
- Candidate core: d26791a5c1f9d902f2449baf62c58490aafa455cab357f8b93bbc029f525da97;
  extension: c7d4a8c02138d0742e1da46e9aff79515d75b364b02171fd0b03e34db5cc3f4b.

Both own text prompts have identical prompt IDs and all32 greedy output IDs
in both arms and match this model branch's prior own references. Both process
peaks, including startup with no reset, are14455050752 bytes allocated and
14816378880 bytes reserved (13.462315 /13.798828 GiB), unchanged after the two
requests. Services stop and physical device1 is released to0 MB/no process.
The original A/B did not retain module/maps/peak data; its original incomplete
fields remain unchanged. This later supplement supplies independent evidence;
module enumeration is not a forward invocation total or a speed measurement.

The public own proof and epilogue-only.patch are in
[Hy model PR5](https://github.com/Tecorigin/tecovllm-modelzoo/pull/5), model
commit2c4378bf0a0f5794b0b3d1fd5a35d2d5e947ce95, under
model_adaptations/HyMT2SCUdoudui/validation/rms_epilogue_simd_20261007.json.
Its canonical folder tree b25d30b56a1e93056ad23e9c5be0d7e60d0d7f15 exactly
matches remote source/proof commit c36d277bebc46c49a08037a37b9dbc5a21cd00b7.
The accepted public runtime/launcher source is unchanged. Earlier303-second
attention evidence is historical, not a fresh epilogue steady test. This
epilogue-only work does not change attention/cache/communication. Current
head official CI/full wheel, official py311, full8192 request and committee
accuracy were not run. No whole-model or stable host wall speedup is claimed.

## InternVL plain output DMA wait overlap (validated2026-10-07; published2026-10-08)

The plain FP16 kernel keeps each output buffer owned by its DMA until the
existing same-buffer reuse wait. It removes the immediate post-store wait and
drains both output handles before freeing SPM. Idle handles start reply/counter
0/0 and their wait is valid. Input/output buffers are distinct. Compute, epsilon,
FP32 reduction/order, SIMD pack, current-stream binding and ABI are unchanged.
The add path stays byte-identical because its residual prefetch may overwrite
the store source. No extra allocation or backend branch is added.

Atomic source delta applies to e29b53c/2949f70, on official main
de27305efed0a17ae926d21d5415d8b915614649. Kernel SHA256
41a517c23ae849f0d36cbdb2746ea2b75e21e5f2efe4dbaf44b6ed99a62193be →
93e8e80d08d401fffaa3060bf8e6e24bd989ab59fd8cb09c079228710f66d5b7. U0 patch
1408340f6b04fc98187d41c5da2095d1782d65e2bcb54a424121815eed811424 reconstructs the tested source exactly.
All559 regular source files were compared; only this kernel changed and binding
source is unchanged. Candidate core ed1358d69, extension e34873d80; accepted E29
core d26791a5/extension c7d4a8c0. These are distinct built artifacts; controls
and A-after are retained to avoid short-shape/compiler timing attribution.

Original focused32 plus4 output-reuse stress plus12 real-TP2 records pass48/48;
candidate and A-after outputs/residuals are bitwise equal to E29. Original CPU
limits .005 plain/.01 add and residual remain. CPU residual bit differences are
reported, not relabeled as exact CPU equality. Default/nondefault streams,
inputs, Fake shape128/4096 and8 real fullgraph backend=eager comparisons per
arm pass. This is structural graph testing, not SDAA device graph capture.

Live36 attention modules per worker confirm TP2 heads16/4,D128 and q/k norm
weights[128],FP16,epsilon1e-6 on each SDAA device. Historical q57952/k14488
(and q139264/k34816) were global-head boundary tests; they are not the per-rank
TP2 shapes. Their original values stay unchanged. The target per-rank matrix
below uses q28976/k7244 at1811 and q69632/k17408 at4352.

Timer is perf_counter with synchronize around the call loop: Python/dispatch
are included. FP16,seed20261007,device1,warmup5,10calls/trial and3trials are
fixed; these are synchronized operator-call measurements, not event-only
device-kernel times.

| Plain shape | A-before three ms/call | Candidate three ms/call | A-after three ms/call | Median A-before / candidate / A-after ms |
| --- | --- | --- | --- | --- |
| [1811, 4096] | [1.061081403, 1.061447401, 1.059555303] | [1.042137400, 1.042557397, 1.042520400] | [1.061078400, 1.060778298, 1.061361399] | 1.061081403 / 1.042520400 / 1.061078400 |
| [4352, 4096] | [2.518270799, 2.518669702, 2.517678798] | [2.478351799, 2.474988898, 2.475429798] | [2.518803801, 2.517313702, 2.518282796] | 2.518270799 / 2.475429798 / 2.518282796 |
| [28976, 128] | [0.711986300, 0.711632305, 0.710182299] | [0.619505503, 0.619515497, 0.617560500] | [0.711528305, 0.711147301, 0.711093302] | 0.711632305 / 0.619505503 / 0.711147301 |
| [7244, 128] | [0.182422501, 0.183323602, 0.183632498] | [0.160160603, 0.161574600, 0.160954602] | [0.183589599, 0.182428496, 0.183445599] | 0.183323602 / 0.160954602 / 0.183445599 |
| [69632, 128] | [1.696930901, 1.697163895, 1.697336999] | [1.474340499, 1.473846496, 1.473919401] | [1.698199904, 1.697528898, 1.697893901] | 1.697163895 / 1.473919401 / 1.697893901 |
| [17408, 128] | [0.430042000, 0.430427998, 0.430921902] | [0.374581100, 0.374914100, 0.374543096] | [0.430993002, 0.429889001, 0.429971900] | 0.430427998 / 0.374581100 / 0.429971900 |

The measured long plain hidden calls improve1.70–1.75% and per-rank Q/K calls
12.20–13.19% against A-before/A-after. No short-shape improvement is claimed;
unchanged add controls also move and short hidden8 has a small negative
observation. All36 matrices/control raw triples are in the own public proof.

The unchanged public TP2/FP16/context4352 launcher passes own text/image
greedy32 before/after this change,2warmup sets+3timed sets. Candidate additionally
passes16 steady rounds over317.156770s. Workers
prove145 selected RMS modules/rank and the unique candidate DSO maps; per-worker
startup/request-window peaks are retained. HTTP model timing:

| Fixed input | Baseline seconds (three raw) | Candidate seconds (three raw) | Median baseline / candidate seconds |
| --- | --- | --- | --- |
| text_0 | [6.465096010, 6.467103455, 6.470622017] | [6.466311547, 6.477757190, 6.470496897] | 6.467103455 / 6.470496897 |
| text_1 | [6.468929351, 6.467297265, 6.467069926] | [6.463382384, 6.462565117, 6.466762146] | 6.467297265 / 6.463382384 |
| image | [6.753588631, 6.754539259, 6.750168019] | [6.749954160, 6.750640298, 6.751550106] | 6.753588631 / 6.750640298 |

Model latency is effectively unchanged; no end-to-end speedup claim. The
metadata-only restart confirms head attributes and supplies no new generation
or performance evidence. Vendor-Python AST/shell syntax, U0 source reconstruction
and git diff --check pass. Current-head official CI/full wheel,official py311,
committee accuracy and new long-input accuracy were not run. The existing odd-D
4B HAL alignment limitation is unchanged.

Own public source/proof: InternVL model PR6,
validation/rms_plain_writeback_20261007.json, remote commit1a0b3e77c46ebb4160df413101049f31db38f11e.
Hy's preceding evidence continues to validate its earlier e29b53c SIMD source,
not this new plain-DMA code. MiniCPM has not yet independently validated this
new source.

## MiniCPM5-1B 独立 SIMD epilogue 验证（2026-10-08）

### 来源与范围

本轮只验证 RMSNorm plain/add 第二遍 epilogue SIMD；没有改变 DMA overlap。候选 B 对应官方 E29 commit `e29b53c256f366e6eee538d656bfbb56e867cefc`，`teco/ual/kernel/rms_norm/rms_norm_fp16.scpp` SHA-256 `41a517c23ae849f0d36cbdb2746ea2b75e21e5f2efe4dbaf44b6ed99a62193be`。实测 B 源码快照的归档 SHA-256 为 `95b0ea93087eca250aaab7c84cba8ab776e3bcbaac3539a8cf505d49d41aa612`；它保留旧说明文档，因此不是官方 E29 Git 归档的字节哈希。源码审计对照 559 个文件：仅 `doc/op_docs/rms_norm.md` 不同，其余 558 个文件（包含全部构建源码与 kernel）逐字一致。

A 是 fresh pre-SIMD 重建（kernel SHA-256 `7a8d56abe6d466c8478da73f2eb5183194d2ec8e4c5a9d354dc971caec2e6f86`），B 是上述 E29 重建；A-after 是同一 A artifact 的复测。A 的 `libteco_ops.so` / `_torch_ext` SHA-256 分别为 `7fc428538a0f01d5eeb1863a56955accc14ae52d1ee95e59fb3dd9ebc7ff6b84` / `7d02b580ed7fd023a0c46cbf63fcb45e863f32c2c0068569a404f7c553fb8c29`；B 分别为 `d26791a5c1f9d902f2449baf62c58490aafa455cab357f8b93bbc029f525da97` / `c7d4a8c02138d0742e1da46e9aff79515d75b364b02171fd0b03e34db5cc3f4b`。A 在本轮重新编译；B 重用已保存的 E29 编译产物。厂商编译器哈希和生成的 kernel/extension flags 在仅替换构建路径后相同。本轮 correctness、计时和模型门禁均在 MiniCPM 分支独立执行，没有与当前 installed `fe` 产物建立源码等价关系。

### MiniCPM 实际 shape focused 数据

以下为同一输入、同步 host operator-call 范围的三次原始计时与中位数，单位 ms。plain/add 原始 oracle 阈值分别为 `0.005` / `0.01`，residual 阈值为 `0.01`；四组 A、B、A-after 输出互相 bit-exact。表中 maxabs 为输出相对 oracle 的最大绝对误差。

| shape / mode | A raw → median (ms) | B raw → median (ms) | A-after raw → median (ms) | maxabs A/B/A-after; residual maxabs |
|---|---|---|---|---|
| `[7,1536]` plain | `[0.028248949092812836, 0.028430900420062244, 0.028186451527290046]` → `0.028248949092812836` | `[0.014917951193638146, 0.014639951405115426, 0.014368450501933694]` → `0.014639951405115426` | `[0.027924400637857616, 0.027848451281897724, 0.02764240198303014]` → `0.027848451281897724` | `0.001953125 / 0.001953125 / 0.001953125`; N/A |
| `[7,1536]` add | `[0.03762540000025183, 0.03784035216085613, 0.037831399822607636]` → `0.037831399822607636` | `[0.022109400015324354, 0.02192544925492257, 0.02228095254395157]` → `0.022109400015324354` | `[0.03723940171767026, 0.03695889899972826, 0.036988899228163064]` → `0.036988899228163064` | `0.001953125 / 0.001953125 / 0.001953125`; `0 / 0 / 0` |
| `[1,1536]` plain | `[0.028349951026029885, 0.028247400769032538, 0.028084401856176555]` → `0.028247400769032538` | `[0.02112494839821011, 0.021648401161655784, 0.021263951202854514]` → `0.021263951202854514` | `[0.027334451442584395, 0.028128898702561855, 0.0281904503935948]` → `0.028128898702561855` | `0.0009765625 / 0.0009765625 / 0.0009765625`; N/A |
| `[1,1536]` add | `[0.03724489943124354, 0.03703740076161921, 0.037555399467237294]` → `0.03724489943124354` | `[0.022344948956742883, 0.022114950115792453, 0.022939449991099536]` → `0.022344948956742883` | `[0.03730239986907691, 0.03716589999385178, 0.03713690093718469]` → `0.03716589999385178` | `0 / 0 / 0`; `0 / 0 / 0` |

按四组中位数计算，A→B 的同步 host operator-call 时间下降 `24.72%–48.18%`；四组中 B 的每次原始计时也都快于 A 与 A-after。该范围只描述本 focused operator-call 测量，不代表纯设备 kernel 时间、模型吞吐或端到端加速。

### 模型级验证状态

使用本分支公开 `run.sh`，TP1、FP16、context 2048、official attention 与已接入的 vendor SwiGLU。A/B 各两次固定 prompt 贪婪生成的完整 32 个 token ID 均匹配本模型的 `validation/baseline_token_ids.json`。实际默认编译为 mode3/backend=eager/CUDAGraphMode.NONE，enforce_eager=false；49 个 TecoopsRMSNorm 权重均为 SDAA FP16、D1536、epsilon1e-6，core 的唯一实际映射与导入 extension 的路径/哈希分别匹配上方 A/B 哈希。

warmup3 后模型请求墙钟三次原值：A `[2439.6632940042764, 2468.2493879809044, 2416.493065014947]`，中位数 `2439.6632940042764` ms；B `[2418.1028910097666, 2396.589677024167, 2412.7150650019757]`，中位数 `2412.7150650019757` ms。本次中位数差约 1.10%，不作为稳定模型加速结论。

B 稳态 `301.92812288494315` 秒、`110` 次请求全部匹配完整 greedy32 参考；观测的 allocator delta/slope 均为 0。两臂请求窗口 peak allocated/reserved 为 `13868604928 / 14969470976` bytes。独占逻辑设备 1 的测试窗口 teco-smi 采样为 `14532` MB，未获得 host PID 与容器 PGID 的直接映射，因此它只提供独占设备测试窗口的显存证据，不能等同于 Torch allocator 或精确 PGID 归属。自有服务进程组已停止，设备释放。

最终公开 focused 脚本 8/8 通过原阈值，源码和 shell/Python 语法、初始化 fail-closed 与 diff-check 通过。首次模型启动的 CPU-only 环境变量泄漏错误保留原始日志；修复仅作用于测试驱动的子环境，未改变 SDK、模型、oracle 或 compiler 配置。

独立复现、完整原值与限制记录在 [MiniCPM 模型 PR4](https://github.com/Tecorigin/tecovllm-modelzoo/pull/4) 的 `model_adaptations/MiniCPM5SCUdoudui/validation/rms_epilogue_simd_20261008.json`。当前官方 CI、py311、组委会精度、长上下文及所有 open PR 的组合 wheel 均未运行。
- 本段覆盖 E29 SIMD ancestor，不代表当前 PR37 kernel `93e8e80d08d401fffaa3060bf8e6e24bd989ab59fd8cb09c079228710f66d5b7`（InternVL DMA）已在 MiniCPM 验证。MiniCPM 本轮 DMA 候选为 NO-GO，未纳入本实验。
