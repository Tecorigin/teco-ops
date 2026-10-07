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
accuracy were not run. Other models' old PR37 evidence does not validate this
new epilogue head. InternVL's independent proof and reproducible minimal
source patch are published in the InternVL model PR validation folder.


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
