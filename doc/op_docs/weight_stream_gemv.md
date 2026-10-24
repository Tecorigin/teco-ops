# tecoopsWeightStreamGemv 设计文档

## 计算原理

面向大模型 **decode 阶段 M=1 线性层**的权重流式 GEMV：一次只处理一行激活
`x[k]`，输出一整列 `out[n]`，计算的核心成本是**把整个 `[n, k]` 权重矩阵从 HBM
流式读进 SPE**，因此 kernel 以「整行连续 slab 的 DMA」组织，而不是按 GEMM 的
tile 组织。

**计算公式：**

```
out[i] = fp16( sum_j fp32( x[j] * weight[i][j] ) )      i ∈ [0, n), j ∈ [0, k)
```

即：fp16 输入、fp32 累加、结果只做一次 fp16 舍入。与厂商 GEMM 相比，操作数精度
与累加精度相同，但**归约顺序不同**（本 kernel 用 4 条交错 fp32 FMA 链 + 一次
16 lane 横向归约），因此输出的逐位结果不保证与厂商 GEMM 一致；两者对同一个
fp32 参考解的偏差都在半个 fp16 ulp 量级。模型侧的等价性由 greedy 32 token 的
`token_ids` 逐位比对（parity gate）守门，而不是由本算子对拍厂商 GEMM 守门。

**参数解释：**

- `x`：激活行，形状 `[k]`
- `weight`：权重矩阵，形状 `[n, k]`，行主序（`weight[i][j]` 位于 `i*k + j`）
- `out`：输出行，形状 `[n]`
- `n`：输出行数（词表 / 隐层输出维）
- `k`：归约长度（输入维）

## 功能实现

### 接口设计

C 接口（`teco/interface/include/tecoops.h`）：

```c++
tecoopsStatus_t tecoopsWeightStreamGemv(
    tecoopsHandle_t handle,
    const void *x,      // [k]     fp16, 设备端
    const void *w,      // [n, k]  fp16, 设备端, 行主序连续
    void *y,            // [n]     fp16, 设备端
    int n,
    int k);
```

Python 接口（`api/torch_ext.cpp` 绑定、`api/tecoops/__init__.py` 导出）：

| 入口 | 语义 | 返回 |
|---|---|---|
| `tecoops.weight_stream_gemv(x, weight, out)` | 算子本体，out= 形式，直接把结果写进 `out` | `None` |
| `tecoops.linear_stream(x, weight, bias=None)` | 模型侧 drop-in 替身（等价 `torch.nn.functional.linear`），按形状路由 | `[m, n]` 张量 |

两个入口都**绑定调用方当前的 SDAA stream**（`tecoopsSetStream`），与
`rms_norm` 等既有算子一致，不做隐式同步。

`linear_stream` 的路由规则（判据写在 `api/torch_ext.cpp` 的
`linear_stream_torch` 内，**不在** Python 层）：

| 条件 | 路由 |
|---|---|
| `x.dim() == 2 && x.size(0) == 1 && x.size(1) == k && x.dtype == half && weight.dtype == half && x.is_contiguous() && weight.is_contiguous() && n % 64 == 0 && k % 16 == 0` | 走 GEMV kernel（快路径） |
| 其余全部（prefill 的 M>1、profile 期形状、`n % 64 != 0`、`k % 16 != 0`、非 half、非连续） | 走厂商 GEMM（`torch::linear`） |

选择放在 C++ 的**单个不透明算子**里而不是 Python 的 `if`：Python 层的分支会被
Dynamo trace 进计算图，profile 期（`M = max_num_batched_tokens`）跟踪到哪一支，
decode 图里就会被固化哪一支；放在 C++ 内则整个调用是一个图节点，由运行时真实
形状决定走哪一边。

快路径上 `bias` 由 C++ 以 **fp16 加法**作用在 GEMV 结果行上（`res + bias`），
不是 GEMM 的融合 epilogue，故带 bias 的结果不保证与 `torch.linear` 逐位一致，
但按其自身口径可逐位复现（测试中按位相等断言）。

**带 bias 路径的数值边界（如实声明）**：`linear_stream+bias` 的 `round_exact`
（逐元素用 fp32 参考舍入到 fp16 后逐位一致的比例）实测为 **0.668–0.754**，明显低于
无 bias 快路径的 **0.988+**。原因是两边的舍入顺序不同：本 kernel 是「GEMV 结果先舍入到
fp16，再以 **fp16 加法** 加上 bias」，而 Python 参考口径是「bias 在 fp32/fp64 中与
归约结果相加后**只做一次**最终舍入」。相加把 `round_exact` 从「仅归约序差」变成
「多一次 fp16 舍入 + 可能的相对误差放大」，因此 **bias 路径不承诺与参考逐元素一致**；
但它仍按自身机制逐位可复现（测试断言 `linear_stream+bias` 与 `GEMV 结果 + fp16 bias` 逐位
相同），且对 fp32/fp64 参考的 `margin`（实测误差 / 允许门限）仍保持 **4.7–8.0×**，
精度余量充分。依赖逐位一致的调用方（若有）应使用无 bias 的 `weight_stream_gemv`
或自行处理舍入语义。

### 参数信息

| 参数 | 输入/输出 | 主机端/设备端 | 说明 |
|---|---|---|---|
| handle | 输入 | 主机端 | Teco-Ops 句柄，管理设备上下文与 stream |
| x | 输入 | 设备端 | 激活行指针，形状 `[k]`，fp16 |
| w | 输入 | 设备端 | 权重矩阵指针，形状 `[n, k]`，fp16，行主序连续 |
| y | 输出 | 设备端 | 输出行指针，形状 `[n]`，fp16 |
| n | 输入 | 主机端 | 输出行数，须满足 `n % 64 == 0` |
| k | 输入 | 主机端 | 归约长度，须满足 `k % 16 == 0` |

返回码：`handle == nullptr` → `TECOOPS_STATUS_NOT_INITIALIZED`；空指针、`n <= 0`、
`k <= 0`、或违反下面的契约 → `TECOOPS_STATUS_BAD_PARAM`。

### 类型限制与形状契约

当前计算分支，主要完成以下功能实现，其余情况暂不支持。

| 参数 | 数据类型 | 维度信息 | 存储格式 |
|---|---|---|---|
| x | float16 | `[k]` | 连续一维 |
| weight | float16 | `[n, k]` | 行主序连续（row-major） |
| out | float16 | `[n]` | 连续一维 |

**契约（唯一正确性门限）：**

```
n % 64 == 0  &&  k % 16 == 0
```

契约由实现、`linear_stream` 快路径判据、zoo 的 `paramCheck`、以及 python 测试
**四处的同一判据**共同表达；违反时接口 **fail-closed**（返回
`TECOOPS_STATUS_BAD_PARAM`，不写输出），不会退化成「静默算错」。Python 绑定
不检查返回码、也不抛异常，因此调用方观察到的是「输出未被写入」，测试正是以
「输出仍是调用前填的 NaN 哨兵」来断言这一点的。

**`n % 64` 的由来**（不是拍出来的数字）：kernel 用 `SplitMode::R1C32_CR` 把
`n` 行按 32 个 SPE 连续切分，每个 SPE 得到 `my_rows = n / 32` 行，并在结尾用
**一次** DMA 把 `my_rows * 2 = n / 16` 字节的输出块刷回 HBM。设备端
`check_memcpy_async()` 对异步搬运的**地址和长度都按 4 B 校验**（`mask 3`），
而每个 SPE 的目标偏移是 `n / 16` 字节的整数倍，因此这一条件与
`n / 16 % 4 == 0` 等价，即 **`n % 64 == 0`**。`k % 16 == 0` 则来自内层 16 lane
（32 B）fp16 向量宽度。

**`n % 256` 只是性能对齐形状，不是正确性约束。** 模型侧真实 decode shape 的
`n`（词表 / 输出维）本来就是 256 的整数倍，早先的 docstring 与 zoo `paramCheck`
按更严的 `n % 256 == 0` 描述（对齐模型侧标定形状的说法）；实现与 `linear_stream`
快路径判据从未要求 256——它们一直只用 `n % 64`。
本 PR 把 docstring、zoo `paramCheck`、测试标签统一收敛到 `% 64`，并补
`n = 128`、`n = 64` 两个边界用例（它们必须走快路径而非回退），正是为了消除
「实现接受（`n % 64 == 0`）但门禁拒绝（`n % 256 != 0`）」这段 64/128/192 的
测试盲区。官方固定 case 仍是 `n = 256` / `n = 512`，不受影响。

**形状 → 实现 对照表**（`linear_stream` 单行快速自检；`weight_stream_gemv` 本体
不判 M，直接由接口契约判 `n/k`）：

| 条件（M=1、fp16、连续为前提） | 实现 | 说明 |
| --- | --- | --- |
| `n % 64 == 0 && k % 16 == 0` | **GEMV 快路径**（`teco_slave_weight_stream_gemv`） | 契约与快路径判据同名（见上） |
| 其余（`M>1` prefill、`n % 64 != 0`、`k % 16 != 0`、非 fp16、非连续） | **厂商 GEMM 回退**（`torch::linear`） | 数值走 GEMM 自身路径，`round_exact` 通常为 1.0 |
| `n % 256 == 0`（模型侧真实形状） | 仍走快路径 | 只是模型侧 decode `n` 的**性能对齐**表述，不是门限；`n=64/128/192` 等非 256 倍数同样被接受 |

模型侧注意：真实 decode 分片把 `n` 标到 256 的整数倍（如 InternVL3_5-8B 逐 rank
`n = 75968 = 64 * 1187`），因此以 `n % 256` 描述模型侧标定成立；但**部署契约**以
`n % 64 == 0 && k % 16 == 0` 为准，任何只满足 `%256` 之外的整倍数 `n`（只要满足
`%64`）也走快路径，不应被上层以 `%256` 二次过滤。

## 分支派发

| 算法取值 | 计算分支 | 含义说明 |
|---|---|---|
| `TECOOPS_ALGO_0`（默认，唯一分支） | `teco_slave_weight_stream_gemv` | 32 SPE 按行切分、权重连续 slab 双缓冲、fp32 累加的基础实现 |
| 其他 | 未实现 | `findWeightStreamGemvBranch()` 恒返回 `0` |

## 性能优化

实现取向：M=1 时整层成本几乎全在「把 `[n, k]` 权重从 HBM 读一遍」，因此不做
GEMM 式分块，而是让每个 SPE 连续、成块地搬自己那 `my_rows` 行。

1. **激活行只取一次并只加宽一次**：`x[k]` 一次性 DMA 进 SPM，并就地转成 fp32
   （`xspm32`）；此后每行权重的内层循环直接按 fp32 读，省掉每条权重行一次
   `vfcvths`。fp16 → fp32 是精确转换，数值路径不变。
2. **权重按整行连续 slab 双缓冲**：每个 DMA 是 `slab` 整行（`slab * k * 2` 字节）
   的连续大块，而不是小粒度 strided 传输；两个 SPM 缓冲轮转，当前 slab 做归约
   时下一 slab 已在飞。
3. **slab 上限取 `my_rows` 的一半**：官方两例的形状会让 `slab == my_rows`，
   双缓冲循环只跑一轮、预取分支永不触发，整块 DMA 完全暴露在计算之前；把 slab
   压到 `(my_rows + 1) / 2` 后循环至少两轮，第二半权重在归约第一半时被预取。
   每行的内层链与归约顺序 **完全不变**，因此结果的逐位值不变，只有 DMA/计算
   时序改变。
4. **每行 4 条交错 fp32 FMA 链**：把单条串行依赖（`k/16` 次 `vfma`）压到
   `k/64`，再由一次 `simd_redsum` 横向归约，降低每行延迟；尾部按 16/标量处理。
5. **输出只刷一次**：每个 SPE 的 `my_rows` 个结果先写进本核 SPM 输出缓冲，
   结尾一次 DMA 刷出（这也是 `n % 64` 契约的来源），避免逐元素小 DMA。
6. **SPM 预算 192 KB**（`WSG_SPM_BUDGET`）：交付的 `flash_attention` 已占约
   450 KB，本 kernel 主动压低占用；SPM 申请失败时打印 `FATAL` 并读 `rcsr 5`
   触发 SPM 故障异常让进程显式失败，而不是让「未写过」的缓冲区冒充结果
   （`rt_spm_malloc` 失败即从所在函数返回，调用点的空指针判断永远不成立，
   故用 helper 把宏展开后再判空）。

### 性能自测

- 测例（官方固定形态）：
  - `zoo/teco/weight_stream_gemv/test_case/case_0.prototxt`：`n = 256, k = 256`，
    uniform `[-0.5, 0.5]`，`error_func: DIFF1`，`threshold 1e-2`
  - `zoo/teco/weight_stream_gemv/test_case/case_1.prototxt`：`n = 512, k = 512`，
    uniform `[-0.25, 0.25]`，同上判据
- 固定命令形态（`3` 次预热 + `50` 次重复，报告**平均值**）：

  ```bash
  ./build/demo --gid=0 --perf_repeat=50 --warm_repeat=3 --gtest_repeat=1 \
      --case_path=zoo/teco/weight_stream_gemv/test_case/case_0.prototxt \
    | grep "Average Hardware Time"
  ```

- 本 PR 当次构建的逐例 `Average Hardware Time` 与 `max_error` 原值记录在验证日志
  （`logs/zoo-weight_stream_gemv-case_0.log`、`logs/zoo-weight_stream_gemv-case_1.log`），
  并原样汇总到 PR 的验证章节。本表不另填数字：绝对时间对构建（设备段布局）与
  设备状态敏感，跨构建比较必须用同一窗口、同一构建根只换库的 A/B。

## 文件结构

```
teco/
├── interface/
│   ├── include/tecoops.h                        # userAPI 声明
│   └── ops/weight_stream_gemv.cpp               # 接口实现（契约校验，fail-closed）
├── ual/
│   ├── args/weight_stream_gemv_args.h           # 参数结构体
│   ├── ops/gemv/
│   │   ├── weight_stream_gemv.hpp               # Op 类定义（唯一分支）
│   │   ├── find_weight_stream_gemv.cpp          # 分支选择（恒 0）
│   │   └── find_weight_stream_gemv.h
│   └── kernel/weight_stream_gemv/
│       ├── weight_stream_gemv.h                 # kernel 声明
│       └── weight_stream_gemv.scpp              # kernel 实现
api/
├── torch_ext.cpp                                # weight_stream_gemv / linear_stream 绑定
└── tecoops/__init__.py                          # 导出
test/zoo/teco/weight_stream_gemv/
├── weight_stream_gemv.cpp                       # 参考实现 + paramCheck（n % 64 / k % 16）
├── weight_stream_gemv.h
└── test_case/{case_0,case_1}.prototxt
python_api_test/test_weight_stream_gemv.py       # Python-API 一致性测试
doc/op_docs/weight_stream_gemv.md                # 本文档
```

### Plugin 层

**未提供**：仓库 `teco/plugin/` 下现有 `pluginCausalConv1d` / `pluginFlashAttention` /
`pluginFlattenRays` / `pluginReduceVariance` / `pluginReshapeAndCache` / `pluginRmsNorm`
六个插件，没有 `pluginWeightStreamGemv*`，推理（Teco-Inference）侧对本算子
**零覆盖**；本算子当前只在 C / Python 两层交付，模型侧经
`tecoops.linear_stream` 接入。若推理侧需要，需另行补插件与 `test_plugin_*`。

## 使用示例

### PyTorch 扩展

```python
import torch
import tecoops

n, k = 512, 2048
x = torch.randn(k, dtype=torch.half, device='sdaa')
weight = torch.randn(n, k, dtype=torch.half, device='sdaa')

# 1) 算子本体：out= 语义，返回 None
out = torch.empty(n, dtype=torch.half, device='sdaa')
tecoops.weight_stream_gemv(x, weight, out)

# 2) 模型侧入口：M=1 走 GEMV，其它形状自动回退厂商 GEMM
row = x.unsqueeze(0)                 # [1, k]，M=1 decode 行
bias = torch.randn(n, dtype=torch.half, device='sdaa')
y = tecoops.linear_stream(row, weight, bias)     # [1, n]
```

### C++ 接口

```c++
tecoopsHandle_t handle;
tecoopsCreate(&handle);

tecoopsWeightStreamGemv(handle, x_dptr, w_dptr, y_dptr, n, k);

tecoopsDestroy(handle);
```

## 真实 decode shape 覆盖（zoo）

模型侧真实 M=1 decode shape 由 live launch census 实测（Gemma-4-12B、TP=2 单 rank、
2026-10-11；原始数据 `op_learning/gemm/gemma-gemv-fallback-census-20261011/raw/ls-shapes-*.tsv`）：

| N | K | 每代调用次数 | zoo 用例 |
| --- | --- | --- | --- |
| 3840 | 2048 | 7440 | `case_6` |
| 3840 | 4096 | 1488 | `case_7` |
| 4096 | 3840 | 7440 | `case_8` |
| 5120 | 3840 | 1488 | `case_9` |
| 3840 | 7680 | 8928 | 未纳入：权重 59 MB，官方 `perf_repeat=50` 下单例成本过高 |
| 15360 | 3840 | 4464 | 未纳入：权重 118 MB |
| 131072 | 3840 | 96 | 未纳入：权重 1.0 GB（embedding 行），不属 zoo 可测范围 |

连同 `case_2..case_5` 的契约边界（`n = 64/128/192`、`k = 48/2032`），zoo 现在覆盖
`n % 64 == 0`、`k % 16 == 0` 的主要取值组合，以及 `xspm32` 的 `k % 32 == 16` 分支。
上述为历史 zoo 覆盖；四模型独立 census 与全部真实 shape 的 Python API 回归已于 2026-10-24 补齐，见本文末节。
## 设备实测（c50，官方固定形态）

测量环境：SCU SDAA 单卡（TECO_AICARD_01 / SDAADriver 3.2.0 / SDAARuntime 3.2.0），
`./build/demo --gid=0 --perf_repeat=50 --warm_repeat=3 --gtest_repeat=1`，每例 3 次取中位。
源码为本 PR 分支（含下面的对齐修复），构建产物 `libteco_ops.so` 为同一 build root。

| case | (n, k) | AHT (ms) | 3 rep 原值 | max_error | 判定 |
| --- | --- | --- | --- | --- | --- |
| case_0 | (256, 256) | 0.008731 | 0.008790 / 0.008705 / 0.008731 | 0 | PASSED |
| case_1 | (512, 512) | 0.011236 | 0.011245 / 0.011236 / 0.011111 | 1.55e-07 | PASSED |
| case_2 | (64, 256) | 0.006041 | 0.006066 / 0.006041 / 0.006012 | 0 | PASSED |
| case_3 | (192, 48) | 0.007525 | 0.007416 / 0.007550 / 0.007525 | 0 | PASSED |
| case_4 | (1024, 2048) | 0.035139 | 0.034935 / 0.035168 / 0.035139 | 3.25e-07 | PASSED |
| case_5 | (128, 2032) | 0.010024 | 0.009964 / 0.010024 / 0.010025 | 0 | PASSED |
| case_6 | (3840, 2048) | 0.082577 | 0.082561 / 0.082577 / 0.082700 | 1.62e-06 | PASSED |
| case_7 | (3840, 4096) | 0.146140 | 0.146140 / 0.146130 / 0.146360 | 1.33e-06 | PASSED |
| case_8 | (4096, 3840) | 0.145540 | 0.145540 / 0.145510 / 0.145660 | 6.64e-07 | PASSED |
| case_9 | (5120, 3840) | 0.204740 | 0.204180 / 0.205180 / 0.204740 | 8.40e-07 | PASSED |

`case_3` / `case_5` 首次实测直接报 `SDAA_ERROR_MISALIGNED_ADDRESS`（warm-up 阶段），
根因与修复见下节；修复后两例 `max_error = 0`。

### 修复：`k % 32 == 16` 的 64 B 对齐缺陷

`weight_stream_gemv.scpp` 的 SPM 布局把 fp16 激活行（`K * sizeof(half)` 字节）与 fp32 激活行
（`K * sizeof(float)` 字节）紧邻摆放。只有 `K % 32 == 0` 时前者的字节数才是 64 的倍数；
`K % 32 == 16`（如 `k = 48 / 2032`）会让 fp32 缓冲区落在 32 B 对齐地址，而内层向量循环对它用
64 B 的 `simd_load`/`simd_store` ⇒ 设备侧 `SDAA_ERROR_MISALIGNED_ADDRESS`（把进程打挂）。
修复是在两次 carve 之间把游标重新对齐到 64 B（入口期指针运算，不引入热路径分支）；
提交 `2e48f83`。这也把此前 review 里“宿主只校验 `k % 16 == 0`、而 `xspm32` 可能非 64 B 对齐”
的建议项闭环为**实测缺陷 + 修复**。

## 四模型真实 decode shape 回归（2026-10-24）

本轮对 Gemma-4-12B-it、MiniCPM5-1B、Hy-MT2-1.8B、InternVL3_5-8B 分别运行
诊断 M=1 census；[合并输入](../validation/20261024/weight-stream-gemv-census.json)
保留每条观察的模型、层、原始 summary 来源及各模型构建元数据。22 条观察对应 22 个
唯一 `(n, k)`（含四模型完整 LMHead），其中 3 个与默认用例重合；默认 12 例增加 19 例后，
共 **31 cases、295 PASS、0 failure，进程退出码 0**。
[完整数值与路由结果](../validation/20261024/weight-stream-gemv-conformance.json)
记录 CPU fp32/fp64 参考、设备 GEMV、`linear_stream` 路由与 bias、非法形状拒绝，
以及信息性的 GEMV/GEMM 舍入差异。容差沿用原测试；输入为按真实 shape 生成的确定性随机
fp16 张量，未读取模型权重。本轮只主张正确性与覆盖，不主张性能提升或模型 token parity。

验证使用 `/home/py312/bin/python`（解析为 `/usr/local/python/bin/python3.12`）及合并团队
算子包，构建测试源码为 `db1597e1265a5fc3e24207b793c9d1dd94327a24`；**未独立重建
standalone PR44**。已核对 PR44 测试提交 `c3e7c6e` 与该构建源码、团队提交 `4b46b97`
的 `weight_stream_gemv` kernel、interface、args、ops 七个文件，均无差异；此源码一致性
不等于其它算子或设备镜像布局一致。

- core SHA-256：`531e0caf206a3c580e03f87965d067e44c0d88f8740a06a3d20abd88377574dc`
- extension SHA-256：`82aa5455938789bc0009f85a947edfa271918a9d0b115866ad7f70bb57fbbf41`

安装同一合并包并设置其 `PYTHONPATH` / `LD_LIBRARY_PATH` 后，在仓库根目录复现：

```bash
flock -x /root/gpu-slot/fix-20261024.lock /home/py312/bin/python \
  python_api_test/test_weight_stream_gemv.py \
  --census doc/validation/20261024/weight-stream-gemv-census.json \
  --output-json /tmp/weight-stream-gemv-conformance.json
```

原实测命令依次传入四个模型 summary（PR3→PR6）；合并文件按同序保留各行，生成的
31 个 case、随机种子与原测试一致。无 `--census` 时仍运行原默认用例。

### 证据再生与适用范围

上面的命令使用仓库内冻结的 census 作为输入，再生的是 **conformance 结果**；
`--output-json` 应写到新文件，保留历史记录。JSON 中绝对路径、数值误差与
库 SHA 随运行环境变化，不要求新报告与历史文件逐字节相同；以退出码 0、
`passed: true`、空 `failures` 和相同 shape 覆盖为验收依据。

`weight-stream-gemv-census.json` 是模型诊断进程观测的 shape 快照，GEMV 测试
本身不会重新运行模型或再生这份观测。新盘点应分别在四个模型自己的 checkout
中，按其 README 的 `SCUD_DECODE_CENSUS=<空目录>` 诊断流程运行原 `run.sh`
和固定 32-token 请求，保存 `decode-shapes-*.jsonl`，记录模型 commit、TP、
dtype、软件版本及库哈希，汇总为含 `model`、`rows` 的 JSON；测试支持按
Gemma、MiniCPM、Hy、InternVL 顺序重复传 `--census <summary.json>`。
历史各模型来源与构建信息保留在冻结 census 的 `models` 和每行 `source_file`
字段中。诊断模式的耗时不作为性能证据。

历史报告仅证明已记录环境：换卡、换驱动/runtime/SDK/PyTorch/Torch-SDAA、
换 kernel/extension 或构建选项后，必须重建并重跑 conformance；模型版本、
TP、dtype、分片方式、模型调用点或请求形态改变后，还须重新采集该模型的
census。旧 JSON 继续作为历史证据，不能自动证明新环境或新模型已验证。

### UAL 目录约定

`teco/ual/ops/gemv/` 是 GEMV 算子族的共享派发目录，本算子的
`find_weight_stream_gemv.{cpp,h}` 和 `weight_stream_gemv.hpp` 在此登记，
目录树见上文。公开算子名、interface、args、kernel 与 zoo 仍使用
`weight_stream_gemv`；不要另建 `ops/weight_stream_gemv/` 重复注册同一实现。
