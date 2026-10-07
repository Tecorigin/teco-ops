# MultiScaleDeformableAttention 反向

`tecoops.ms_deform_attn_backward` 提供 SDAA 反向原语。Python 入口固定调用 list 实现 `tecoopsMsDeformAttnBackwardList`；原有 atomic C ABI `tecoopsMsDeformAttnBackward` 保留，签名与行为不变。`tecoops.ms_deform_attn` 将已有前向与该反向接入 PyTorch eager 一阶 autograd。

## 输入和返回值

```python
grad_value, grad_locations, grad_weights = tecoops.ms_deform_attn_backward(
    value, spatial_shapes, sampling_locations, attention_weights, grad_output
)
```

张量布局为 `value [N,S,H,D]`、`spatial_shapes [L,2]`（int64）、`sampling_locations [N,Q,H,L,P,2]`、`attention_weights [N,Q,H,L,P]` 和 `grad_output [N,Q,H*D]`。`H` 表示 head 数，`Q` 表示 query 数。三个返回梯度分别与 value、采样位置和注意力权重同形；`spatial_shapes` 无梯度。

输入与 `grad_output` 必须是同一 SDAA 设备上的连续张量，且 dtype 统一为 FP32 或 FP16；`spatial_shapes` 必须是该设备上的连续 int64 张量。各维度为正数，支持 `D <= 128`、`L <= 8`。调用方须保证每层空间形状的高宽均为正数，且所有高宽乘积之和等于 `S`。Python 绑定检查工作区计数可由 int32 索引；`N*S*H` 和 `4*N*Q*H*L*P` 均须不超过 `INT32_MAX`。该接口要求 SDAA 张量，不提供 CPU fallback。

反向遵循前向的双线性采样语义：zero padding、`align_corners=False`。采样位置梯度对应归一化坐标，x/y 分量分别乘以该 level 的宽度/高度；注意力权重梯度对传入的原始权重求导，算子不附加 softmax。

## List C API 与工作区

C 声明位于 [`teco/interface/include/tecoops.h`](../../teco/interface/include/tecoops.h)。新增 list 入口的完整签名为：

```c
tecoopsStatus_t tecoopsMsDeformAttnBackwardList(
    tecoopsHandle_t handle,
    const void *value,
    const int64_t *spatial_shapes,
    const void *sampling_locations,
    const void *attention_weights,
    const void *grad_output,
    float *grad_value,
    void *grad_value_fp16,
    void *grad_locations,
    void *grad_weights,
    int32_t *value_heads, int32_t *value_next, float *node_wx, float *node_wy,
    int batch, int value_len, int num_heads, int head_dim,
    int num_queries, int num_levels, int num_points,
    tecoopsDataType_t data_type, tecoopsAlgo_t algo);
```

四个 list workspace 的容量为：

- `value_heads`: `int32[N*S*H]`
- `value_next`: `int32[4*N*Q*H*L*P]`
- `node_wx`、`node_wy`: 各为 `float[4*N*Q*H*L*P]`

令 `heads=N*S*H`、`nodes=4*N*Q*H*L*P`，额外 list workspace 共需 `4*heads + 12*nodes` 字节。另需 `float grad_value[N,S,H,D]`；FP16 还需独立的半精度 `grad_value_fp16[N,S,H,D]`。FP32 时 `grad_value` 同时承载返回的 value 梯度，`grad_value_fp16` 可传 `nullptr`；FP16 时 value 梯度从单独的半精度输出读取。`grad_locations`、`grad_weights` 分别按输入形状和 dtype 分配。

List 调用在 handle 的 current stream 上按 Init → Produce → Reduce → HalfCast 排队；FP32 跳过最后的 cast。Init 会初始化链表头，调用方不需要预清零 workspace。Producer 只为有效采样 corner 复制 `D * sizeof(dtype)` 字节，并发布链表节点；Reduce 负责写入全部 FP32 value 梯度，包括没有采样节点的位置。多个 producer 通过原子插入建立链表，Reduce 的累加顺序不固定，因此 value 梯度可能存在非确定性。位置和权重梯度由各自唯一 owner 写入。

工作区及输出区域必须互不重叠，并保持有效直到 handle stream 完成；异步调用涉及的输入也须在该 stream 完成前保持有效。FP16 的 `grad_value_fp16` 必须与 FP32 `grad_value` workspace 分开。FP16 次正规输入按原始 half 位模式展开；value 梯度以 FP32 累加后转换为 half，转换保留次正规结果并采用 round-to-nearest-even。

## Autograd 用法

`tecoops.ms_deform_attn(value, spatial_shapes, sampling_locations, attention_weights)` 返回 `[N,Q,H*D]`。反向返回 value、采样位置和注意力权重的梯度，不返回 `spatial_shapes` 梯度。包装器使用 `once_differentiable`，支持 eager 一阶反向，不支持二阶梯度。当前没有 FakeTensor/meta 或 `torch.compile` 集成支持证据，不应视作这些模式已受支持。

在仓库根目录、隔离构建产物可从 `api/` 加载时，可用厂商 Python `/home/py312/bin/python` 运行以下示例；无需全局安装扩展：

```python
import sys
sys.path.insert(0, "api")
import torch_sdaa  # 注册 SDAA 后端
import torch
import tecoops

device, dtype = "sdaa", torch.float32
shapes = torch.tensor([[2, 3]], device=device, dtype=torch.int64)
value = torch.randn(2, 6, 2, 4, device=device, dtype=dtype)
locations = torch.rand(2, 5, 2, 1, 3, 2, device=device, dtype=dtype)
weights = torch.rand(2, 5, 2, 1, 3, device=device, dtype=dtype)
grad_output = torch.randn(2, 5, 8, device=device, dtype=dtype)

# 直接读取三个梯度
gv, gl, gw = tecoops.ms_deform_attn_backward(
    value, shapes, locations, weights, grad_output
)

# 在 eager 训练图中使用配对包装器
value.requires_grad_()
locations.requires_grad_()
weights.requires_grad_()
output = tecoops.ms_deform_attn(value, shapes, locations, weights)
(output * grad_output).sum().backward()
```

## 验证入口

[`python_api_test/test_ms_deform_attn_backward.py`](../../python_api_test/test_ms_deform_attn_backward.py) 包含 CPU 参考梯度、有限差分、raw API 与 autograd 对拍、输入拒绝检查、边界和 FP16 次正规/RNE 覆盖；`--real-shapes` 增加 encoder/decoder 尺寸覆盖。C++ 测例见 [`ms_deform_attn_backward.cpp`](../../test/zoo/teco/ms_deform_attn_backward/ms_deform_attn_backward.cpp)、其 CPU 参考 [`ms_deform_attn_backward.py`](../../test/zoo/teco/ms_deform_attn_backward/ms_deform_attn_backward.py) 和 [`case_0.prototxt`](../../test/zoo/teco/ms_deform_attn_backward/test_case/case_0.prototxt)。

```bash
# CPU oracle 与 wrapper 合约
/home/py312/bin/python python_api_test/test_ms_deform_attn_backward.py --cpu-only

# SDAA micro cases 与 real shapes（默认使用已安装的 wheel）
# build_ext --inplace 开发流需显式选择完整本地包：
# export PYTHONPATH="$PWD/api:${PYTHONPATH:-}"
# 测试不会把尚未构建的 api/ 目录插到已安装 wheel 前面；
# 包、_torch_ext 与唯一已加载 libteco_ops.so 必须来自同一包目录。
/home/py312/bin/python python_api_test/test_ms_deform_attn_backward.py --real-shapes

/home/py312/bin/python python_api_test/test_ms_deform_attn_backward_list.py
/home/py312/bin/python python_api_test/test_ms_deform_attn_backward_offsets.py

# C++ fixture（按项目说明配置 SDK/parser，并构建本地算子库与 test/build 后）
export PATH=/home/py312/bin:$PATH
export PYTHONPATH="$PWD/test/zoo/teco:$PWD/api:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$PWD/api/tecoops:/home/py312/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
cd test/build
./demo --gid=0 --perf_repeat=1 --warm_repeat=1 \
  --cases_dir=../zoo/teco/ms_deform_attn_backward/test_case
```

另有 [test_ms_deform_attn_backward_list.py](../../python_api_test/test_ms_deform_attn_backward_list.py) 的长链/空链验证，以及 [test_ms_deform_attn_backward_offsets.py](../../python_api_test/test_ms_deform_attn_backward_offsets.py) 的 36 组双输入 offset=1 位级验证。

## 随机官方模型结果

<!-- FINAL_MODEL_RESULT -->
固定 seed=1234 的官方 6 层 encoder / 6 层 decoder 随机权重 DeformableTransformer，FP32、N=2、S=22223、H=8、D=32、decoder Q=300；loss probes 固定 seed=20261004。预热一次后连续计时三次，输入、权重及 loss 相同，CPU oracle 与复制检查不计入计时。

| 完整 forward + backward | 三次原值（秒） | 中位数（秒） | 峰值 allocated（MiB） |
|---|---|---|---|
| SDAA grid 基线 | 22.829681091 / 22.595644849 / 21.796284871 | 22.595644849 | 9696.833984 |
| 本 PR forward + List backward + consumer/producer DMA | 16.696383636 / 16.695467289 / 16.708399124 | 16.696383636 | 3944.024414 |

中位训练耗时下降 26.108%，峰值 allocated 约下降 59.3%。该结果属于组合路径，不能单独归因 CAS 或某一项 DMA。307.558 秒内共 18 轮（216 次前向和 216 次反向原语调用），每轮均检查 3 项输出和全部 236 个参数/输入梯度；输出 atol/rtol=2e-4，梯度 atol=2e-5、rtol=1e-3。独立 smoke 中最大输出误差 2.324581e-6、最大梯度误差 1.800777e-5，重复前向位级一致，SGD 更新 230 个参数。真实 encoder/decoder FP32/FP16 原语对拍、C++ 实际 DIFF1 门限 5e-5、长链/空链、默认/非默认 stream 和 36 组 value+grad_output offset=1 的严格位级回归均通过。

上述测试使用随机权重与合成 loss；预训练检测 checkpoint 和官方任务精度未验证，生产模型默认 grid 不在本 PR 中修改。当前证据不支持 FP16 完整模型性能或任务准确率声明。

## List producer grad_output record (2026-10-07)

The List-only bilinear backward specialization stages one owner row of
grad_output in an aligned LDM T[128] record with an exact D*sizeof(T) blocking
copy, then reads those elements with the existing load_scalar half/float
conversion. The atomic specialization has no record or copy. Compile-time
selection preserves corner DMA, List ownership/consumer, FP32 arithmetic,
CAS behavior, output layout and ABI. This is one mechanism, not a new
algorithm or backend selection in forward.

Official main is de27305efed0a17ae926d21d5415d8b915614649; this delta applies to
PR40 head e92caeacbcc709ea934a98cf6bbadfc2afee1df9. The compiled paired baseline
is the frozen b565d402 source on model/deformable-detr commit46f6d26.
All18 runtime source files match current e92 Git bytes; b565→e92 changes only
README and three forward test-executor BSD headers. This establishes source
equivalence, not an independently rebuilt e92 wheel.
Candidate target source SHA256:
17e2b03a8e4168861ee0d4bea4b62b9d39f84d38421e158b9152babd7b9ed23e.
Fresh candidate core SHA256:
618195bf59c3bc5946d7865a4eddd6131eec93b788884b8b5a9d99e7656f9a56.
Paired baseline core is ef437793422d1c7b89e4f1680b8430f53c5d28b7b21f3bae908911457853217d;
both use byte-identical extension
4c44e07c3e02f9433416d2afe7c28d55b3b87c6bbac6f40f4b62549af240392a.
Vendor Python /home/py312/bin/python resolves to
/usr/local/python/bin/python3.12; SDK/runtime3.2.0, Torch2.12.0a0+0d62256 and
Torch-SDAA20260623.8.51+d942f23 are unchanged.

Original focused backward132 comparisons, offset216 exact gradient checks,
List48 cases including hot65536/empty/both streams, and owned forward tests
pass. FP64 finite-difference/quantized reference/mock-autograd checks, syntax
and diff checks pass. The initial forward bootstrap shadowed the local
package with installed tecoops; that failed log is retained, and rerunning
the same test with explicit candidate PYTHONPATH passed. No threshold changed.

Independent actual VOC geometry is N2/S494/H8/D32/L1/P4, encoderQ494 and
decoderQ300, from two padded608x810 images. L1 primitive timing below uses
visible device1, same inputs, warmup1 and three sequential wall-clock trials,
with synchronization. Units are seconds per backward call.

| Dtype /query role | Baseline three raw seconds | Candidate three raw seconds | Median baseline →candidate |
| --- | --- | --- | --- |
| FP32 /encoder494 | .010953173,.010984883,.010958273 | .010979623,.010992003,.010938033 | .010958273 →.010979623 |
| FP32 /decoder300 | .006720493,.006702043,.006698533 | .006664404,.006704563,.006696643 | .006702043 →.006696643 |
| FP16 /encoder494 | .019909751,.019745651,.019779981 | .018773673,.018396995,.018574864 | .019779981 →.018574864 |
| FP16 /decoder300 | .013969566,.013405497,.013691296 | .013351487,.012815219,.013284647 | .013691296 →.013284647 |

FP16 same-geometry primitive latency decreases6.09%/2.97% in these trials.
FP32 primitive ranges overlap; no stable FP32 gain is claimed. The actual
accepted VOC model is FP32, so FP16 primitive results are not its model gain.

Actual model forward/loss/backward checks pass with12 native forward and12
native backward calls,286 finite parameter gradients and loss15.6343355178833.
Its no-capture timing path removes CPU tensor capture before warmup and
three measured trials; finite-gradient checks run after elapsed time.
Model baseline [.606179200,.624600624,.621106923], median.621106923s;
candidate [.610509759,.611365727,.621194002], median.611365727s. Ranges overlap,
so the1.568% observed median difference is not claimed as stable model gain.

Candidate reference-checked steady300.711627s/380rounds passes4560 forward
and4560 backward calls, with no optimizer steps. Maximum output error
2.50339508e-6 and gradient error3.09944153e-6 stay within the original
forward atol/rtol2e-4 and gradient atol2e-5/rtol1e-3. Peak allocation/reservation
is888.853516/958MiB. Owned device processes are released.

The independent model proof is in the deformable model PR's
PyTorch/contrib/Detection/Deformable-DETR/validation-producer-go-20261007.json. Prior full20
training/accuracy receipts belong to their original baseline; training was
not repeated for this memory-load change. Current-head official CI/full
wheel and final committee accuracy were not run. No L4 performance
extrapolation, pretrained-weight copy or global package change is included.
