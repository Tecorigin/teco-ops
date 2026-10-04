# MultiScaleDeformableAttention 反向

`tecoops.ms_deform_attn_backward` 为 MultiScaleDeformableAttention 提供 SDAA 反向原语；`tecoops.ms_deform_attn` 将已有前向与该反向接入 PyTorch 的一阶 autograd。现有 `tecoopsMsDeformAttnForward` C ABI 保持不变，原始前向入口仍不自行注册 autograd。

## 输入、输出与限制

原始 Python API 的签名为：

```python
grad_value, grad_sampling_locations, grad_attention_weights = (
    tecoops.ms_deform_attn_backward(
        value, spatial_shapes, sampling_locations, attention_weights, grad_output
    )
)
```

张量布局与前向相同：`value [N,S,M,D]`、`spatial_shapes [L,2]`（int64）、`sampling_locations [N,Lq,M,L,P,2]`、`attention_weights [N,Lq,M,L,P]`；`grad_output` 为 `[N,Lq,M*D]`。返回三项梯度分别与 `value`、`sampling_locations` 和 `attention_weights` 同形。`spatial_shapes` 不求梯度。

输入、`grad_output` 必须是同一 SDAA 设备上的连续张量，且统一为 FP32 或 FP16；`spatial_shapes` 必须是该设备上的连续 int64 张量。批次、空间长度、head 数、head_dim、query 数、level 数和 point 数须为正数；支持 `D <= 128`、`L <= 8`。调用方还须保证每个 `(H,W)` 为正数且所有 `H*W` 之和等于 `S`。API 不做 CPU fallback；传入 CPU 张量会被拒绝。

## C API 工作区

C 入口声明在 [`teco/interface/include/tecoops.h`](../../teco/interface/include/tecoops.h)：

```c
tecoopsStatus_t tecoopsMsDeformAttnBackward(
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
    int batch, int value_len, int num_heads, int head_dim,
    int num_queries, int num_levels, int num_points,
    tecoopsDataType_t data_type, tecoopsAlgo_t algo);
```

`grad_value` 始终指向调用方分配的 FP32 设备缓冲区，大小至少为 `N*S*M*D` 个 float；API 会在同一 handle stream 上先将它清零，再累加 `dvalue`，因此调用前无需预清零。FP32 时该缓冲区就是返回的 value 梯度，`grad_value_fp16` 可传 `nullptr`。FP16 时必须额外提供半精度 `grad_value_fp16` 输出缓冲区；它必须与 FP32 workspace 不重叠。API 在同一 stream 上完成 FP32 累加后，再以 round-to-nearest-even 转为该输出。位置和权重梯度写入 `grad_locations`、`grad_weights`，其元素类型与输入相同。

反向按 batch/query/head 为位置与权重梯度分配唯一写入者；value 梯度由多个采样点通过 atomic 累加，因此其结果不保证确定性。坐标导数使用与前向一致的双线性采样语义：zero padding、`align_corners=False`；对归一化采样坐标的导数分别乘以宽度 `W` 和高度 `H`。权重导数是对传入的原始 `attention_weights` 求导，算子内部不附加 softmax。

## 一阶 autograd 用法

`tecoops.ms_deform_attn(value, spatial_shapes, sampling_locations, attention_weights)` 返回 `[N,Lq,M*D]`，可用于普通一阶反向传播。其 backward 返回 value、采样位置和注意力权重的梯度；`spatial_shapes` 无梯度。该包装器标记为 `once_differentiable`，不支持二阶梯度；value 的 atomic 累加也不保证确定性。

在仓库根目录、已有隔离构建产物可从 `api/` 加载时，用厂商 Python `/home/py312/bin/python` 运行下例；示例不安装或修改全局 Python 包：

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

# 需要直接访问三个梯度时使用 raw API。
gv, gl, gw = tecoops.ms_deform_attn_backward(
    value, shapes, locations, weights, grad_output
)

# 训练图使用配对包装器；backward 调用同一 raw backward API。
value.requires_grad_()
locations.requires_grad_()
weights.requires_grad_()
output = tecoops.ms_deform_attn(value, shapes, locations, weights)
(output * grad_output).sum().backward()
```

## 验证范围

[`python_api_test/test_ms_deform_attn_backward.py`](../../python_api_test/test_ms_deform_attn_backward.py) 提供独立 CPU 梯度 oracle、有限差分、raw API 与 autograd 包装器对拍，以及边界、重叠采样、subnormal、FP16 RNE tie 和非默认 stream 覆盖。已通过的算子级设备对拍包括这些 micro cases，以及 `N=2, S=22223, M=8, D=32, L=4, P=4` 的 encoder `Lq=22223` 和 decoder `Lq=300`，FP32/FP16 均与 CPU 参考对拍。模型级随机权重训练验证及性能取舍见下文；预训练检测与官方任务精度尚未验证。


This first feature commit uses FP32 atomic value-gradient accumulation. Subsequent commits independently optimize value-gradient ownership and record reads. The wrapper is eager first-order only, with no FakeTensor/meta or torch.compile registration.
