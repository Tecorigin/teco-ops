# MultiScaleDeformableAttention 前向

`tecoops.ms_deform_attn_forward` 是 Deformable-DETR 推理路径使用的 SDAA UAL 算子。它接收

```text
value               [N, S, M, D]
spatial_shapes      [L, 2] int64
sampling_locations  [N, Lq, M, L, P, 2]
attention_weights   [N, Lq, M, L, P]
```

并返回 `[N, Lq, M*D]`。坐标语义与 PyTorch `grid_sample` 的
`mode="bilinear"`, `padding_mode="zeros"`, `align_corners=False` 相同。FP32 与 FP16
均有独立的 UAL kernel 分支。

FP16 输入中的 subnormal 通过原始位模式显式展开为 FP32：符号 × mantissa × 2^-24；
普通值保留原转换。这样避免极小采样坐标在乘图像尺寸前被 flush-to-zero，计算与累加仍使用 FP32。
回归包含宽度 167 的正、负极小坐标、零与最小 normal 值，误差门限不变。

该接口是前向推理原语，不注册 autograd 反向实现。Deformable-DETR 训练适配继续使用
模型仓库中的可微 `grid_sample` 兼容路径。输入必须位于同一 SDAA 设备并且连续，
`spatial_shapes` 必须是设备上的 `int64` 张量；调用方应在进入热路径前准备好这些布局。

```python
output = tecoops.ms_deform_attn_forward(
    value, spatial_shapes, sampling_locations, attention_weights
)
```

正确性覆盖见 `python_api_test/test_ms_deform_attn_forward.py`，该测试将 FP32/FP16
结果与 PyTorch 参考实现对拍。
