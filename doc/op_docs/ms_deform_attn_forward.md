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
模型仓库中的可微 `grid_sample` 兼容路径。当前支持 1–128 的 head_dim 和 1–8 个 level；非法参数的 C API 状态由 Python 绑定抛出异常。
输入必须位于同一 SDAA 设备并且连续，
`spatial_shapes` 必须是设备上的 `int64` 张量；调用方应在进入热路径前准备好这些布局。

```python
output = tecoops.ms_deform_attn_forward(
    value, spatial_shapes, sampling_locations, attention_weights
)
```

正确性覆盖见 `python_api_test/test_ms_deform_attn_forward.py`，该测试将 FP32/FP16
结果与 PyTorch 参考实现对拍。

## Value-record DMA validation (2026-10-04)

Each valid bilinear corner copies its contiguous value head vector into an aligned per-SPE T[128] SPM record with blocking SDK memcpy, then uses the existing scalar bit loader and arithmetic order. The record is consumed before the next overwrite. Geometry, dtype/ABI/layout/current-stream binding and output stores are unchanged; there is no backend switch, extra workspace or asynchronous overlap.

N2/S22223/M8/D32/L4/P4; encoder Lq22223, decoder Lq300; feature maps 100x167,50x84,25x42,13x21. Vendor /home/py312/bin/python, Torch2.12.0a0, Torch-SDAA20260623.8.51, driver/runtime3.2.0. Separate processes with the same seed20261004 quantized CPU inputs/gold and build flags; one warmup then three consecutive synchronized calls. CPU oracle/copy verification is outside timing. The baseline is a fresh clean PR40 7cc9c85 build without backward sources.

| dtype/shape | Baseline three seconds; median | Candidate three seconds; median | Median ratio |
|---|---|---|---|
| torch.float32/encoder | 3.636025397/3.632216866/3.642604181; 3.636025397 | 0.423396385/0.423247665/0.423144326; 0.423247665 | 8.5908x |
| torch.float32/decoder | 0.049008488/0.049189768/0.049515327; 0.049189768 | 0.005797135/0.005795375/0.005779665; 0.005795375 | 8.4878x |
| torch.float16/encoder | 3.884175320/3.886218835/3.884347579; 3.884347579 | 0.706576727/0.707082615/0.706673206; 0.706673206 | 5.4967x |
| torch.float16/decoder | 0.052652519/0.052580090/0.052576849; 0.052580090 | 0.009561125/0.009563445/0.009590055; 0.009563445 | 5.4980x |

Alternating real encoder/decoder FP32/FP16 and default/nondefault stream passed 300.388253s/932calls. Tight max_abs gates remain FP32 5e-5/FP16 2e-3. Public Python tests additionally cover D1/3/31/32/128, contiguous storage_offset1, unaligned half records and amplified signed half-subnormal values. C++ case0 passes actual DIFF1 max_error5e-5, observed2.57307e-8; the local executor now binds the declared threshold instead of inheriting the framework default1e-2.

Fixed seed1234 random-weight official 6encoder/6decoder transformer no_grad forward: grid 8.176868655/8.252117659/8.206105379s median8.206105379; candidate 3.733160698/3.732751320/3.733125958s median3.733125958. Three outputs pass unchanged atol/rtol2e-4 (max errors2.324581e-6/0/0), candidate repeated outputs bit-identical; peak allocated 3357.516->785.139MiB. This is an isolated inference probe, not pretrained detection, training throughput, or official accuracy; production training retains the differentiable grid path.
