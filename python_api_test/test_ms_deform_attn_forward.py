#!/usr/bin/env python3
"""Accuracy test for the inference-only MSDeformAttn forward primitive."""

from contextlib import nullcontext

import torch

import _bootstrap
import tecoops


def reference(value, spatial_shapes, sampling_locations, attention_weights):
    n, _, heads, dim = value.shape
    _, queries, _, levels, points, _ = sampling_locations.shape
    outputs = []
    start = 0
    for level in range(levels):
        height, width = [int(x) for x in spatial_shapes[level].tolist()]
        value_level = (
            value[:, start : start + height * width]
            .reshape(n, height * width, heads, dim)
            .permute(0, 2, 3, 1)
            .reshape(n * heads, dim, height, width)
        )
        grid = (
            (2 * sampling_locations[:, :, :, level] - 1)
            .permute(0, 2, 1, 3, 4)
            .reshape(n * heads, queries, points, 2)
        )
        outputs.append(
            torch.nn.functional.grid_sample(
                value_level,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        )
        start += height * width

    stacked = torch.stack(outputs, dim=-2).reshape(
        n * heads, dim, queries, levels * points
    )
    weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(
        n * heads, 1, queries, levels * points
    )
    return (stacked * weights).sum(-1).reshape(n, heads, dim, queries).permute(
        0, 3, 1, 2
    ).reshape(n, queries, heads * dim)


def check(
    dtype,
    batch=1,
    queries=5,
    heads=2,
    dim=3,
    use_non_default_stream=False,
):
    device = torch.device("sdaa")
    levels, points = 2, 2
    value_len = 16
    shapes_cpu = torch.tensor([[3, 4], [2, 2]], dtype=torch.int64)
    value_cpu = (
        torch.arange(batch * value_len * heads * dim, dtype=torch.float32)
        .reshape(batch, value_len, heads, dim)
        .remainder(23)
        .mul(0.0625)
        .sub(0.5)
        .to(dtype)
    )
    locations_cpu = (
        torch.arange(
            batch * queries * heads * levels * points * 2, dtype=torch.float32
        )
        .reshape(batch, queries, heads, levels, points, 2)
        .remainder(9)
        .add(1)
        .mul(0.1)
        .to(dtype)
    )
    weights_cpu = (
        torch.arange(batch * queries * heads * levels * points, dtype=torch.float32)
        .reshape(batch, queries, heads, levels, points)
        .remainder(7)
        .add(1)
        .div(8)
        .to(dtype)
    )

    if use_non_default_stream:
        if not hasattr(torch.sdaa, "Stream") or not hasattr(torch.sdaa, "stream"):
            raise RuntimeError("torch.sdaa Stream API is required for stream coverage")
        stream = torch.sdaa.Stream()
        stream_context = torch.sdaa.stream(stream)
    else:
        stream = None
        stream_context = nullcontext()

    with stream_context:
        shapes = shapes_cpu.to(device=device)
        value = value_cpu.to(device=device)
        # Queue device-side producers before the extension call. In the stream
        # case, the call must observe these writes on the same non-default stream.
        value = value.mul(1.25).add(0.25).contiguous()
        locations = locations_cpu.to(device=device)
        locations = locations.mul(0.75).add(0.125).contiguous()
        weights = weights_cpu.to(device=device)
        weights = weights.mul(0.5).add(0.25).contiguous()
        output = tecoops.ms_deform_attn_forward(value, shapes, locations, weights)
        if stream is not None:
            stream.synchronize()

    # Keep the oracle independent of the SDAA kernel and use FP32 CPU
    # grid_sample for both supported input dtypes.
    expected = reference(
        value.cpu().float(),
        shapes.cpu(),
        locations.cpu().float(),
        weights.cpu().float(),
    )
    error = (output.cpu().float() - expected).abs().max().item()
    limit = 5e-5 if dtype == torch.float32 else 2e-3
    print(
        f"dtype={dtype} shape=(N={batch},Lq={queries},M={heads},D={dim}) "
        f"non_default_stream={use_non_default_stream} max_error={error:.6e} "
        f"{'PASSED' if error < limit else 'FAILED'}"
    )
    return error < limit


if __name__ == "__main__":
    results = [
        check(torch.float32),
        check(torch.float16),
        check(
            torch.float32,
            batch=2,
            queries=3,
            heads=3,
            dim=4,
            use_non_default_stream=True,
        ),
        check(
            torch.float16,
            batch=2,
            queries=3,
            heads=3,
            dim=4,
            use_non_default_stream=True,
        ),
    ]
    passed = all(results)
    print("ALL PASSED" if passed else "SOME FAILED")
    raise SystemExit(0 if passed else 1)
