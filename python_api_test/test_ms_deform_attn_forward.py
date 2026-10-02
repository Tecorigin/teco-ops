#!/usr/bin/env python3
"""Accuracy test for the inference-only MSDeformAttn forward primitive."""

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


def check(dtype):
    device = torch.device("sdaa")
    shapes = torch.tensor([[3, 4], [2, 2]], dtype=torch.int64, device=device)
    batch, queries, heads, dim, points = 1, 5, 2, 3, 2
    value = torch.randn(batch, 16, heads, dim, dtype=dtype, device=device)
    locations = torch.rand(
        batch, queries, heads, 2, points, 2, dtype=dtype, device=device
    ).contiguous()
    weights = torch.rand(
        batch, queries, heads, 2, points, dtype=dtype, device=device
    ).contiguous()
    output = tecoops.ms_deform_attn_forward(value, shapes, locations, weights)
    expected = reference(value, shapes, locations, weights)
    error = (output.float() - expected.float()).abs().max().item()
    limit = 5e-5 if dtype == torch.float32 else 2e-3
    print(f"dtype={dtype} max_error={error:.6e} {'PASSED' if error < limit else 'FAILED'}")
    return error < limit


if __name__ == "__main__":
    passed = check(torch.float32) and check(torch.float16)
    print("ALL PASSED" if passed else "SOME FAILED")
    raise SystemExit(0 if passed else 1)
