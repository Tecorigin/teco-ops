#!/usr/bin/env python3
# BSD 3- Clause License Copyright (c) 2024, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.

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
    storage_offset=0,
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
        if storage_offset:
            packed = torch.empty(value.numel() + storage_offset, dtype=dtype, device=device)
            packed[storage_offset:].copy_(value.reshape(-1))
            value = packed[storage_offset:].reshape(value.shape)
            assert value.is_contiguous() and value.storage_offset() == storage_offset
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


def check_subnormal_coordinates(dtype):
    device = torch.device("sdaa")
    batch, queries, heads, dim = 2, 4, 2, 3
    height, width = 1, 167
    shapes_cpu = torch.tensor([[height, width]], dtype=torch.int64)
    value_cpu = torch.full(
        (batch, height * width, heads, dim), 16.0, dtype=torch.float32
    ).to(dtype)
    x_coordinates = torch.tensor(
        [6.079673767e-5, -5.793571472e-5, 0.0, 2.0**-14],
        dtype=torch.float32,
    )
    locations_cpu = torch.zeros(
        (batch, queries, heads, 1, 1, 2), dtype=torch.float32
    )
    locations_cpu[..., 0] = x_coordinates.view(1, queries, 1, 1, 1)
    locations_cpu[..., 1] = 0.5
    locations_cpu = locations_cpu.to(dtype)
    weights_cpu = torch.ones((batch, queries, heads, 1, 1), dtype=dtype)

    # Transfer directly from CPU so device-side arithmetic cannot flush the
    # subnormal half coordinate bit patterns before the operator sees them.
    shapes = shapes_cpu.to(device=device)
    value = value_cpu.to(device=device).contiguous()
    locations = locations_cpu.to(device=device).contiguous()
    weights = weights_cpu.to(device=device).contiguous()
    output = tecoops.ms_deform_attn_forward(value, shapes, locations, weights)

    expected = reference(
        value_cpu.float(), shapes_cpu, locations_cpu.float(), weights_cpu.float()
    )
    error = (output.cpu().float() - expected).abs().max().item()
    limit = 5e-5 if dtype == torch.float32 else 2e-3
    print(
        f"dtype={dtype} subnormal-coordinate shape=(N={batch},Lq={queries},"
        f"M={heads},D={dim}) max_error={error:.6e} "
        f"{'PASSED' if error < limit else 'FAILED'}"
    )
    return error < limit


def check_subnormal_values():
    dtype = torch.float16
    shapes_cpu = torch.tensor([[2, 3]], dtype=torch.int64)
    value_cpu = (torch.arange(2 * 6 * 3 * 3).reshape(2, 6, 3, 3).remainder(17) - 8)
    value_cpu = (value_cpu.float() * 2.0**-24).to(dtype)
    locations_cpu = torch.full((2, 3, 3, 1, 2, 2), 0.375, dtype=dtype)
    weights_cpu = torch.full((2, 3, 3, 1, 2), 32768.0, dtype=dtype)
    stream = torch.sdaa.Stream()
    with torch.sdaa.stream(stream):
        packed = torch.cat((torch.zeros(1, dtype=dtype), value_cpu.flatten())).to("sdaa")
        value = packed[1:].reshape(value_cpu.shape)
        output = tecoops.ms_deform_attn_forward(
            value, shapes_cpu.to("sdaa"), locations_cpu.to("sdaa"), weights_cpu.to("sdaa")
        )
    stream.synchronize()
    expected = reference(value_cpu.float(), shapes_cpu, locations_cpu.float(), weights_cpu.float())
    error = (output.cpu().float() - expected).abs().max().item()
    print(f"FP16 amplified signed-subnormal values, offset1 max_error={error:.6e}")
    return error < 2e-3


def check_rejected_parameters():
    for dim, levels in [(129, 1), (4, 9)]:
        shapes = torch.ones((levels, 2), dtype=torch.int64, device="sdaa")
        value = torch.ones((1, levels, 1, dim), device="sdaa")
        locations = torch.full((1, 1, 1, levels, 1, 2), 0.5, device="sdaa")
        weights = torch.ones((1, 1, 1, levels, 1), device="sdaa")
        try:
            tecoops.ms_deform_attn_forward(value, shapes, locations, weights)
        except RuntimeError as error:
            if "rejected parameters" not in str(error):
                raise
        else:
            raise AssertionError(f"unsupported dim={dim}, levels={levels} returned an output")
    print("C API rejected-parameter propagation PASSED")
    return True


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
        check_subnormal_coordinates(torch.float32),
        check_subnormal_coordinates(torch.float16),
        check_rejected_parameters(),
    ]
    for dtype in (torch.float32, torch.float16):
        for dim in (1, 3, 31, 32, 128):
            results.append(check(dtype, batch=2, heads=3, dim=dim,
                                 use_non_default_stream=dim in (3, 31), storage_offset=1))
    results.append(check_subnormal_values())
    passed = all(results)
    print("ALL PASSED" if passed else "SOME FAILED")
    raise SystemExit(0 if passed else 1)
