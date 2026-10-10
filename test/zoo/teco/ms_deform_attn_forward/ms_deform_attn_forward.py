# BSD 3-Clause License
#
# Copyright (c) 2024, Tecorigin Co., Ltd.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import json
import sys

import torch
import torch.nn.functional as F

sys.path.append("../zoo/teco/")
sys.path.append("../")
from executor import *  # noqa: F401,F403


def reference(value, spatial_shapes, sampling_locations, attention_weights):
    batch, _, heads, dim = value.shape
    _, queries, _, levels, points, _ = sampling_locations.shape
    outputs = []
    start = 0
    for level in range(levels):
        height, width = [int(x) for x in spatial_shapes[level].tolist()]
        value_level = value[:, start:start + height * width]
        value_level = value_level.reshape(batch, height * width, heads, dim)
        value_level = value_level.permute(0, 2, 3, 1).reshape(
            batch * heads, dim, height, width
        )
        grid = (2 * sampling_locations[:, :, :, level] - 1).permute(
            0, 2, 1, 3, 4
        ).reshape(batch * heads, queries, points, 2)
        outputs.append(F.grid_sample(
            value_level, grid, mode="bilinear", padding_mode="zeros",
            align_corners=False,
        ))
        start += height * width

    stacked = torch.stack(outputs, dim=-2).reshape(
        batch * heads, dim, queries, levels * points
    )
    weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(
        batch * heads, 1, queries, levels * points
    )
    return (stacked * weights).sum(-1).reshape(
        batch, heads, dim, queries
    ).permute(0, 3, 1, 2).reshape(batch, queries, heads * dim)


def check_inputs(param_path, input_lists, reuse_lists, output_lists):
    if not param_path or len(input_lists) != 4 or reuse_lists or len(output_lists) != 1:
        return False
    return True


def test_ms_deform_attn_forward(param_path, input_lists, reuse_lists, output_lists, device):
    if not check_inputs(param_path, input_lists, reuse_lists, output_lists):
        return
    params = read_prototxt(param_path)
    input_params = params["input"]
    output_params = params["output"]
    if isinstance(output_params, dict):
        output_params = [output_params]
    value = to_tensor(input_lists[0], input_params[0], device=device)
    spatial_shapes = to_tensor(input_lists[1], input_params[1], device=device)
    locations = to_tensor(input_lists[2], input_params[2], device=device)
    weights = to_tensor(input_lists[3], input_params[3], device=device)
    result = reference(value.float(), spatial_shapes, locations.float(), weights.float())
    with open(output_lists[0], "wb") as f:
        save_tensor(f, result, output_params[0]["dtype"])


def main():
    with open(sys.argv[1], "r") as f:
        params = json.load(f)
    test_ms_deform_attn_forward(
        params["param_path"], params["input_lists"], params["reuse_lists"],
        params["output_lists"], sys.argv[2]
    )


if __name__ == "__main__":
    main()
