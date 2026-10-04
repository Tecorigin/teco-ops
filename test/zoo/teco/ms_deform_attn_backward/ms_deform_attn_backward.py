# BSD 3-Clause License
#
# Copyright (c) 2024, Tecorigin Co., Ltd.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
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
from pathlib import Path
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from executor import read_prototxt, to_tensor, save_tensor


def reference_gradients(value, shapes, locations, weights, grad_output):
    value, locations, weights = [x.float().detach().requires_grad_() for x in (value, locations, weights)]
    n, _, heads, dim = value.shape
    queries, points = locations.shape[1], locations.shape[4]
    result = value.new_zeros((n, queries, heads, dim))
    start = 0
    for level, (height, width) in enumerate(shapes.tolist()):
        image = value[:, start:start + height * width].permute(0, 2, 3, 1).reshape(n * heads, dim, height, width)
        grid = (2 * locations[:, :, :, level].permute(0, 2, 1, 3, 4) - 1).reshape(n * heads, queries, points, 2)
        sampled = F.grid_sample(image, grid, padding_mode="zeros", align_corners=False)
        sampled = sampled.reshape(n, heads, dim, queries, points).permute(0, 3, 1, 4, 2)
        result = result + (sampled * weights[:, :, :, level, :, None]).sum(3)
        start += height * width
    return torch.autograd.grad(result.flatten(2), (value, locations, weights), grad_output.float())


def test_ms_deform_attn_backward(param_path, input_lists, reuse_lists, output_lists, device):
    if device != "cpu" or len(input_lists) != 5 or reuse_lists or len(output_lists) != 3:
        raise ValueError("MSDA backward baseline requires CPU, five inputs and three outputs")
    params = read_prototxt(param_path)
    tensors = [to_tensor(path, spec, device="cpu") for path, spec in zip(input_lists, params["input"])]
    gradients = reference_gradients(*tensors)
    for path, spec, gradient in zip(output_lists, params["output"], gradients):
        with open(path, "wb") as stream:
            save_tensor(stream, gradient, spec["dtype"])


if __name__ == "__main__":
    with open(sys.argv[1]) as stream:
        params = json.load(stream)
    test_ms_deform_attn_backward(params["param_path"], params["input_lists"], params["reuse_lists"],
                                params["output_lists"], sys.argv[2])
