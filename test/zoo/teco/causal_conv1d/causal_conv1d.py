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
# encoding:utf-8


import os
import sys
import json
import math
import torch
import torch.nn.functional as F
import numpy as np
sys.path.append("../zoo/teco/")
sys.path.append("../")
from executor import *


def _causal_conv1d_fn_torch(
    out: torch.Tensor,
    conv_states: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    pad_slot_id: int,
) -> None:

    weight_conv = weight.transpose(0, 1).contiguous()
    dim, width = weight_conv.shape
    batch = query_start_loc.shape[0] - 1
    dtype_in = x.dtype

    weight_3d = weight_conv.unsqueeze(1)

    if cache_indices is not None:
        cache_idx_list = cache_indices.tolist()
    else:
        cache_idx_list = list(range(batch))

    if has_initial_state is not None:
        has_init_list = has_initial_state.tolist()
    else:
        has_init_list = [False] * batch

    for i in range(batch):
        state_idx = int(cache_idx_list[i])

        if cache_indices is not None and state_idx == pad_slot_id:
            continue

        start = int(query_start_loc[i].item())
        end = int(query_start_loc[i + 1].item())
        seqlen_i = end - start

        if seqlen_i == 0:
            continue

        if x.shape[0] == dim:
            seq_x = x[:, start:end].unsqueeze(0)
        else:
            seq_x = x[start:end, :].T.contiguous().unsqueeze(0)

        if has_init_list[i]:
            prefix = conv_states[state_idx].T.unsqueeze(0)
        else:
            prefix = torch.zeros(
                1, dim, width - 1, dtype=seq_x.dtype, device=seq_x.device
            )


        full_x = torch.cat([prefix, seq_x], dim=-1)


        conv_out = F.conv1d(
            full_x.to(weight_conv.dtype),
            weight_3d,
            padding=0,
            groups=dim,
        )


        conv_out = F.silu(conv_out).to(dtype_in)

        if out.shape[0] == dim:
            out[:, start:end] = conv_out.squeeze(0)
        else:
            out[start:end, :] = conv_out.squeeze(0).T.contiguous()

        new_state = (
            full_x[:, :, -(width - 1):]
            .squeeze(0)
            .T
            .contiguous()
        )
        conv_states[state_idx] = new_state


def check_inputs(param_path, input_lists, reuse_lists, output_lists):
    if param_path == "":
        print("The path of prototxt file is empty.")
        return False
    if len(input_lists) != 6:
        print("The number of input data is wrong (expected 6: x, weight, query_start_loc, cache_indices, has_initial_state, convState).")
        return False
    if len(reuse_lists) != 0:
        print("The number of reuse data is wrong.")
        return False
    if len(output_lists) != 2:
        print("The number of output data is wrong (expected 2: out, convState).")
        return False
    return True


def test_causal_conv1d(param_path, input_lists, reuse_lists, output_lists, device):
    if not check_inputs(param_path, input_lists, reuse_lists, output_lists):
        return
    if device == "cuda":
        is_avail, used_device = is_device_available(device)
        if not is_avail:
            return

    params = read_prototxt(param_path)
    input_params = params["input"]
    output_params = params["output"]

    # Read input tensors
    x = to_tensor(input_lists[0], input_params[0], device=device)  # [total_seqlen, dim] from prototxt
    total_seq_len = x.shape[0]
    dim = x.shape[1]
    x = x.T.contiguous()  # transpose to (dim, total_seqlen) for torch baseline
    weight = to_tensor(input_lists[1], input_params[1], device=device)  # [width, dim]
    query_start_loc = to_tensor(input_lists[2], input_params[2], device=device)  # [batch+1]
    cache_indices = to_tensor(input_lists[3], input_params[3], device=device)  # [batch]
    has_initial_state = to_tensor(input_lists[4], input_params[4], device=device)  # [batch]
    conv_states = to_tensor(input_lists[5], input_params[5], device=device)  # [batch, convStateLen, dim]

    # Promote to float32 for precision
    x = x.to(torch.float32)
    weight = weight.to(torch.float32)
    conv_states = conv_states.to(torch.float32)

    # Get kernel params from prototxt
    kernel_param = params.get("tecokernel_param", {})
    causal_param = kernel_param.get("causal_conv1d_param", {})
    pad_slot_id = int(causal_param.get("pad_slot_id", -1))

    # Prepare output tensor (dim, total_seqlen) for torch baseline
    batch = query_start_loc.shape[0] - 1
    width = weight.shape[0]

    out = torch.zeros(dim, total_seq_len, dtype=torch.float32, device=x.device)

    # Call torch reference implementation
    _causal_conv1d_fn_torch(
        out, conv_states, x, weight,
        query_start_loc, cache_indices, has_initial_state, pad_slot_id,
    )

    # Transpose output back to (total_seqlen, dim) for saving
    out = out.T.contiguous()

    # Save outputs
    with open(output_lists[0], "wb") as f:
        save_tensor(f, out, output_params[0]["dtype"])
    with open(output_lists[1], "wb") as f:
        save_tensor(f, conv_states, output_params[1]["dtype"])


def parse_params(filename):
    with open(filename, "r") as f:
        params = json.load(f)
    return params


if __name__ == "__main__":
    params = parse_params(sys.argv[1])
    device = sys.argv[2]
    test_causal_conv1d(params["param_path"], params["input_lists"],
                       params["reuse_lists"], params["output_lists"], device)
