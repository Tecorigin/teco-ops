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

import importlib.util
import json
from pathlib import Path
import sys
import torch
import torch_sdaa

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('msda_list_existing_checks', root / 'python_api_test/test_ms_deform_attn_backward.py')
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)
tecoops = checks.owned_extension()
torch.set_num_threads(4)
torch.sdaa.set_device(0)
for dtype in (torch.float32,torch.float16):
    for name, side, queries in [('single_hot_chain',1,65536), ('four_hot_chains',2,65536), ('all_empty_heads',2,8192)]:
        shapes = torch.tensor([[side,side]],dtype=torch.int64)
        value = (torch.arange(side*side*3).remainder(9)-4).reshape(1,side*side,1,3).mul(2**-3).to(dtype)
        locations = torch.full((1,queries,1,1,1,2),3.0 if name=='all_empty_heads' else .5,dtype=dtype)
        weights = (torch.arange(queries).remainder(3)-1).reshape(1,queries,1,1,1).mul(2**-3).to(dtype)
        grad = (torch.arange(queries*3).remainder(5)-2).reshape(1,queries,3).mul(2**-8).to(dtype)
        data = (value,shapes,locations,weights,grad)
        for nondefault in ([False,True] if name=='single_hot_chain' else [False]):
            before = len(checks.TEST_RESULTS)
            checks.device_check(tecoops,dtype,name,nondefault=nondefault,data=data)
            # These dyadic inputs make contributions and all three sums exact.
            for row in checks.TEST_RESULTS[before:]:
                assert row['max_abs_error']==0, row
Path(root / 'list_stress_validation.json').write_text(json.dumps(dict(passed=True,
    longest_reachable_chain=65536, max_nodes_published=262144,
    gradients=checks.TEST_RESULTS, scope='signed dyadic single/four hot chains, empty heads, D3, raw/paired, FP32/FP16, default/nondefault'),indent=2)+'\n')
print('LIST STRESS PASS',flush=True)
