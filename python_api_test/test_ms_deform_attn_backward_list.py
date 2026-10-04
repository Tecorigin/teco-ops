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
