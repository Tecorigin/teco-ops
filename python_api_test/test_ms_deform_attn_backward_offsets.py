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

from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import torch
import torch_sdaa

root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('existing_msda_value_dma_oracle',root/'python_api_test/test_ms_deform_attn_backward.py')
checks=importlib.util.module_from_spec(spec); spec.loader.exec_module(checks)
tecoops=checks.owned_extension()
torch.sdaa.set_device(0); torch.set_num_threads(4)
rows=[]
for dtype in (torch.float32,torch.float16):
    for dim in (3,31,128):
        for pattern in ('normal','half_subnormal_go','half_subnormal_value'):
            value_scale=2**-24 if pattern=='half_subnormal_value' else 2**-3
            value=(torch.arange(8*2*dim).remainder(17)-8).reshape(1,8,2,dim).mul(value_scale).to(dtype)
            shapes=torch.tensor([[2,3],[1,2]],dtype=torch.int64)
            locations=torch.full((1,3,2,2,2,2),.5,dtype=dtype)
            weights=torch.full((1,3,2,2,2),2**12 if pattern=='half_subnormal_go' else 1,dtype=dtype)
            grad=(torch.arange(3*2*dim).remainder(31)-15).reshape(1,3,2*dim).mul(2**-24 if pattern=='half_subnormal_go' else 2**-6).to(dtype)
            data=(value,shapes,locations,weights,grad)
            _,gold=checks.oracle(data)
            for nondefault in (False,True):
                stream=torch.sdaa.Stream() if nondefault else None
                with torch.sdaa.stream(stream) if stream else nullcontext():
                    s,loc,w=[x.to('sdaa') for x in data[1:4]]
                    value_buffer=torch.full((value.numel()+1,),100,dtype=dtype,device='sdaa')
                    v=value_buffer[1:].view(value.shape)
                    v.copy_(value)
                    assert v.storage_offset()==1 and v.is_contiguous()
                    buffer=torch.full((grad.numel()+1,),100,dtype=dtype,device='sdaa')
                    go=buffer[1:].view(grad.shape)
                    go.copy_(grad)
                    assert go.storage_offset()==1 and go.is_contiguous()
                    raw=tecoops.ms_deform_attn_backward(v,s,loc,w,go)
                    leaves=[x.detach().requires_grad_() for x in (v,loc,w)]
                    out=tecoops.ms_deform_attn(leaves[0],s,leaves[1],leaves[2])
                    paired=torch.autograd.grad(out,leaves,go)
                stream.synchronize() if stream else torch.sdaa.synchronize()
                for mode,result in [('raw',raw),('paired',paired)]:
                    for name,actual,expected in zip(('value','locations','weights'),result,gold):
                        expected=expected.to(dtype)
                        cpu=actual.cpu()
                        # Dyadic inputs make all sums exactly representable before
                        # the existing final HALF rounding, so require exact bits.
                        assert torch.equal(cpu,expected),(dtype,dim,pattern,nondefault,mode,name,float((cpu.float()-expected.float()).abs().max()))
                        if dtype==torch.float16:
                            assert torch.equal(cpu.view(torch.int16),expected.view(torch.int16))
                        rows.append(dict(dtype=str(dtype),D=dim,pattern=pattern,nondefault=nondefault,mode=mode,gradient=name,max_abs_error=0.0,value_storage_offset=int(v.storage_offset()),grad_output_storage_offset=int(go.storage_offset())))
                print(f'OFFSET PASS {dtype} D{dim} {pattern} nondefault={nondefault}',flush=True)
assert len(rows)==216
(root/'value_dma_offset_validation.json').write_text(json.dumps(dict(passed=True,cases=36,gradients=rows,
    scope='storage_offset=1 value and grad_output buffers, signed dyadic values, HALF subnormal value and grad_output input bits, raw/paired'),indent=2)+'\n')
print('VALUE + GO DMA OFFSET PASS',flush=True)
