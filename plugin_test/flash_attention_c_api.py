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

"""Direct C API rejection checks; requires SDAA runtime, not the plugin SDK.

Run with the same library mapped by the SDK under the shared device flock:
  /home/py312/bin/python plugin_test/flash_attention_c_api.py --core /path/to/libteco_ops_plugin.so
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path

import numpy as np


def run_flash(core, arrays, scale=None, cumulative=False, gid=0):
    """Call the mapped C implementation with device allocations and a sentinel."""
    rt = C.CDLL('/opt/tecoai/lib64/libsdaart.so', mode=C.RTLD_GLOBAL)
    lib = C.CDLL(str(Path(core).resolve()), mode=C.RTLD_GLOBAL)
    ptr = C.c_void_p
    rt.sdaaMalloc.argtypes = [C.POINTER(ptr), C.c_size_t]
    rt.sdaaFree.argtypes = [ptr]
    rt.sdaaMemcpy.argtypes = [ptr, ptr, C.c_size_t, C.c_int]
    lib.tecoopsCreate.argtypes = [C.POINTER(ptr)]
    lib.tecoopsDestroy.argtypes = [ptr]
    lib.tecoopsCreateTensorDescriptor.argtypes = [C.POINTER(ptr)]
    lib.tecoopsDestroyTensorDescriptor.argtypes = [ptr]
    lib.tecoopsSetTensorNdDescriptor.argtypes = [ptr, C.c_int, C.c_int, C.POINTER(C.c_int), ptr]
    def checked(code):
        if code != 0:
            raise RuntimeError(f'SDAA/descriptor operation status={code}')
    checked(rt.sdaaSetDevice(gid))
    host = [np.ascontiguousarray(x) for x in arrays]
    out = np.full(host[0].shape, 7.0, dtype=np.float16)
    allocations, descriptors = [], []
    handle = ptr()
    checked(lib.tecoopsCreate(C.byref(handle)))
    try:
        for x in host + [out]:
            p = ptr()
            checked(rt.sdaaMalloc(C.byref(p), x.nbytes))
            allocations.append(p)
            checked(rt.sdaaMemcpy(p, ptr(x.ctypes.data), x.nbytes, 1))
        for x in (host[3], host[0], host[1], host[2], out):
            d = ptr()
            checked(lib.tecoopsCreateTensorDescriptor(C.byref(d)))
            descriptors.append(d)
            dims = (C.c_int * x.ndim)(*x.shape)
            checked(lib.tecoopsSetTensorNdDescriptor(d, 4 if x.dtype == np.int32 else 1,
                                                     x.ndim, dims, None))
        suffix = ('CuSeqlensQ' if cumulative else '') + ('WithScale' if scale is not None else '')
        fn = getattr(lib, 'tecoopsFlashAttention' + suffix)
        fn.argtypes = [ptr, C.c_int, C.c_int, C.c_int] + ([C.c_float] if scale is not None else []) + [ptr] * 13
        qlens = np.diff(host[4]) if cumulative else host[4]
        args = [handle, int(max(qlens)), int(max(host[5])), int(host[1].shape[0])]
        if scale is not None:
            args.append(float(scale))
        args += [allocations[4], allocations[5], descriptors[0], allocations[3],
                 descriptors[1], allocations[0], descriptors[2], allocations[1],
                 descriptors[3], allocations[2], descriptors[4], allocations[6], None]
        status = fn(*args)
        checked(rt.sdaaDeviceSynchronize())
        checked(rt.sdaaMemcpy(ptr(out.ctypes.data), allocations[6], out.nbytes, 2))
        return status, out
    finally:
        for d in descriptors:
            lib.tecoopsDestroyTensorDescriptor(d)
        lib.tecoopsDestroy(handle)
        for p in allocations:
            rt.sdaaFree(p)


def check_unsupported(core, gid=0):
    """Check both unsupported layouts with owned, prefilled device output."""
    core = Path(core).resolve(strict=True)
    digest = hashlib.sha256(core.read_bytes()).hexdigest()
    for head_size, block_size in ((48, 32), (64, 64)):
        arrays = [np.ones((32, 8, head_size), dtype=np.float16),
                  np.zeros((1, 4, block_size, head_size), dtype=np.float16),
                  np.zeros((1, 4, block_size, head_size), dtype=np.float16),
                  np.zeros((1, 1), dtype=np.int32),
                  np.array([32], dtype=np.int32),
                  np.array([block_size], dtype=np.int32)]
        status, out = run_flash(core, arrays, gid=gid)
        assert status == 6, f"head={head_size}, block={block_size}: expected status 6, got {status}"
        assert np.all(out == 7.0), f"head={head_size}, block={block_size}: rejected C API wrote output"
        print(json.dumps(dict(core=str(core), sha256=digest, gid=gid,
                              head_size=head_size, block_size=block_size,
                              C_API_status=status, sentinel_untouched=True)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", required=True, help="Exact core/plugin .so to test")
    parser.add_argument("--gid", type=int, default=0, help="Visible SDAA device ordinal")
    options = parser.parse_args()
    check_unsupported(options.core, options.gid)
