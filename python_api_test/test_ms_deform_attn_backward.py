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

"""Independent MSDA first-order oracle and optional SDAA backward regression.

--cpu-only never imports the native extension or starts an SDAA task.
--real-shapes is additional operator-shape coverage, not a model training claim.
"""
import argparse
import json
import tempfile

TEST_RESULTS = []
from contextlib import nullcontext
import math
import hashlib
import importlib
from pathlib import Path
import sys
import types
import importlib.util
import torch
import torch.nn.functional as F


def reference(value, shapes, locations, weights):
    n, _, heads, dim = value.shape
    queries, points = locations.shape[1], locations.shape[4]
    result = value.new_zeros((n, queries, heads, dim))
    offset = 0
    for level, (height, width) in enumerate(shapes.tolist()):
        image = value[:, offset:offset + height * width].permute(0, 2, 3, 1)
        image = image.reshape(n * heads, dim, height, width)
        grid = locations[:, :, :, level].permute(0, 2, 1, 3, 4)
        grid = (grid * 2 - 1).reshape(n * heads, queries, points, 2)
        sampled = F.grid_sample(image, grid, align_corners=False, padding_mode="zeros")
        sampled = sampled.reshape(n, heads, dim, queries, points).permute(0, 3, 1, 4, 2)
        result = result + (sampled * weights[:, :, :, level, :, None]).sum(3)
        offset += height * width
    return result.flatten(2)


def scalar_gradient(value, shapes, locations, weights, grad_output):
    """CPU-only analytic oracle with direct zero-padded neighbor derivatives."""
    gv, gl, gw = torch.zeros_like(value), torch.zeros_like(locations), torch.zeros_like(weights)
    n, _, heads, dim = value.shape
    grad = grad_output.reshape(n, -1, heads, dim)
    start = 0
    for level, (height, width) in enumerate(shapes.tolist()):
        for batch in range(n):
            for query in range(locations.shape[1]):
                for head in range(heads):
                    for point in range(locations.shape[4]):
                        x, y = locations[batch, query, head, level, point].tolist()
                        x, y = x * width - .5, y * height - .5
                        x0, y0 = math.floor(x), math.floor(y)
                        fx, fy = x - x0, y - y0
                        aw = weights[batch, query, head, level, point]
                        g = grad[batch, query, head]
                        sample = value.new_zeros(dim)
                        dx, dy = value.new_zeros(dim), value.new_zeros(dim)
                        for iy, wy in enumerate((1 - fy, fy)):
                            for ix, wx in enumerate((1 - fx, fx)):
                                xx, yy = x0 + ix, y0 + iy
                                if 0 <= xx < width and 0 <= yy < height:
                                    index = start + yy * width + xx
                                    v = value[batch, index, head]
                                    sample += v * wx * wy
                                    dx += v * (-1 if ix == 0 else 1) * wy
                                    dy += v * (-1 if iy == 0 else 1) * wx
                                    gv[batch, index, head] += g * aw * wx * wy
                        gw[batch, query, head, level, point] = (g * sample).sum()
                        gl[batch, query, head, level, point, 0] = (g * aw * dx).sum() * width
                        gl[batch, query, head, level, point, 1] = (g * aw * dy).sum() * height
        start += height * width
    return gv, gl, gw


def inputs(dtype, name):
    if name == "half_rounding":
        value = torch.zeros(1, 1, 1, 8, dtype=dtype)
        shapes = torch.tensor([[1, 1]], dtype=torch.int64)
        locations = torch.full((1, 1, 1, 1, 1, 2), .5, dtype=dtype)
        weights = torch.full((1, 1, 1, 1, 1), 2 ** -11, dtype=dtype)
        grad = (torch.tensor([1, 3, 5, 7, -1, -3, -5, -7], dtype=dtype)
                * (2 ** -14)).reshape(1, 1, 8)
        return value, shapes, locations, weights, grad
    torch.manual_seed(1701)
    shapes = torch.tensor([[2, 3], [1, 2]], dtype=torch.int64)
    n, queries, heads, dim, points = 1, 3, 2, (128 if name == "dim128" else 4), 2
    value = torch.randn(n, 8, heads, dim).mul(.2).to(dtype)
    locations = torch.rand(n, queries, heads, 2, points, 2).mul(.6).add(.173).to(dtype)
    weights = torch.randn(n, queries, heads, 2, points).mul(.3).to(dtype)
    if name == "knot":
        shapes = torch.tensor([[2, 2]], dtype=torch.int64)
        value = value[:, :4].contiguous()
        locations = locations[:, :, :, :1].contiguous()
        weights = weights[:, :, :, :1].contiguous()
        locations[:, 0, :, :, 0] = torch.tensor([-.25, .25], dtype=dtype)
        locations[:, 1, :, :, 0] = torch.tensor([.25, -.25], dtype=dtype)
        locations[:, 2, :, :, 0] = torch.tensor([1.25, .25], dtype=dtype)
    elif name == "overlap":
        locations[:] = torch.tensor([.233, .317], dtype=dtype)
    elif name == "border":
        locations[:, 0, :, :, 0] = torch.tensor([0., 1.], dtype=dtype)
        locations[:, 1, :, :, 0] = torch.tensor([-.8, 1.8], dtype=dtype)
    elif name == "tiny_value_grad":
        weights.fill_(2 ** -10)
    elif name == "subnormal":
        value = (torch.arange(value.numel()).reshape(value.shape).remainder(17) - 8).mul(2 ** -24).to(dtype)
    grad = torch.randn(n, queries, heads * dim).mul(.2).to(dtype)
    if name == "tiny_value_grad":
        grad.fill_(2 ** -12)
    return value, shapes, locations, weights, grad


def oracle(data):
    value, shapes, locations, weights, grad = data
    leaves = [x.float().detach().requires_grad_() for x in (value, locations, weights)]
    out = reference(leaves[0], shapes, leaves[1], leaves[2])
    grads = torch.autograd.grad(out, leaves, grad.float())
    return out.detach(), tuple(x.detach() for x in grads)


def cpu_checks():
    count = 0
    for dtype in (torch.float32, torch.float16):
        for name in ("overlap", "border", "knot", "multilevel", "subnormal", "tiny_value_grad", "half_rounding"):
            data = inputs(dtype, name)
            _, grads = oracle(data)
            v, shapes, loc, weights, grad = data
            analytical = scalar_gradient(v.float(), shapes, loc.float(), weights.float(), grad.float())
            for actual, expected in zip(analytical, grads):
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
            count += 1
    # Finite differences away from pixel boundaries; also checks coordinate W/H factors.
    shapes = torch.tensor([[2, 3]], dtype=torch.int64)
    torch.manual_seed(22)
    value = torch.randn(1, 6, 1, 2, dtype=torch.float64, requires_grad=True)
    loc = torch.tensor([.213, .319], dtype=torch.float64).reshape(1, 1, 1, 1, 1, 2).requires_grad_()
    weights = torch.randn(1, 1, 1, 1, 1, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda v, l, w: reference(v, shapes, l, w), (value, loc, weights), eps=1e-6)
    print(f"CPU gradient oracle PASS: {count} quantized cases + FP64 finite differences", flush=True)



def cpu_wrapper_contract():
    # Load the package with mock native symbols; no device runtime or shared library is used.
    root = Path(__file__).resolve().parents[1]
    package = "_msda_cpu_wrapper_contract"
    native = types.ModuleType(package + "._torch_ext")
    def forbidden(*args):
        raise AssertionError("unexpected native call in CPU wrapper check")
    for name in ("flatten_rays", "morton3D_invert", "reshape_and_cache", "rms_norm",
                 "flash_attn_varlen_func", "causal_conv1d_fn_torch", "ms_deform_attn_forward"):
        setattr(native, name, forbidden)
    captured = []
    def backward_mock(*args):
        captured.append(args)
        assert args[-1].is_contiguous()
        return torch.ones_like(args[0]), torch.ones_like(args[2]), torch.ones_like(args[3])
    native.ms_deform_attn_backward = backward_mock
    sys.modules[native.__name__] = native
    spec = importlib.util.spec_from_file_location(package, root / "api/tecoops/__init__.py",
                                                 submodule_search_locations=[str(root / "api/tecoops")])
    module = importlib.util.module_from_spec(spec)
    sys.modules[package] = module
    spec.loader.exec_module(module)
    v, shapes, loc, weights, _ = inputs(torch.float32, "overlap")
    try:
        module.ms_deform_attn(v, shapes, loc, weights)
    except ValueError:
        pass
    else:
        raise AssertionError("autograd wrapper accepted CPU forward")
    ctx = types.SimpleNamespace(saved_tensors=(v, shapes, loc, weights))
    grad = torch.ones(1, 1, 8).expand(1, 3, 8)
    result = module._MSDeformAttn.backward(ctx, grad)
    assert len(captured) == 1 and len(result) == 4 and result[1] is None
    assert result[0].shape == v.shape and result[2].shape == loc.shape and result[3].shape == weights.shape
    print("CPU mocked autograd wrapper contract PASS", flush=True)


def device_check(tecoops, dtype, name, nondefault=False, data=None):
    data = inputs(dtype, name) if data is None else data
    expected_out, expected_grads = oracle(data)
    stream = torch.sdaa.Stream() if nondefault else None
    with torch.sdaa.stream(stream) if stream else nullcontext():
        v, shapes, loc, weights, grad = [x.to("sdaa") for x in data]
        # Producers execute on the selected stream before workspace zeroing/native reads.
        v, loc, weights, grad = [x.clone().contiguous() for x in (v, loc, weights, grad)]
        actual = tecoops.ms_deform_attn_backward(v, shapes, loc, weights, grad)
        leaves = [x.detach().requires_grad_() for x in (v, loc, weights)]
        out = tecoops.ms_deform_attn(leaves[0], shapes, leaves[1], leaves[2])
        paired = torch.autograd.grad(out, leaves, grad)
    if stream:
        stream.synchronize()
    else:
        torch.sdaa.synchronize()
    atol, rtol = (2e-3, 3e-3) if dtype == torch.float16 else (3e-5, 3e-4)
    for mode, result in zip(("raw", "autograd"), (actual, paired)):
        for index, (got, expected) in enumerate(zip(result, expected_grads)):
            assert got.dtype == dtype and got.device.type == "sdaa"
            got_cpu = got.cpu().float()
            expected_cpu = expected.to(dtype).float()
            TEST_RESULTS.append(dict(case=name, dtype=str(dtype), nondefault_stream=nondefault, mode=mode,
                                     gradient=("value", "locations", "weights")[index],
                                     max_abs_error=float((got_cpu - expected_cpu).abs().max()),
                                     rms_error=float((got_cpu - expected_cpu).square().mean().sqrt())))
            this_atol, this_rtol = atol, rtol
            if (name == "subnormal" and index > 0) or (name == "tiny_value_grad" and index == 0):
                this_atol = 2 ** -24 if dtype == torch.float16 else 1e-11
                this_rtol = 0 if dtype == torch.float16 else 3e-4
                if torch.count_nonzero(expected_cpu):
                    assert torch.count_nonzero(got_cpu), "subnormal gradients flushed entirely to zero"
            torch.testing.assert_close(got_cpu, expected_cpu, atol=this_atol, rtol=this_rtol)
            if name == "half_rounding" and dtype == torch.float16 and index == 0:
                assert torch.equal(got.cpu().view(torch.int16), expected.to(dtype).view(torch.int16)), "half RNE tie bits differ"
    torch.testing.assert_close(out.cpu().float(), expected_out, atol=atol, rtol=rtol)
    print(f"SDAA backward PASS: {name} {dtype} nondefault={nondefault}", flush=True)


def rejection_checks(tecoops):
    data = inputs(torch.float32, "overlap")
    try:
        tecoops.ms_deform_attn_backward(*data)
    except (RuntimeError, ValueError):
        pass
    else:
        raise AssertionError("raw backward accepted CPU pointers")
    v, shapes, loc, w, g = [x.to("sdaa") for x in data]
    bad = [(v, shapes.cpu(), loc, w, g), (v, shapes, loc.half(), w, g),
           (v, shapes, loc, w, g[..., :1]), (v, shapes, loc, w, g.transpose(1, 2))]
    for args in bad:
        try:
            tecoops.ms_deform_attn_backward(*args)
        except RuntimeError:
            pass
        else:
            raise AssertionError("invalid backward inputs accepted")
    wide_v = torch.zeros(1, 8, 2, 129, device="sdaa")
    wide_g = torch.zeros(1, 3, 258, device="sdaa")
    nine_shapes = torch.ones(9, 2, dtype=torch.int64, device="sdaa")
    nine_v = torch.zeros(1, 9, 2, 4, device="sdaa")
    nine_loc = loc[:, :, :, :1].expand(1, 3, 2, 9, 2, 2).contiguous()
    nine_w = w[:, :, :, :1].expand(1, 3, 2, 9, 2).contiguous()
    for invalid in ((wide_v, shapes, loc, w, wide_g), (nine_v, nine_shapes, nine_loc, nine_w, g)):
        try:
            tecoops.ms_deform_attn_backward(*invalid)
        except RuntimeError:
            pass
        else:
            raise AssertionError("D129/L9 unsupported configuration accepted")
    print("backward input rejection PASS: CPU/device/dtype/layout/shape/D129/L9", flush=True)


def real_shapes(tecoops):
    shapes = torch.tensor([[100, 167], [50, 84], [25, 42], [13, 21]], dtype=torch.int64)
    length = int(shapes.prod(1).sum())
    for queries, label in ((length, "encoder"), (300, "decoder")):
        torch.manual_seed(19)
        data = (torch.randn(2, length, 8, 32).mul(.1), shapes,
                torch.rand(2, queries, 8, 4, 4, 2), torch.rand(2, queries, 8, 4, 4).mul(.05),
                torch.randn(2, queries, 256).mul(.05))
        for dtype in (torch.float32, torch.float16):
            quantized = tuple(t if t.dtype == torch.int64 else t.to(dtype) for t in data)
            device_check(tecoops, dtype, label, data=quantized)


def owned_extension():
    """Use the installed wheel, or an in-place build explicitly on PYTHONPATH.

    Never prepend an unbuilt api/ tree over the package selected by the caller.
    Keep the extension/core provenance checks relative to that selected package.
    """
    tecoops = importlib.import_module("tecoops")
    extension = importlib.import_module("tecoops._torch_ext")
    package_dir = Path(tecoops.__file__).resolve().parent
    ext_path = Path(extension.__file__).resolve()
    assert ext_path.parent == package_dir, f"unexpected extension: {ext_path}"
    cores = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        path = line.split()[-1]
        if path.startswith("/") and Path(path).name == "libteco_ops.so":
            cores.add(Path(path).resolve())
    assert len(cores) == 1, f"expected one owned libteco_ops.so, found {cores}"
    core = next(iter(cores))
    assert core.parent == package_dir, f"unexpected loaded core: {core}"
    print(f"tecoops.__file__={tecoops.__file__}", flush=True)
    print(f"_torch_ext.__file__={extension.__file__}", flush=True)
    for label, path in (("extension", ext_path), ("core", core)):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{label} path={path} SHA256={digest}", flush=True)
    print(f"/proc/self/maps libteco_ops.so={sorted(map(str, cores))}", flush=True)
    return tecoops


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--real-shapes", action="store_true")
    parser.add_argument(
        "--output-json",
        help="Persist the detailed report here; by default the same JSON is emitted from a temporary directory and cleaned up.",
    )
    args = parser.parse_args()
    cpu_checks()
    cpu_wrapper_contract()
    if not args.cpu_only:
        import torch_sdaa  # register SDAA; device execution starts only in this branch
        tecoops = owned_extension()
        rejection_checks(tecoops)
        for dtype in (torch.float32, torch.float16):
            for name in ("overlap", "border", "knot", "multilevel", "subnormal", "tiny_value_grad", "half_rounding"):
                device_check(tecoops, dtype, name)
            device_check(tecoops, dtype, "dim128")
            device_check(tecoops, dtype, "overlap", nondefault=True)
        if args.real_shapes:
            real_shapes(tecoops)

    if not args.cpu_only:
        report = dict(passed=True, real_shapes=args.real_shapes, gradients=TEST_RESULTS)
        serialized = json.dumps(report, indent=2) + "\n"
        if args.output_json:
            report_path = Path(args.output_json).expanduser()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(serialized)
            print(f"backward validation JSON saved to {report_path}", flush=True)
        else:
            with tempfile.TemporaryDirectory(prefix="msda-backward-validation-") as report_dir:
                report_path = Path(report_dir) / "backward_validation.json"
                report_path.write_text(serialized)
                print("BEGIN backward_validation.json (temporary; use --output-json to retain)", flush=True)
                print(serialized, end="", flush=True)
                print("END backward_validation.json", flush=True)
