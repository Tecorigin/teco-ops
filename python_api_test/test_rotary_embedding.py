#!/usr/bin/env python3
# BSD 3- Clause License Copyright (c) 2024, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
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
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Python-API conformance test for ``tecoops.rotary_embedding``.

Operator contract, taken from the delivered sources (not from a guess):

  * ``teco/ual/args/rotary_embedding_args.h``  -- argument struct / layouts;
  * ``teco/ual/kernel/rotary_embedding/rotary_embedding_fp16.scpp`` -- the
    arithmetic actually executed by ``teco_slave_rotary_embedding_fp16``;
  * ``api/torch_ext.cpp`` -- the ``m.def("rotary_embedding", ...)`` wrapper;
  * ``api/tecoops/__init__.py`` -- the export.

      q/k            : fp16, [num_tokens, heads * head_dim]   (row stride free)
      cos_sin_cache  : fp16, [max_position, head_dim] = [cos | sin]
      positions      : int64, [num_tokens]
      q_out/k_out    : fp16, [num_tokens, heads * head_dim]   (may alias q/k)
      is_neox        : must be True (NeoX / half-split layout)

      for each (t, h, d in [0, head_dim/2)), c = cos[t][d], s = sin[t][d],
      x1 = x[t][h][d], x2 = x[t][h][d + head_dim/2]:
          o1 = fp16(fp32(x1)*fp32(c)) - fp16(fp32(x2)*fp32(s))
          o2 = fp16(fp32(x2)*fp32(c)) + fp16(fp32(x1)*fp32(s))

  Every fp16 operation rounds to fp16 exactly once, which is the reference
  chain's order (vLLM ``RotaryEmbedding.forward_static`` +
  ``ApplyRotaryEmb.forward_static``), so the delivered kernel is expected to be
  bit-identical to it.

Two gates, both non-zero-on-failure so ``python_api_test`` CI steps fail loudly:

  1. CPU gate (always runs, also with ``--cpu-only``): the faithful kernel model
     must be bit-identical to the vLLM reference chain on the real MiniCPM5-1B
     geometries (decode T=1 and prefill T=7/T=512, dense and padded fused-QKV row
     stride).  ``max_error == 0``.
  2. Device gate (run only when an SDAA device is present): the delivered kernel
     must be bit-identical to the same reference chain, except on elements where
     an fp16 subnormal is involved: SDAA flushes fp16 denormals while the CPU
     IEEE model does not (measured by the rms_norm_compat work).  Every
     deviation must be inside that mask.

  In-place calls use the same CPU/subnormal gate, plus a strict bit comparison
  against an out-of-place device call on identical inputs. Padded cases move the
  entire fused-QKV buffer to SDAA before slicing, so both input and in-place
  output retain the real row stride. The untouched V slice must remain exact.

Run as CI runs it::

    cd python_api_test && python test_rotary_embedding.py

``--cpu-only`` skips the device steps; it is a self-check path for the CPU gate.
"""
import argparse
import json
import sys

import _bootstrap  # noqa: F401  (adds the repo api/ tree when tecoops is not installed)
import torch

HALF = torch.float16
SUBNORMAL = 2.0 ** -14


def record(fail_list, **kw):
    print("  " + " ".join("%s=%s" % (k, v) for k, v in kw.items()))
    if kw.get("ok") is False:
        fail_list.append(kw)


def make_case(t, hq, hkv, d, seed, padded=False, scale=1.0):
    """Real fused-QKV layout: q/k are views into one [T, (Hq+2*Hkv)*D] buffer."""
    g = torch.Generator().manual_seed(seed)
    full = (hq + 2 * hkv) * d
    row = full if padded else None
    if padded:
        qkv = (torch.randn((t, full), generator=g, dtype=torch.float32) * scale).to(HALF)
    else:
        g2 = torch.Generator().manual_seed(seed)
        q = (torch.randn((t, hq * d), generator=g2, dtype=torch.float32) * scale).to(HALF)
        k = (torch.randn((t, hkv * d), generator=g2, dtype=torch.float32) * scale).to(HALF)
        qkv = torch.cat((q, k, torch.zeros((t, hkv * d), dtype=HALF)), dim=-1)
    maxpos = 4096
    inv = 1.0 / (10000.0 ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    pos = torch.arange(maxpos, dtype=torch.float32)
    freqs = torch.outer(pos, inv)
    cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(HALF)
    positions = (torch.arange(t, dtype=torch.int64) * 7 + 3) % maxpos
    return qkv, cache, positions


def slices(qkv, hq, hkv, d):
    q = qkv[:, :hq * d]
    k = qkv[:, hq * d:(hq + hkv) * d]
    return q, k


def reference_chain(q, k, cache, positions, hq, hkv, d):
    """vLLM 0.20.2 RotaryEmbedding.forward_static + ApplyRotaryEmb.forward_static."""
    cos_sin = cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)
    return (_apply(q, cos, sin, hq, d), _apply(k, cos, sin, hkv, d))


def _apply(x, cos, sin, heads, d):
    t = x.shape[0]
    xv = x.view(t, heads, d)
    cosb = cos.unsqueeze(-2)
    sinb = sin.unsqueeze(-2)
    x1, x2 = torch.chunk(xv, 2, dim=-1)
    o1 = x1 * cosb - x2 * sinb
    o2 = x2 * cosb + x1 * sinb
    return torch.cat((o1, o2), dim=-1).reshape(t, heads * d)


def kernel_model(q, k, cache, positions, hq, hkv, d):
    """Rounding-order model of teco_slave_rotary_embedding_fp16."""
    h = d // 2
    outs = []
    for x, heads in ((q, hq), (k, hkv)):
        out = torch.empty((x.shape[0], heads * d), dtype=HALF)
        for t in range(x.shape[0]):
            p = int(positions[t])
            c = cache[p, :h].float()
            s = cache[p, h:2 * h].float()
            for hh in range(heads):
                x1 = x[t, hh * d:hh * d + h].float()
                x2 = x[t, hh * d + h:hh * d + d].float()
                t1 = (x1 * c).to(HALF)
                t2 = (x2 * s).to(HALF)
                t3 = (x2 * c).to(HALF)
                t4 = (x1 * s).to(HALF)
                out[t, hh * d:hh * d + h] = (t1.float() - t2.float()).to(HALF)
                out[t, hh * d + h:hh * d + d] = (t3.float() + t4.float()).to(HALF)
        outs.append(out)
    return outs[0], outs[1]


def denormal_involved(q, k, cache, positions, hq, hkv, d):
    """Mask of elements whose reference chain touches an fp16 subnormal."""
    h = d // 2
    masks = []
    for x, heads in ((q, hq), (k, hkv)):
        m = torch.zeros((x.shape[0], heads * d), dtype=torch.bool)
        for t in range(x.shape[0]):
            p = int(positions[t])
            c = cache[p, :h].float()
            s = cache[p, h:2 * h].float()
            for hh in range(heads):
                x1 = x[t, hh * d:hh * d + h].float()
                x2 = x[t, hh * d + h:hh * d + d].float()
                t1 = (x1 * c).to(HALF).float()
                t2 = (x2 * s).to(HALF).float()
                t3 = (x2 * c).to(HALF).float()
                t4 = (x1 * s).to(HALF).float()
                o1 = (t1 - t2).to(HALF).float()
                o2 = (t3 + t4).to(HALF).float()
                first = torch.stack([t1, t2, t3, t4], 0).abs()
                fh = ((first > 0) & (first < SUBNORMAL)).any(0)
                sh = ((o1.abs() > 0) & (o1.abs() < SUBNORMAL)) | \
                     ((o2.abs() > 0) & (o2.abs() < SUBNORMAL))
                m[t, hh * d:hh * d + h] = fh | sh
                m[t, hh * d + h:hh * d + d] = fh | sh
        masks.append(m)
    return masks[0], masks[1]


CASES = [
    ("decode T=1   Hq=16 Hkv=2 D=128 dense", 1, 16, 2, 128, False, 21, 1.0),
    ("prefill T=7  Hq=16 Hkv=2 D=128 dense", 7, 16, 2, 128, False, 22, 1.0),
    ("prefill T=512 Hq=16 Hkv=2 D=128 dense", 512, 16, 2, 128, False, 23, 1.0),
    ("decode T=1   Hq=16 Hkv=2 D=128 padded-fused-QKV", 1, 16, 2, 128, True, 24, 1.0),
    ("prefill T=7  Hq=16 Hkv=2 D=128 padded-fused-QKV", 7, 16, 2, 128, True, 25, 1.0),
    ("generic T=5  Hq=8  Hkv=2 D=64", 5, 8, 2, 64, False, 26, 1.0),
    ("generic T=3  Hq=4  Hkv=4 D=128 MHA", 3, 4, 4, 128, False, 27, 1.0),
]


def check_export_contract(fail):
    try:
        import tecoops
    except Exception as exc:  # pragma: no cover
        record(fail, gate="export", ok=False, why="import tecoops failed: %r" % (exc,))
        return False
    ok = hasattr(tecoops, "rotary_embedding") and "rotary_embedding" in tecoops.__all__
    record(fail, gate="export", ok=ok, has_attr=hasattr(tecoops, "rotary_embedding"),
           in_all=("rotary_embedding" in tecoops.__all__), n_exports=len(tecoops.__all__))
    return ok


def check_cpu_gate(fail):
    all_ok = True
    for tag, t, hq, hkv, d, padded, seed, scale in CASES:
        qkv, cache, positions = make_case(t, hq, hkv, d, seed, padded, scale)
        q, k = slices(qkv, hq, hkv, d)
        ref_q, ref_k = reference_chain(q, k, cache, positions, hq, hkv, d)
        mod_q, mod_k = kernel_model(q, k, cache, positions, hq, hkv, d)
        this = True
        for name, got, ref in (("q", mod_q, ref_q), ("k", mod_k, ref_k)):
            md = float((got.float() - ref.float()).abs().max())
            mm = int((got.view(torch.int16) != ref.view(torch.int16)).sum())
            this = this and mm == 0 and md == 0.0
            record(fail, gate="cpu", case=tag, tensor=name, max_error="%.6e" % md,
                   bit_mismatch=mm, total=ref.numel(), ok=(mm == 0))
        all_ok = all_ok and this
    return all_ok


def device_available():
    try:
        import torch_sdaa  # noqa: F401  - registers the SDAA device
    except Exception as exc:
        return False, "import torch_sdaa failed: %r" % (exc,)
    try:
        return bool(torch.sdaa.is_available()), "torch.sdaa.is_available()"
    except Exception as exc:
        return False, "torch.sdaa.is_available() raised %r" % (exc,)


def device_inputs(qkv, hq, hkv, d, padded):
    if padded:
        # Moving the slices separately would silently make them contiguous.
        qkv_d = qkv.to("sdaa")
        q_d, k_d = slices(qkv_d, hq, hkv, d)
        assert q_d.stride() == k_d.stride() == ((hq + 2 * hkv) * d, 1)
        return q_d, k_d, qkv_d
    q, k = slices(qkv, hq, hkv, d)
    return q.to("sdaa"), k.to("sdaa"), None


def check_device_output(fail, gate, case, name, got, ref, mask):
    neq = got.view(torch.int16) != ref.view(torch.int16)
    bad = int((neq & ~mask).sum())
    masked = int((neq & mask).sum())
    md = float((got.float() - ref.float()).abs().max())
    record(fail, gate=gate, case=case, tensor=name, bit_mismatch=int(neq.sum()),
           bit_mismatch_unmasked=bad, bit_mismatch_in_denormal_mask=masked,
           max_error="%.6e" % md, ok=(bad == 0))
    return bad == 0


def check_device_gate(fail):
    import tecoops
    all_ok = True
    for tag, t, hq, hkv, d, padded, seed, scale in CASES:
        qkv, cache, positions = make_case(t, hq, hkv, d, seed, padded, scale)
        q, k = slices(qkv, hq, hkv, d)
        ref_q, ref_k = reference_chain(q, k, cache, positions, hq, hkv, d)
        mask_q, mask_k = denormal_involved(q, k, cache, positions, hq, hkv, d)

        q_d, k_d, _ = device_inputs(qkv, hq, hkv, d, padded)
        cache_d = cache.to("sdaa")
        pos_d = positions.to("sdaa")
        out_q = torch.empty((t, hq * d), dtype=HALF, device="sdaa")
        out_k = torch.empty((t, hkv * d), dtype=HALF, device="sdaa")
        tecoops.rotary_embedding(q_d, k_d, cache_d, pos_d, out_q, out_k, True)
        torch.sdaa.synchronize()

        this = True
        for name, got_d, ref, mask in (("q", out_q, ref_q, mask_q),
                                       ("k", out_k, ref_k, mask_k)):
            good = check_device_output(fail, "device", tag, name, got_d.cpu(), ref, mask)
            this = this and good
        all_ok = all_ok and this
    return all_ok


def check_inplace_and_rejection(fail):
    import tecoops
    ok = True
    # Keep the original failing seed, and cover all device geometries in-place.
    inplace_cases = CASES + [
        ("CI seed=31 T=7 Hq=16 Hkv=2 D=128 padded-fused-QKV",
         7, 16, 2, 128, True, 31, 1.0),
    ]
    for tag, t, hq, hkv, d, padded, seed, scale in inplace_cases:
        qkv, cache, positions = make_case(t, hq, hkv, d, seed, padded, scale)
        q, k = slices(qkv, hq, hkv, d)
        ref_q, ref_k = reference_chain(q, k, cache, positions, hq, hkv, d)
        mask_q, mask_k = denormal_involved(q, k, cache, positions, hq, hkv, d)
        q_d, k_d, qkv_d = device_inputs(qkv, hq, hkv, d, padded)
        cache_d, pos_d = cache.to("sdaa"), positions.to("sdaa")
        out_q = torch.empty((t, hq * d), dtype=HALF, device="sdaa")
        out_k = torch.empty((t, hkv * d), dtype=HALF, device="sdaa")
        # The inputs are identical for both calls; the first must not mutate them.
        tecoops.rotary_embedding(q_d, k_d, cache_d, pos_d, out_q, out_k, True)
        torch.sdaa.synchronize()
        for name, got_d, original in (("q", q_d, q), ("k", k_d, k)):
            bad = int((got_d.cpu().view(torch.int16) != original.view(torch.int16)).sum())
            record(fail, gate="outplace_input", case=tag, tensor=name,
                   bit_mismatch=bad, ok=(bad == 0))
            ok = ok and bad == 0

        tecoops.rotary_embedding(q_d, k_d, cache_d, pos_d, q_d, k_d, True)
        torch.sdaa.synchronize()
        for name, got_d, out_d, ref, mask in (("q", q_d, out_q, ref_q, mask_q),
                                             ("k", k_d, out_k, ref_k, mask_k)):
            got, out = got_d.cpu(), out_d.cpu()
            bad = int((got.view(torch.int16) != out.view(torch.int16)).sum())
            # No subnormal mask here: aliases must match device output exactly.
            record(fail, gate="inplace", case=tag, tensor=name,
                   row_stride=got_d.stride(0), bit_mismatch=bad, ok=(bad == 0))
            ok = ok and bad == 0
            good = check_device_output(fail, "inplace_cpu", tag, name, got, ref, mask)
            ok = ok and good
            good = check_device_output(fail, "outplace_cpu", tag, name, out, ref, mask)
            ok = ok and good
        if qkv_d is not None:
            v = qkv_d.cpu()[:, (hq + hkv) * d:]
            original_v = qkv[:, (hq + hkv) * d:]
            bad = int((v.view(torch.int16) != original_v.view(torch.int16)).sum())
            record(fail, gate="inplace_v_untouched", case=tag,
                   bit_mismatch=bad, ok=(bad == 0))
            ok = ok and bad == 0

    # contract rejections: the wrapper must fail closed (never silently rotate) AND
    # must not have written the output buffers before raising.  The outputs are
    # pre-filled with a sentinel so a partially-written buffer is detectable; this is
    # the same sentinel contract as the #38/#40 rejection assertions.
    hq, hkv, d = 16, 2, 128
    sentinel = 7.0
    q = torch.zeros((2, hq * d), dtype=HALF, device="sdaa")
    k = torch.zeros((2, hkv * d), dtype=HALF, device="sdaa")
    cache_d = cache.to("sdaa")
    pos_d = positions[:2].to("sdaa")
    for name, kwargs, want in (
        ("non-neox", dict(is_neox=False), "NeoX"),
        ("fp32 q", dict(is_neox=True, q=torch.zeros((2, hq * d), device="sdaa")), "fp16"),
    ):
        out_q = torch.full((2, hq * d), sentinel, dtype=HALF, device="sdaa")
        out_k = torch.full((2, hkv * d), sentinel, dtype=HALF, device="sdaa")
        try:
            tecoops.rotary_embedding(kwargs.get("q", q), k, cache_d, pos_d, out_q, out_k,
                                     kwargs["is_neox"])
            record(fail, gate="reject", case=name, ok=False, why="no exception raised")
            ok = False
            continue
        except Exception as exc:
            good = want in str(exc)
            record(fail, gate="reject", case=name, ok=good, exc=str(exc)[:120])
            ok = ok and good
        # The rejection must happen before any store: a non-zero count means the
        # caller would receive a half-written tensor after a failed call.
        torch.sdaa.synchronize()
        changed_q = int((out_q.cpu() != sentinel).sum())
        changed_k = int((out_k.cpu() != sentinel).sum())
        untouched = changed_q == 0 and changed_k == 0
        record(fail, gate="reject_untouched", case=name, sentinel=sentinel,
               changed_q=changed_q, changed_k=changed_k, ok=untouched)
        ok = ok and untouched
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-only", action="store_true",
                        help="skip every device step; self-check for the CPU gate only")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    fail = []
    results = {"gate": {}}
    print("=" * 78)
    print("tecoops.rotary_embedding conformance")
    print("=" * 78)
    results["gate"]["export"] = check_export_contract(fail)
    results["gate"]["cpu_bit_exact"] = check_cpu_gate(fail)

    if args.cpu_only:
        # --cpu-only must be strictly device-free: no torch_sdaa import at all.
        assert "torch_sdaa" not in sys.modules, "cpu-only mode must not import torch_sdaa"
        print("[ATTEST] cpu-only: 'torch_sdaa' not in sys.modules")
        dev_ok, dev_msg = False, "--cpu-only"
    else:
        dev_ok, dev_msg = device_available()
    print("device_available: %s (%s)" % (dev_ok, dev_msg))
    results["device_available"] = [dev_ok, dev_msg]
    if args.cpu_only or not dev_ok:
        reason = "--cpu-only" if args.cpu_only else dev_msg
        print("SKIP device gates: %s" % reason)
        results["gate"]["device"] = None
        results["gate"]["inplace_and_rejection"] = None
    else:
        results["gate"]["device"] = check_device_gate(fail)
        results["gate"]["inplace_and_rejection"] = check_inplace_and_rejection(fail)

    if args.output_json:
        with open(args.output_json, "w") as fh:
            json.dump({"gates": results, "failures": fail}, fh, indent=1)

    print("=" * 78)
    if fail:
        print("FAILURES: %d" % len(fail))
        for f in fail:
            print("  %s" % (f,))
        print("RESULT: FAILED")
        return 1
    print("RESULT: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
