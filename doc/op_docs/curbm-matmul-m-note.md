# D512/BM32 GEMM M scoped to `cur_BM` — provenance note

- Date: 2026-10-10 (server date, `date -u`)
- Branch: `op/accepted-all`
- Base commit: `0e74500ddb41816624e63875cd1787b1c5393acf`
  (`flash_attention.scpp` blob `d542b9a33c031320c3202c5ba1a96f04fdc24bc1`)
- Kernel: `teco/ual/kernel/flash_attention/flash_attention.scpp`,
  `flash_attention_half<32>` / `teco_slave_flash_attention_half_d512`
- Change: 2 hunks, +12 / -4 lines, one file, no other file touched
- Applied patch: `candidate.patch`, sha256
  `b430248f9e3ba142d79d78c82f4269fbdffa1672316977519a292300c05688f8`
- Resulting blob: `ddcaae3d5a86f26b7b4c1f36aa70d3b754fe07c6`
  (sha256 of the patched file `44b54880d20997fe6b7113b039f80f94d023abb786e1f80ceceacfa035d11e39`,
  identical to the value the exporting workstream recorded for the same patch applied to
  `op/flash-attention-d256-d512-tile` at `795bfb8`)

## Mechanism

In the isolated D512 specialization `flash_attention_half<32>` both GEMMs used the compile-time
tile height `BM` (= 32) as their **M** dimension, while only `cur_BM` query rows of the tile are
ever observed. At decode `cur_BM = 1`, so ~31/32 of both GEMMs is work whose output rows are
discarded.

```
QK^T : tensorQ {BM, BK}      -> {gemm_m,  BK}      ; tensorC {BM, BN}       -> {gemm_m,  BN}
P@V  : tP     {BM, BN}       -> {gemm_m2, BN}      ; tO     {BM, gemm_bn}   -> {gemm_m2, gemm_bn}
        gemm_m / gemm_m2 = (BM == 32 ? cur_BM : BM)   // if constexpr
```

Two statements (`int gemm_m = BM;` / `int gemm_m2 = BM;`) each followed by
`if constexpr (BM == 32) { gemm_m = cur_BM; }` bind the M dimension at compile time. There is
**no runtime branch** and **no `getenv`** in the hot path, and no backend/fallback selection is
added. The GEMM block config (`M128_N32`), K/V DMA, the `ktmp` -> `kbuf` transpose, V tiling,
softmax / Step A/B/E, the clearing policy, the cache layout and the ABI are untouched, and the
`BM == 64` (D256) / `BM == 128` instantiations of the same template are byte-identical copies of
their previous text.

A GEMM's C rows are independent (row *i* of `C = A @ B` depends only on row *i* of `A`), and every
observed consumer (Step A/B/E, `normalize`, the output DMA) is already `cur_BM`-scoped and reads
only rows `< cur_BM`. Narrowing M therefore cannot change any observed value, so the original
`0.02` FP32 reference threshold is **kept, not relaxed**.

## Accepted evidence — carried over from the private accelerator run

Source of the measurement: private accelerator repo
`/root/SichuanUniversity_AI_accelerator-gemma`, branch `model/gemma-4-12b-it`, task dir
`op_learning/attention/gemma-d512-curbm-matmul-m-20261009/` (`STATUS.md` — "gates COMPLETE,
candidate ACCEPTED"; `stage_results/gate-result.json`; `learned_patterns.md`;
`official-pr-export.md` — "local export only — NOT pushed, no PR opened").

`gate-result.json` `status = "accept"`; all of the following raw values are the author's, not
re-measured here:

- candidate build: core `5d7c02c5…`, source `fee238d4…`, kernel object `037502b1…`,
  patch `b430248f…`; the extension was **not** recompiled (ABI unchanged).
- focused 80 rows (20 shapes x poison x default/nondefault, seed 20261007) vs the accepted
  baseline core `a0117935…`: **uint16 bitwise 0 differences**, cache/input bits exact,
  poison-tail and output-padding assertions pass.
- prefill 8 rows (nq 2/33/64/128 x 2 streams): **8/8 `baseline_bitdiff = 0`**,
  `cache_unchanged = true`, maxabs <= 0.0019.
- model gate TP2/FP16/ctx4352/chunk512: greedy-32 token IDs identical to the frozen baseline;
  **315.66 s / 65 steady requests / zero errors**, concurrency 1/4/8 pass; `performance_claim:
  false`.
- paired A1 -> B -> A2 timing (drift control, all `bitdiff = 0`, A-group drift <= 0.227 %):
  n1/kv4352 **-4.2423 %**, n4 mixed **-4.1189 %**, n8 mixed **-4.3994 %**.
- **No whole-model speedup is claimed.** The model gate is a correctness gate; the
  `-4.1…-4.4 %` figure is **operator-level only**.
- `n1/kv33` is **not resolvable** in that environment: the focused-session `+12.8896 %` is
  retracted as measurement contamination (the paired value is `-0.2466 %` while the two A
  replicates drift `+1.0890 %` on their own).

## Verification performed on THIS organizer copy

- `git apply` of `candidate.patch` succeeded cleanly on the first attempt (`git apply -v`: "Applied
  patch ... cleanly"); no `-C1` and no `patch --fuzz` was needed. `git diff --stat` = one file,
  **+12 / -4** (the 2 expected hunks; the "one file" and hunk count match the export, the line
  count is 12/4 rather than the 8/4 estimated before the apply).
- `git diff --check` — clean.
- `git diff --name-only 0e74500..HEAD` lists exactly the one kernel file plus this note; every
  non-blank line the patch adds is present and every line it removes is gone (12/12 present,
  4/4 removed); diffing `0e74500` against itself gives 0 lines (self-control).
- Forbidden-set greps on the patched file vs the base file: `M128_N32` 2 -> 2,
  `floatv16` 11 -> 11, `getenv` 0 -> 0, `BM == 64` 0 -> 0, `BM == 128` 0 -> 0,
  `BM == 32` 4 -> 6 (the two new `if constexpr` guards). No changed line mentions
  `ktmp`, `kbuf`, `vbuf`, `softmax`, `clear`, `scale` or `normalize`; the whole diff is the two
  `make_tensor<2>` shape changes plus the two new compile-time guards.
- CPU-only compile of this commit with the SDAA toolchain
  (`source /opt/tecoai/setvars.sh`; `WITH_TORCH=ON WITH_INFERENCE_PLUGIN=OFF
  /home/py312/bin/python setup.py build_ext --inplace`): **rc = 0**,
  2 warning lines, both the environmental `torch_sdaa and extension build with different
  sdaa_runtime version` `UserWarning`.
  `libteco_ops.so` sha256 `1fdcbee609c1f319908dbab7a4b5c3debf1754b91390316b82745682f4627cf6`.
- Control build of the parent `0e74500` in the same directory, same command: **rc = 0**,
  `libteco_ops.so` sha256
  `3b7a8b436f138f1cfd1d0bdf31f9681876cbed8cae0ced9d215de8cf51909d5a`. The two shas **differ**, so
  this commit really does change the compiled library. A repeat build of this commit reproduced
  `1fdcbee6…` bit-for-bit.

## Scope / honesty statement

**No device validation was performed on this organizer copy.** No accelerator card was used, no
SDAA kernel was launched, no model was run, and the focused / prefill / model / ABA gates were
**not** re-executed here. The correctness and performance numbers above are cited from the private
accelerator run on the task directory named above; they are carried over, not reproduced. The only
gates executed on this copy are the source-level and CPU-only ones listed immediately above, plus
the fast-forward push of this branch. Re-running the device gates on this organizer branch remains
open work, as does the upstream/organizer CI, which has not seen this head.
