# ALGO0 `BM == 128` Step B/E SIMD widen — provenance (2026-10-10)

## What changed

`teco/ual/kernel/flash_attention/flash_attention.scpp`, **two lines only**:

| Line | Before | After |
| ---: | --- | --- |
| `:305` (Step B, `o_accum` rescale) | `if constexpr (BM == 32) {` | `if constexpr (BM == 32 \|\| BM == 128) {` |
| `:388` (Step E, `o_accum` accumulate) | `if constexpr (BM == 32) {` | `if constexpr (BM == 32 \|\| BM == 128) {` |

The 16-wide `floatv16` bodies (`:306-313`, `:389-397`) are kept **verbatim**.
`BM == 64` (D256) and `BM == 32` (D512) are byte-identical in behaviour.
No run-time branch, no `getenv`, no new SPM allocation (`rt_spm_malloc` 13 → 13).

## Why it is bit-exact

Both loops are **element-wise** fp32 operations — Step B is `o_accum[ri][rk] *= s`
and Step E is `o_accum[ai][hb*32+ak] += outO[ai][ak]` — one rounding per element and
**no reassociation**. Blocking the same operations 16-wide cannot change any lane's
result; only the number of elements moved per instruction changes. `softmax_scale`,
the causal mask, the online-softmax state, every float→half conversion, the
`Matmul`/`M128_N32` configuration and the `l_block` summation order are untouched.

## Models served (one change, two workstreams)

* **InternVL3_5-8B** (priority 2) Rank-1 — ALGO0 / head_dim 128 / real prefill shape `seq=1811`.
* **MiniCPM5-1B** (priority 5) M1 — head_dim 128, the same `flash_attention_half<128>` instantiation.

## Verified — CPU (no card)

`r0b_biteq_model.py`, numpy 1.26.4, **21/21 checks PASS**:

* 480 explicit per-lane cases: scalar form vs 16-blocked vector form vs numpy form;
  `BK ∈ {64,128}`, `cur_BM ∈ {1,7,128}`; values random / denormal / signed-zero /
  1e37 / mixed-magnitude `10^±30`; `s ∈ {0, 1, −0, 0.5, 2, 1e-8, 0.9999999, 1.0000001}`.
* 300 randomised fuzz trials, 1 872 768 element-ops.
* 13 full-geometry end-to-end runs (tile loop + causal mask + online-softmax recurrence
  + Step A/B/C/D/E + normalize→half), **3 534 tile pairs**, covering InternVL `seq=1811`
  (855), `case_5` `kv=35845` (1121) and MiniCPM `q=kv=7` / `q=1` / `q=128` / `q=129` /
  `kv=33` / `q=1,kv=4352` / `q=1024,kv=4352`:
  `acc` diff_words32 = **0**, `obuf` diff_halfwords16 = **0**, maxabs = **0.0**.
* **Negative controls differ** (fp16 intermediate: 512 words; fp64 deferred
  accumulation: 157 339 words) — the equality is not vacuous.

Alignment: every Step-B / Step-E / normalize byte offset is 64-byte aligned for
`BK ∈ {64,128}`; the Step-B offset set is **identical** to the normalize loop's,
which already runs unconditionally at `BM == 128` (`:409-417`).

## Verified — device (single card, no contention)

Per arm: `teco-smi` `No Process Running` + `0MB / 65536MB` + empty
`ps | grep -Ei 'vllm|EngineCore|demo|unit_test'`, three consecutive checks 60 s apart,
plus a re-check immediately before every launch.

* **R2 (decisive)** — baseline `d6f5fee8…` vs candidate `c68af2c8…` on identical
  prototxt inputs, compared as
  `bitdiff = int(np.count_nonzero(cand.view(np.uint16) != base.view(np.uint16)))`:
  **12/12 `oData` dumps, bitdiff = 0, 11 993 600 halfwords total.**
* **R1** — focused `demo` (`cd test && source env.sh`):
  **22/24 PASSED**. The only failure is `case_0`, and it fails **identically on both
  arms** (`max_error = 1.21774e+03` in baseline *and* candidate) → a pre-existing
  test-case/reference issue, not caused by this change. Every other case's
  `max_error` is identical between the two arms (1.16610e-04 … 1.89809e-04;
  `case_9` = 9.69876e-07; `case_11` real shape `seq=1811` = 1.89809e-04).

**No performance measurement was made** (R3/R4/R5 not authorised); the A/B arms ran
with `--perf_repeat=1` inside one clean window and constitute no timing evidence.

## Boundary and open items

* **Operator-level only.** `tecoops.flash_attn_varlen_func` has 0 calls in the real
  forward of both models (the delivered path uses SDPA), so **no end-to-end or
  model-level gain is claimed**.
* The widen adds an implicit `BK % 16 == 0` precondition to the `BM == 128` path
  (both former scalar loops now step `o_accum` in 16-float `floatv16` strides).
  It is implied today by the baseline's own `for (hb = 0; hb < BK / gemm_bn; hb++)`
  (`:348`), which already requires `BK % 32 == 0` for every head dim to be accumulated
  at all — so no shape on which the baseline is correct can overrun.

## Dispatch precondition, enforced fail-closed (`find_flash_attention.cpp`)

`BK` is **not** a compile-time value (`int BK = args.size_per_head;` in the kernel), so
the precondition above cannot be written as a `static_assert`; it is bound instead in the
existing outer dispatch `findFlashAttentionBranch()` (one `head_dim % 16 != 0` check, once
per operator dispatch — no kernel hot-path branch, no `getenv`). Returning `-1` reuses the
file's existing convention: `FlashAttentionOp::findImpl()` maps it to
`Status::NOT_IMPLEMENTED`, so such a shape now fails closed instead of silently dropping
`head_dim` tail columns. Head dims 64 / 128 / 256 / 512 — every shape under test — bypass
the guard unchanged.

* **Out of scope here (separate follow-up):** the baseline's `BK / gemm_bn` integer
  truncation (`:348`) also silently drops `head_dim % 32` tail columns on *every* branch.
  That defect predates this change and is deliberately **not** folded into this commit.
