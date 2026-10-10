# D512/BM32 cur_BM-scoped post-processing loops — provenance note

- Date: 2026-10-09 (server date, `date -u`)
- Branch: `op/flash-attention-d256-d512-tile`
- Base commit: `3d6793a2423cfc465dc88cc7489f9d5fb6c065ab`
  (`flash_attention.scpp` blob `684ed0d00f32caf2c8800df6d1b9ccc0f2407198`)
- Kernel: `teco/ual/kernel/flash_attention/flash_attention.scpp`,
  `flash_attention_half<32>` / `teco_slave_flash_attention_half_d512`
- Change: 2 hunks, +9 / -0 lines, one file, no other file touched

## Mechanism

The D512 specialization did two full-tile post-processing passes per KV tile, both bounded by
`int total = BM * BN;` (`BM*BN = 1024` elements): the `softmax_scale` multiply pass and the Step C
`float -> half` conversion pass. Only rows `[0, cur_BM)` of the score tile are ever read back —
`apply_causal_mask`, `online_softmax_step`, the Step B rescale, the Step E SV-add, the normalize
step and the output DMA are all `cur_BM`-scoped. At decode `cur_BM = 1`, so 31 of 32 row-groups of
that vector work are pure redundancy.

This commit narrows both loop bounds to `cur_BM * BN`, guarded by `if constexpr (BM == 32)` so the
narrowing is folded at compile time: **no runtime branch and no `getenv`** in the hot path, and no
backend/fallback selection added. Rows `>= cur_BM` that are no longer scaled/converted have exactly
one consumer — the `P @ V` output rows that are themselves discarded — so every observed output is
**bit-identical**.

The `BM == 64` (D256) and `BM == 128` instantiations of the same template are untouched: the added
statements are discarded for them at compile time.

## Accepted evidence — carried over from the private accelerator run

Source of the measurement: private accelerator repo `/root/SichuanUniversity_AI_accelerator-gemma`,
commit `c195874` "perf(gemma): scope D512 BM32 post-processing loops to cur_BM rows", evidence dir
`op_learning/attention/gemma-d512-curbm-loops-20261009/` (`learning_task.md`, `candidate.patch`,
`stage_results/gate-result.json`).

- `decision`: `accept measured operator-level gain`
- focused gate: 80 cases, `focused.output_bitwise_equal = true`
- prefill control: 8 rows, `prefill.all_bitwise_equal = true`, cache bitwise unchanged
- medians of 3 runs (same seed/warmup 5/calls 10), candidate vs accepted baseline:
  N1/KV33 **-1.81 %**, N1/KV4352 **-0.67 %**, N4 **-0.70 %**, N8 **-0.60 %**
  (raw ms: 0.244820 -> 0.240392; 9.846237 -> 9.780570; 12.602612 -> 12.514548; 41.068665 -> 40.820485)
- accepted patch sha256: `aba5b61c17d3222a90c2d820fa2957d35f43c9da172e9b3f5e81c3d6a8c2e0ab`
- candidate core sha256 `a011793561d323dcede5c1a4d947054d2151ab37a06c1568f39fee50086dd313`
  vs baseline core sha256 `d61208eeba2362714f9aab67d74d8f3dd9e57121991c90f0b5a17f62ee0fe4ef`
- on that run the model gate also matched: TP2/FP16/ctx4352 fixed-prompt greedy-32 token IDs equal
  to the frozen baseline, ~313 s steady state; model-level `performance_claim = false`.

## Verification performed on THIS organizer copy

- `git diff` is exactly the two hunks above; `git apply --check -R` of the accepted
  `candidate.patch` succeeds, i.e. this tree is equivalent to base + the accepted patch.
- `git diff --check` clean.
- CPU-only compile with the SDAA toolchain, `source /opt/tecoai/setvars.sh`:
  `BUILD_RC=0`, **0 warnings** (`-Werror` is active in the build).
  Artifacts: `libteco_ops.so` sha256
  `d847f33eb746d249a282dd8be36afc3e32a8fb561372b12fb269fa5850981d30`,
  `flash_attention.scpp.o` sha256
  `fce80c7e6a8853935dfeaf536d9f972a456846a680dec574c0b51b569407e0b5`.
- Codegen invariance: the same compile line was run over the base source and over this commit and
  the emitted IR was diffed per symbol. `teco_slave_flash_attention_half` (BM128) and
  `teco_slave_flash_attention_half_d256` (BM64) — plus every other function and both device stubs —
  are byte-identical; only `teco_slave_flash_attention_half_d512` (BM32) differs.

## Scope / honesty statement

**No device validation was performed on this organizer copy.** No accelerator card was used, no SDAA
kernel was launched, no model was run, and the focused / prefill / model gates were **not**
re-executed here. The correctness and performance numbers above are cited from the private
accelerator run at `c195874`; they are carried over, not reproduced. The only gates executed on this
copy are the source-level and CPU-only ones listed immediately above. Re-running the device gates on
this organizer branch remains open work.
