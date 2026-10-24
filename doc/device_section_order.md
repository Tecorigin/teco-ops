# Device-section order: measurement recipe

This document records the byte-level A/B measurement procedure used to calibrate and verify the
device-section-order pin in `teco/CMakeLists.txt` (option `TECO_ENABLE_DEVICE_SECTION_ORDER`,
default `ON`). It involves no new build machinery: it reuses the standard build and the demo binary
that the repository already produces.

## Background

The order of the device-code sections inside the `.sdaa_fatbin` embedded in `libteco_ops.so`
equals the order in which the object files appear on the official link line
(`tecocc $<TARGET_OBJECTS:ops_objs> -flto ... --sdaa-link`), which in turn equals the order of the
`SRC_UAL_KERNEL` source list in `teco/CMakeLists.txt`. Permuting only the object order therefore
moves the device-section VMAs; per-tile instruction-fetch behaviour of the SDAA device caches
depends on the absolute VMA a kernel section lands on, so a few tens of bytes of section placement
can move a µs-scale kernel by ±50%.

## What the fingerprint covers (and what it does not)

The configure-time fingerprint (`teco-ops device-section order: ... fingerprint=<sha1> (count=<n>)`)
hashes only the **canonical kernel TU order** — the relative `<subdir>/<file>.scpp` list built from
`SRC_UAL_KERNEL`. It deliberately contains no checkout path, no absolute path and no build flag.

Consequences:

- The fingerprint travels with the **source order**, not with the build directory: the same commit
  hashes identically on every machine and in every checkout (that is what makes it usable as a
  recorded expected value).
- It does **not** model the environment in which the recorded order was measured. Every address in
  this document was measured on a specific `(device model, SDAA driver version, TecoDNN/runtime
  version)` triple; a different triple may map a different order into the fast slots. The
  measurement environment is therefore a **precondition of reusing a recorded order**, e.g.:

  ```text
  recorded order : <sha1 from the accepted layout>
  measured on     : SDAA 3.2.0, driver 3.2.0, TecoDNN 3.2.0, device <model>
  ```

  Record that triple next to the accepted fingerprint (PR body / receipt). A build on a different
  device or runtime must treat the pin as unverified and re-measure, even though the fingerprint
  still matches — a matching fingerprint means "same TU order", never "same device layout".

## Build two trees (A/B)

1. Prepare two full checkouts of the same commit, differing **only** by the device-section order
   setting, e.g.:

   ```bash
   git clone <the repo> tree-A && cd tree-A && git checkout <rev>
   git clone <the repo> tree-B && cd tree-B && git checkout <rev>
   # tree-B: change the one list move / option under test in teco/CMakeLists.txt
   ```

   Nothing else may differ: same kernel TU set (no added/removed modules), same toolchain, same
   compile options. Any kernel TU add/remove invalidates every recorded address (see
   `teco/CMakeLists.txt` fingerprint WARNING).

2. Configure + build both trees with identical arguments, e.g.:

   ```bash
   # in tree-A and tree-B alike:
   source /opt/tecoai/setvars.sh
   bash ./build.sh --build teco
   ```

   Optionally record the section-order fingerprint printed at configure time
   (`teco-ops device-section order: ... fingerprint=<sha1> (count=<n>)`) for both trees to prove
   the two arms actually differ in the intended way.

## Measurement (byte-level, official fixed form)

3. Keep the SDAA environment identical across arms (same device, same clocks, run A and B
   back-to-back inside the same window — cross-window comparisons of µs-level kernels are noise
   dominated).

4. Run the official fixed-form benchmark (do **not** modify the demo or the case files):

   ```bash
   ./build/demo --gid=0 --perf_repeat=50 --warm_repeat=3 --gtest_repeat=1 \
       --case_path=zoo/teco/<op>/test_case/<case>.prototxt | grep "Average Hardware Time"
   ```

5. Repetition protocol: for every case run **3 reps per tree**, within the same window, and score
   the **median** of the 3 `Average Hardware Time` values. Report all raw values, not only the
   median.

6. Correctness gate: the same run's correctness comparison must stay at the existing tolerance
   (`max_error` bit-for-bit unchanged between arms for a pure order move — an order-only change
   must never alter numerics).

## Device-layout evidence

7. To see *where* a kernel was placed, hash/compare the segment table instead of trusting the
   timing alone: extract the device section headers of the two `.sdaa_fatbin` sections of
   `build/lib/libteco_ops.so` (e.g. dump section names, addresses, sizes with the SDAA SDK's
   `readelf`-equivalent on the fatbin) and compare:
   - the 34–38 device section records `(name, addr, size)` as the primary criterion — the *only*
     byte-level difference between two pure-order arms is the section order/addresses (plus
     compile-time-embedded source-path strings, which are path noise);
   - a SHA-1 over the sorted-then-zipped `(name,addr,size)` list ("segment table hash") as a single
     reproducible number for the receipt.

   Do **not** use the absolute path embedded in `.rodata`/`.comment` for any decision: it depends
   on the checkout directory and is not comparable across machines (the configure-time fingerprint
   in `teco/CMakeLists.txt` is deliberately hashed from relative file names for the same reason).

## Strict mode (opt-in CI gate)

A fingerprint mismatch prints a `WARNING` and the build continues by default. That is right for a
developer tree (a stale recorded value must not stop work) but wrong for CI: once any PR adds or
removes a kernel TU, the recorded order no longer applies, the pin silently stops matching, and a
layout regression can land unnoticed.

CI therefore turns the mismatch into a configure failure with an explicit opt-in switch:

```bash
cmake -DTECO_ENABLE_DEVICE_SECTION_ORDER=ON \
      -DTECO_OPS_DEVICE_SECTION_ORDER_STRICT=ON \
      -S teco -B build/teco
```

| switch | mismatch behaviour | intended use |
| --- | --- | --- |
| `TECO_OPS_DEVICE_SECTION_ORDER_STRICT=OFF` (default) | `WARNING`, configure continues | developer builds |
| `TECO_OPS_DEVICE_SECTION_ORDER_STRICT=ON` | `FATAL_ERROR`, configure stops | CI / release gate |

Exemption channel (same in both modes): pass an empty expected value
`-DTECO_OPS_DEVICE_SECTION_ORDER_EXPECTED_FP=` to disable the comparison for one configure. This
keeps "re-measure on purpose" a conscious, greppable action instead of an ignored warning.

Rationale for default OFF: turning the default into a hard failure would break every downstream
build of a tree whose pin is intentionally stale while a measurement window is being scheduled.
The switch is the compromise between "never silently lose the pin" (CI) and "never block an
unrelated build" (developers).

## Judging

8. Verdict per move: the target op's cases must improve (median-of-3, both arms the same window);
   the full zoo case table must show no unintended displacement regression beyond the noise band;
   all cases must stay `PASSED`. Record both arms' raw medians, the segment-table hash of the
   winner, and the fingerprint value of the winning source order.

9. Refresh the recorded expected fingerprint: after a re-measure that accepts a new order, copy the
   fingerprint line from the newly accepted tree's configure step into
   `TECO_OPS_DEVICE_SECTION_ORDER_EXPECTED_FP` in `teco/CMakeLists.txt` (or pass it via
   `-DTECO_OPS_DEVICE_SECTION_ORDER_EXPECTED_FP=<sha>`), together with the measurement environment
   triple above. A future configure whose order no longer matches emits a `WARNING` by default
   (build continues) and a `FATAL_ERROR` under `-DTECO_OPS_DEVICE_SECTION_ORDER_STRICT=ON`,
   so drift is never silent and CI can gate on it.
## Pinned relative order (what `TECO_ENABLE_DEVICE_SECTION_ORDER=ON` enforces)

The pin is a sequence of existence-guarded `list(FIND)` / `list(REMOVE_AT)` / `list(INSERT)` moves,
so it is idempotent and degrades to a no-op when an anchor TU is absent (e.g. on a tree where the
owning PR has not landed). The **relations** below are the contract; the absolute positions follow
from the rest of the glob order and are therefore allowed to shift with unrelated TU additions:

| Relation (left is linked before right) | Why it is pinned |
| --- | --- |
| `zz_ms_deform_attn_backward_ha` → `weight_stream_gemv` | R17 one-TU swap that lands WSG in its measured fast class |
| `weight_stream_gemv` → `rms_norm_compat` | A3: the arrangement that also puts `rotary_embedding` into its measured slot |
| `z_reshape_and_cache_dma` → `zzz_flash_attention_d128` | packs the contributor TUs behind WSG |
| `zzz_flash_attention_d128` → `z_ms_deform_attn_forward_span` | same |

### How to assert it

1. Configure with `-DTECO_ENABLE_DEVICE_SECTION_ORDER=ON` and read the printed fingerprint line
   (`teco-ops device-section order: ... fingerprint=<sha1> (count=<n>)`); a changed count means a
   TU was added/removed and every recorded address must be re-measured.
2. Run the configure step a second time — the moves are idempotent, so the fingerprint must not change.
3. For a hard gate in CI, configure with `-DTECO_OPS_DEVICE_SECTION_ORDER_STRICT=ON` plus
   `-DTECO_OPS_DEVICE_SECTION_ORDER_EXPECTED_FP=<sha1>`: a mismatch fails the configure step instead
   of being discovered from a device-side measurement.
4. Absence of an anchor (a TU whose PR has not landed) must produce a *no-op*, not a failure; that is
   why the moves are guarded by `GREATER -1` rather than requiring the anchor.
5. Configure-time assertion (`TECO_OPS_DEVICE_SECTION_ORDER_ASSERT`, default `ON`): when
   `TECO_ENABLE_DEVICE_SECTION_ORDER=ON`, CMake asserts that the anchor TU set
   (`weight_stream_gemv/weight_stream_gemv.scpp`, `rotary_embedding/rotary_embedding_fp16.scpp`,
   `zzz_flash_attention_d128/zzz_flash_attention_d128.scpp`) is present in `SRC_UAL_KERNEL` and
   preserves the calibrated relative order (`weight_stream_gemv` < `rotary_embedding` <
   `zzz_flash_attention_d128`). An absent anchor is reported as STATUS and skipped; an inverted
   relative position among the present anchors triggers `FATAL_ERROR`. A passing partial-tree
   assertion does not certify absent anchors. Exemption channel: pass
   `-DTECO_OPS_DEVICE_SECTION_ORDER_ASSERT=OFF` to exempt the relative-order assertion, or
   `-DTECO_ENABLE_DEVICE_SECTION_ORDER=OFF` to disable device-section ordering entirely.

### Merge checklist and fingerprint refresh

Land the kernel owners #35/#40/#41/#44/#45/#48 and the reduction changes #52 before
closing the merged-tree layout. In particular, rotary is #48 and RMSNorm compat is #45;
neither is a missing, unsubmitted operator. Keep this PR's independent fingerprint as its
own receipt; do not reuse it as the merged tree's expected value.

Every PR that adds or removes a kernel TU must refresh
`TECO_OPS_DEVICE_SECTION_ORDER_EXPECTED_FP` in that same PR after measuring the
complete candidate tree. A resulting STRICT configure failure belongs to that
kernel PR; it is not a CI infrastructure failure. Changes to an existing TU can
also move absolute addresses without changing this order fingerprint and still
require address and performance checks. The configure assertion additionally
rejects duplicate TU entries without changing the source list.

On the complete merged tree, build once with STRICT disabled to read the candidate order,
record the absolute device section `(name,address,size)` table, and run all original fixed
cases with correctness and three timing repetitions. Repeat any claimed layout benefit in
a second window. Only then set `TECO_OPS_DEVICE_SECTION_ORDER_EXPECTED_FP` to that measured
tree's fingerprint and reconfigure with STRICT enabled. A source-size change requires new
address measurements even if the TU-order fingerprint is unchanged. Do not add `list(SORT)`
or padding kernels to repair a mismatch.

## Propagating the pin to other operators

A section-order move is only justified by a *measurement*, never by analogy: the fast/slow classes
are address-specific and can differ by tens of percent between two addresses ~0x400 apart, while an
unrelated TU change can shift the whole device image and silently flip an operator's class. The
protocol that produced the relations above therefore has to be repeated for any new operator:

1. Build two artefacts from the **same source tree and same build root**, differing only in the
   link order (or only in the `.so`), and interleave the arms in one measurement window.
2. Use the official fixed form (`./build/demo --gid=0 --perf_repeat=50 --warm_repeat=3`) or the
   project's A/B harness with 3 repetitions, report the median, and repeat the window once more.
3. Accept a move only if the target case improves and the **whole case table** does not regress
   beyond noise (a single-case win is not enough; the layout tax is paid by every other operator).
4. Record the resulting relation here and re-calibrate the fingerprint; if the class cannot be
   reproduced in a second window, drop the move instead of keeping an unexplained 2–3 %.
5. A move for `reshape_and_cache` / `rms_norm` / `rotary_embedding` must be measured with that
   operator's own cases; the WSG numbers in this file are **not** transferable evidence.
## Device measurement: what one move actually buys (c50, 2026-10-22)

Measurement tree: a control tree that carries the `weight_stream_gemv` kernel TU and `rms_norm`
(the other pin anchors do not exist there), so exactly **one** move is enabled: `weight_stream_gemv.scpp`
immediately before `rms_norm_fp16.scpp`, using the same existence-guarded FIND/REMOVE_AT/INSERT idiom
as the delivered pin. Both arms are built from the **same build root**; only the kernel TU order differs,
and the configure fingerprints differ (`b61cee1b…` base vs `84b0a11e…` pin), which proves the move is real.

Command: `./build/demo --gid=0 --perf_repeat=50 --warm_repeat=3 --gtest_repeat=1`, 3 reps, median.
SCU SDAA single card, SDAADriver/SDAARuntime 3.2.0, identical `libteco_ops.so` except link order.

| case | base (ms) | pin (ms) | Δ |
| --- | --- | --- | --- |
| wsg case_0 (256×256) | 0.008731 | 0.003833 | **−56.10%** |
| wsg case_1 (512×512) | 0.011236 | 0.006713 | **−40.25%** |
| wsg case_2 (64×256) | 0.006041 | 0.002890 | **−52.17%** |
| wsg case_3 (192×48) | 0.007525 | 0.003180 | **−57.74%** |
| wsg case_4 (1024×2048) | 0.035139 | 0.032372 | −7.87% |
| wsg case_5 (128×2032) | 0.010024 | 0.006566 | −34.50% |
| wsg case_6 (3840×2048, real) | 0.082577 | 0.079278 | −4.00% |
| wsg case_7 (3840×4096, real) | 0.146140 | 0.143050 | −2.11% |
| wsg case_8 (4096×3840, real) | 0.145540 | 0.143050 | −1.71% |
| wsg case_9 (5120×3840, real) | 0.204740 | 0.203690 | −0.51% |
| **rms_norm case_0** | 0.122870 | 0.125160 | **+1.86%** |
| reshape_and_cache case_0 | 0.133670 | 0.133650 | −0.01% |
| reduce_variance case_0 | 0.058042 | 0.057788 | −0.44% |

All `max_error` values are bit-identical between the two arms.

### What this measurement says

1. **The pinned benefit is real but shape-dependent**: small/medium shapes gain 35–58 % (dispatch /
   i-cache bound), while the real model-scale shapes gain only 0.5–4 % because they are dominated by
   weight bandwidth and the address class only moves the constant overhead.
2. **The cost lands on the displaced neighbour**, not on untouched operators: `rms_norm` — the very
   anchor the move steps over — regresses +1.86 %, while operators whose sections did not move
   (`reshape_and_cache`, `reduce_variance`) stay inside noise (−0.01 %, −0.44 %).
3. **Therefore the benefit is not transferable by analogy**: a move for another operator needs its own
   two-arm measurement *plus* the neighbour-cost accounting; this is exactly the protocol above.
