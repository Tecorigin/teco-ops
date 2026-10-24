# Shared binding and flash ABI integration receipt (2026-10-24)

The open PRs are independently reviewable trees. They edit overlapping functions,
so choosing an entire file from the last PR can drop an earlier fix. This document
defines the combined contract, a dependency order, and the tested resolution.
It does not claim that the official repository has merged any PR.

## Resolution and dependency order

1. Keep #35's read-only key/value contiguous materialization for fused-QKV views.
2. Keep #37's RMSNorm current-stream binding and alignment checks, then #53's
   current-stream binding for the other Python entries. Keep one binding per entry.
3. Combine #36's device metadata validation, #38's complete-output/empty allocation,
   #39's cumulative query entry, #42's explicit scale/plugin contract, and #47's
   rejection/status checks. These are one Python flash wrapper, not five wrappers.
4. Apply #41's D256/D512 and retained D128 kernels to that ABI. Do not restore its
   reverted D128 score double-pane experiment. Its kernel entry must preserve the
   cumulative metadata flag from #39 in every specialization.
5. Integrate the remaining kernel owners before refreshing #50's layout receipt.
   Rotary is #48, RMSNorm compat is #45, and reduce_variance is #52. A partial-tree
   fingerprint cannot certify the complete tree.

The combined source is available at team ops commit
[`db1597e`](https://github.com/tecorigin-caet/teco-ops-SCUdoudui/tree/db1597e1265a5fc3e24207b793c9d1dd94327a24).
The two small follow-ups that close the ABI interaction are
[`6beadb7`](https://github.com/tecorigin-caet/teco-ops-SCUdoudui/commit/6beadb7373151d8ed9035d7c0a085b4ae6afe861)
(metadata decoding in both kernel TUs) and
[`63d0583`](https://github.com/tecorigin-caet/teco-ops-SCUdoudui/commit/63d05835e49ca7169201b67677b5e2e64c762034)
(combined scale/cumulative C entry and callers).
Use these exact function/field resolutions when an ordinary merge reports conflicts;
do not select an older complete `api/torch_ext.cpp` or kernel file by PR number.

## Canonical ABI

`FlashAttentionArgs.q_seq_lens_are_cumulative` belongs to the shared args layout.
All kernel TUs and the host implementation must be rebuilt against that same header.
The four public entries delegate to one internal host implementation:

| C entry | Query metadata | Scale |
| --- | --- | --- |
| `tecoopsFlashAttention` | per-sequence `[batch]` | default `1/sqrt(D)` |
| `tecoopsFlashAttentionWithScale` | per-sequence `[batch]` | explicit |
| `tecoopsFlashAttentionCuSeqlensQ` | cumulative `[batch+1]` | default |
| `tecoopsFlashAttentionCuSeqlensQWithScale` | cumulative `[batch+1]` | explicit |

The Python wrapper validates rank/minimum length, pins the current stream and calls
`CuSeqlensQWithScale` directly with device metadata. The plugin's ONNX input is
explicitly **per-sequence `q_seq_lens`**; it calls the legacy or scaled per-sequence
entry. Passing `[0,L]` to a per-sequence entry is invalid metadata, not a prefill
special case. The kernel decodes metadata at entry without a backend-selection
branch in the arithmetic loop. Keep #47's output sentinel/status tests and release
descriptors before throwing a Python error.

## Tested merged artifact

Vendor Python `/home/py312/bin/python`, SDAA SDK/runtime 3.2.0, isolated install
`/root/tmpwork/fix-20261024/team/src/install`:

| Artifact | SHA-256 |
| --- | --- |
| core | `531e0caf206a3c580e03f87965d067e44c0d88f8740a06a3d20abd88377574dc` |
| extension | `82aa5455938789bc0009f85a947edfa271918a9d0b115866ad7f70bb57fbbf41` |
| wheel | `ca600ebda03b34adb070e25e615f746d68123350c291afdd1a4403e4caa111c6` |

Focused Python flash (default/explicit scale, prefill/decode/chunked/GQA/mixed,
zero-length padding, rejection/sentinel), RMSNorm (even tail/current stream and
odd-width rejection), reshape/cache (contiguous and fused-QKV), and reduce_variance
tests passed against this artifact. Each of the four models independently passed
fixed-prompt greedy-32 with its own launcher and this bundle. Real decode census
replay covers 22 observed shapes and 31 conformance cases: 295 PASS, zero failures;
see the team tree's `doc/validation/20261024/` results. No performance gain is claimed.

TVM/TecoInference plugin end-to-end execution is not available in this host's vendor
Python environment. The C API tests cover rejection/sentinel behavior but do not
substitute for plugin SDK execution. Official current-head CI/review remains a
separate gate; older green comments do not certify a new merged artifact.

## Rebuild and layout gates

After resolving the complete tree, rebuild/reinstall both core and extension into
one isolated prefix, verify `/proc/self/maps` and record new SHA-256 values. Run
the focused tests above, every fixed zoo case, and each affected model's greedy-32
and steady-state gate. Do not reuse these bytes as the expected hash for another
build root. Any performance claim additionally needs three original timings plus
median, whole-table accounting and a second layout window where required.

Refresh the TU-order fingerprint and absolute device `(name,address,size)` table
from that build, then enable the strict fingerprint check. Source size can move
absolute addresses even when the TU-order hash is unchanged. Preserve pure TU
ordering; do not introduce `list(SORT)` or padding kernels.
