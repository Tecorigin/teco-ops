# rotary_embedding zoo 用例说明

本目录下有两个并列的测例目录，用途不同（`hy` = 异构，指 Hy-MT2-1.8B 线）：

| 目录 | 用途 | 内容 |
| --- | --- | --- |
| `test_case/` | **通用测例**：MiniCPM5-1B 几何的本队自建用例 | `case_0` decode T=1（dense q/k）、`case_1` prefill T=7（dense）、`case_2` decode 且 q/k 保留真实 fused-QKV 行 stride（q/k 行宽 2560 = (16+2+2)·128，输出 dense） |
| `test_case_hy/` | **Hy-MT2-1.8B 真实形状用例**（异构模型线） | `case_hy_0` decode T=1、`case_hy_1` prefill T=30；`Hq=16, Hkv=4, head_dim=128`，q/k 保留真实 fused-QKV 行 stride（3072 = (16+2·4)·128）而输出 dense（2048/512），即 in_stride ≠ out_stride |

差别要点：

- 两者都是 NeoX（half-split）配对、fp16、`DIFF1 / 1e-4` 门限；`test_case/` 是
  MiniCPM5-1B 的解码/预填充几何（`Hq=16, Hkv=2`），`test_case_hy/` 是
  Hy-MT2-1.8B 的几何（`Hq=16, Hkv=4`，即 GQA 比例不同）。
- `test_case_hy/` 的核心价值是行 stride：q/k 是 fused-QKV 切出的视图
  （in_stride = 3072），而 `q_out/k_out` 是 dense（2048/512），与
  `hunyuan_v1.py` 的 rope 层调用形态一致；`test_case/` 中只有 `case_2` 覆盖
  「padded 输入 + dense 输出」，且是 MiniCPM 的 2560 行宽。
- 这些目录未生成 `case_*/` 子目录，demo 走进程内 C++ 参考（同其余 zoo 用例约定）。

跑法（官方固定形态，任一 case）：

```bash
./build/demo --gid=0 --perf_repeat=50 --warm_repeat=3 --gtest_repeat=1 \
    --case_path=zoo/teco/rotary_embedding/test_case_hy/case_hy_0.prototxt
```
