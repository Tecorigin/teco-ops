# reduce_variance：接口与数值契约

Python 入口为 `tecoops.reduce_variance(input, axis, correction)`，输出与
`torch.var(input, dim=axis, correction=correction, keepdim=False)` 同形状。
当前 Python 绑定只接受连续的 SDAA `float32` 输入、合法的 axis
（允许负 axis）及 `correction=0/1`；不支持的 dtype 在 kernel 启动前拒绝。

## SIMD 累加与精度

本 PR 改变了归约的累加顺序与中间精度，不能承诺与旧实现或 CPU 参考逐位一致。
left/middle 路径使用 FP32 SIMD 累加；right 路径不再因为行宽不是 16 的倍数
而整行退回串行 FP64 累加，而是先按 16 lane 累加 FP32 部分和，再合并部分和，
最后处理不足 16 元素的标量尾部。因此旧 FP64 路径的逐位结果不是当前 API 契约。
`simd_loadu` 支持不对齐行起点，尾循环仅访问实际行宽以内的元素。

正确性按既有官方测例的 DIFF3_MAX 判据及 Python 测试的 `atol=1e-6`、
`error_num=0` 验收；不得把重复运行逐位一致解释为对 CPU 参考逐位一致。
该阈值是当前回归测例的验收标准，不是任意输入分布的通用误差上界。
大偏置、强消去或新的数值范围仍须与参考实现重新验证。

截至 2026-10-25（北京时间）的官方 CI 实测如下；下方来源链接固定该次被测提交，
这些结果不代表后续新 head 已通过 CI：

| 测例 | shape / axis / correction | 最大绝对误差 |
| --- | --- | ---: |
| left | `(150,3,3,200)` / 0 / 0 | 4.768372e-7 |
| middle | `(3,120,3,200)` / 1 / 0 | 5.960464e-7 |
| right | `(3,120,3,200)` / 3 / 0 | 1.192093e-7 |
| correction | `(2,3,4,8)` / 2 / 1 | 2.384186e-7 |

四例 `error_num=0`，且相同输入两次 launch 的 FP32 位模式逐位一致。
这些是该次 CI 的实测结果，不能推广为所有 shape 的误差保证。

来源：[官方 CI 原值](https://github.com/Tecorigin/teco-ops/pull/52#issuecomment-6386772611)。

## 复现

使用本次源码构建并更新安装 wheel 后，确认映射的 core/extension SHA-256：

```bash
/home/py312/bin/python python_api_test/test_reduce_variance.py
```

固定官方 zoo 性能测例仍使用原命令与原始 case，不用新增回归 shape 替换它们：

```bash
./build/demo --gid=0 --perf_repeat=50 --warm_repeat=3 --gtest_repeat=1 \
  --case_path=zoo/teco/reduce_variance/test_case/case_2.prototxt
```

Python 回归覆盖三种 axis、correction=1、重复运行确定性和 fp16 拒绝路径；
新增的 `case_3/4` 另覆盖非 16 倍数的列宽，不改变原固定性能测例。
