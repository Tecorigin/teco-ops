# tecoopsFlattenRays 设计文档

## 计算原理

将输入光线数组展平为连续的结果数组。输入 `rays` 是一个二维数组，形状为 `(N, 2)`，每行包含两个值：
- `offset`：结果数组中的起始位置
- `num_steps`：该光线包含的步数

对于第 `n` 条光线，将 `res[offset ... offset + num_steps - 1]` 填充为光线索引值 `n`。

![flatten_rays](./pics/flatten_rays.png)

**计算公式：**
```
对于 n in [0, N):
    offset = rays[n * 2]
    num_steps = rays[n * 2 + 1]
    对于 i in [0, num_steps):
        res[offset + i] = n
```

**参数解释：**
- `rays`：输入光线数组，每条光线包含两个值：`offset`（起始位置）和 `num_steps`（步数）
- `N`：光线数量
- `M`：每条光线的数据长度（当前未使用，固定为2）
- `res`：输出结果数组

## 功能实现

### 接口设计

参考 PyTorch 的 `flatten` 操作和光线重建场景需求，设计 userAPI 接口：

```c++
tecoopsStatus_t tecoopsFlattenRays(
    tecoopsHandle_t handle,
    const int *rays, uint32_t N, uint32_t M, int *res,
    tecoopsAlgo_t algo);
```

### 参数信息

其中，各参数含义如下：

| 参数 | 输入/输出 | 主机端/设备端 | 说明 |
|---|---|---|---|
| handle | 输入 | 主机端 | Teco-Ops 句柄，管理设备上下文 |
| rays | 输入 | 设备端 | 指向光线数组的指针，形状为 `(N, 2)` |
| N | 输入 | 主机端 | 光线数量 |
| M | 输入 | 主机端 | 每条光线的数据长度（固定为2） |
| res | 输出 | 设备端 | 指向结果数组的指针 |
| algo | 输入 | 主机端 | 算法选择，当前仅支持 `TECOOPS_ALGO_0` |

### 类型限制

当前计算分支，主要完成以下功能实现，其余情况暂不支持。

| 参数 | 数据类型 | 维度信息 | 存储格式 |
|---|---|---|---|
| rays | int32 | `(N, 2)` | Array |
| res | int32 | `(max_offset + max_steps,)` | Array |
| N | uint32 | 标量 | - |
| M | uint32 | 固定为2 | - |

## 性能优化

### 数据分块

计算总量为 `N` 条光线，均分到各个 SPE 线程并行执行。每个线程处理的任务量为 `per_spe_num`，对应区间 `[start, end)`。

![flatten_tilling](./pics/flatten_tilling.png)

### 伪代码思路

```
total_rays = N
per_spe_num = (N + spe_num - 1) / spe_num
start = threadIdx * per_spe_num
end = MIN(start + per_spe_num, N)

for n in [start, end):
    offset = rays[n * 2]
    num_steps = rays[n * 2 + 1]
    res_ptr = res + offset
    for i in [0, num_steps):
        res_ptr[i] = n
```

### 性能数据

| 测试环境 | 硬件时间 (us) |
|---|---|
| CI 测试环境 | 待补充 |

## 分支派发

| 算法取值 | 计算分支 | 含义说明 |
|---|---|---|
| `TECOOPS_ALGO_0` | `tecoKernelFlattenRaysInt` | 基础实现，使用单线程束并行计算 |
| `TECOOPS_ALGO_1` ~ `TECOOPS_ALGO_9` | 未实现 | 暂不支持 |

## 文件结构

```
teco/
├── interface/
│   ├── include/tecoops.h           # userAPI 声明
│   └── ops/flatten_rays.cpp        # 接口实现
├── ual/
│   ├── args/flatten_rays_args.h   # 参数结构体
│   ├── ops/flatten_rays/
│   │   ├── flatten_rays.hpp       # Op 类定义
│   │   ├── find_flatten_rays.cpp  # 分支选择
│   │   └── find_flatten_rays.h
│   └── kernel/flatten_rays/
│       ├── flatten_rays.h         # kernel 声明
│       └── flatten_rays.scpp       # kernel 实现
```

## 使用示例

```python
import torch
import tecoops

rays = torch.tensor([[0, 3], [3, 2], [5, 4]], dtype=torch.int32, device='sdaa')
N, M = rays.shape
res = torch.empty(9, dtype=torch.int32, device='sdaa')

tecoops.flatten_rays(rays, N, M, res)
# res = [0, 0, 0, 1, 1, 2, 2, 2, 2]
```