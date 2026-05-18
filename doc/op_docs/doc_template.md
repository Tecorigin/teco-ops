# tecoops{OpName} 设计文档

## 计算原理

*主要提供这些信息：功能描述、计算公式、公式中的参数解释、计算图示（可选）。*

## 功能实现

### 接口设计

*结合当前算子的实际计算需求，参考 cuda、pytorch、paddlepaddle 等接口，进行 userAPI 设计。*

```c++
// userAPI design result
tecoopsStatus_t tecoops{OpName}(
    tecoopsHandle_t handle,
    ...);
```

### 参数信息

其中，各参数含义如下：

| 参数 | 输入/输出 | 主机端/设备端 | 说明 |
|---|---|---|---|
| param_1 | 输入/输出 | 主机端/设备端 | 参数功能描述。 |
| param_2 | 输入/输出 | 主机端/设备端 | 参数功能描述。 |
| ... | 输入/输出 | 主机端/设备端 | 参数功能描述。 |
| param_n | 输入/输出 | 主机端/设备端 | 参数功能描述。 |

### 类型限制

当前计算分支，主要完成以下功能实现，其余情况暂不支持。

| 参数 | 数据类型 | 维度信息 | 存储格式 |
|---|---|---|---|
| param_1 | 当前功能支持状况 | 当前功能支持状况 | 当前功能支持状况 |
| ... | 当前功能支持状况 | 当前功能支持状况 | 当前功能支持状况 |

## 性能优化

*建议包含以下内容：*

1. 标明自己实现的具体计算分支
2. 优化设计说明：通过伪代码、图示等办法，描述自己设计并实现的性能优化办法，如数据分块、广播操作、流水设计等
3. 性能自测数据（测例路径 + tecotest 的硬件时间均值）

## 分支派发

| 算法取值 | 计算分支 | 含义说明 |
|---|---|---|
| `TECOOPS_ALGO_0` | `tecoKernel{OpName}...` | 基础实现 |
| `TECOOPS_ALGO_1` | `tecoKernel{OpName}...` | 优化实现 |
| ... | ... | ... |

## 文件结构

```
teco/
├── interface/
│   ├── include/tecoops.h           # userAPI 声明
│   └── ops/{opname}.cpp           # 接口实现
├── ual/
│   ├── args/{opname}_args.h       # 参数结构体
│   ├── ops/{opname}/
│   │   ├── {opname}.hpp           # Op 类定义
│   │   ├── find_{opname}.cpp      # 分支选择
│   │   └── find_{opname}.h
│   └── kernel/{opname}/
│       ├── {opname}.h            # kernel 声明
│       └── {opname}.scpp          # kernel 实现
```

## 使用示例

```python
import torch
import tecoops

# 使用示例
```