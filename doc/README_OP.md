# 算子开发指南

本文档介绍如何在 Teco-Ops 项目中添加新算子、编写测试用例及运行测试的完整流程。

## 开发前准备

推荐在开始算子开发前，先阅读以下基础手册：

- [SDAA C 零基础入门](http://docs.tecorigin.com/release/sdaac_beginner_guide/#a85ae5a993af55d6ae2cf60feda4f55f)：面向零基础学习者的入门教程。
- [SDAA C 编程指南](http://docs.tecorigin.com/release/sdaac/)：介绍 SDAA C 编程语言、语言规范、函数接口、数学函数、程序编译、程序调试及性能调优等内容。
- [性能优化手册-SDAAC 篇](http://docs.tecorigin.com/release/sddac_perf_opt/)：介绍程序并行、函数接口、数学函数、程序编译过程中的性能优化内容。
- [性能优化手册-算子篇](http://docs.tecorigin.com/release/op_perf_opt/)：介绍经典的计算与访存优化办法，包括向量指令、指令流水线、矩阵乘法加速单元、双缓冲、广播优化等。

## 常用概念

本项目使用以下核心概念，理解这些概念有助于更好地理解代码架构和算子实现：

| 概念 | 说明 |
|------|------|
| **ops** | 算子（Operator）的缩写，指执行特定计算逻辑的功能单元，如矩阵乘法、卷积、激活函数等 |
| **ual** | 统一算子库（Unified Accelerated Libraries）的缩写，是算子的核心实现层，负责分支选择和设备端计算 |
| **algo** | 算法（Algorithm）的缩写，用于指定同一算子的不同实现方式，通过 `algo` 参数可以在运行时选择最优的计算路径 |
| **args** | 参数（Arguments）的缩写，用于封装算子执行所需的配置信息，如计算参数和分支派发参数 |
| **handle** | 句柄（Handle）的缩写，作为算子执行的上下文，管理设备信息（如 `spe_num`、`stream` 等） |
| **tensor** | 张量，表示多维数据数组，是算子的输入输出数据格式 |
| **descriptor** | 描述符，用于描述张量的属性（shape、dtype、layout 等）或算子的配置信息 |
| **workspace** | 工作空间，算子执行时申请的临时存储区域，用于存放中间计算结果 |
| **kernel** | 核函数，设备端（Device）执行的计算函数，通常使用 `__global__` 属性标记 |
| **branch/dispatch** | 分支派发，根据输入参数（如数据类型、算法类型）选择对应 kernel 实现的过程 |
| **SPM** | 片上共享内存（Shared Private Memory），一种高速片上存储，容量有限（约 235KB），需谨慎申请 |

## 代码风格

详见 [算子提交规范 - 代码风格](PR.md#代码风格)。

> **注意：** 建议同时创建对应算子的设计文档，参考 [算子设计文档模板](op_docs/doc_template.md) 和已有算子文档（如 [flatten_rays 设计文档](op_docs/flatten_rays.md)）进行编写。

## 分层架构设计

本项目采用 interface + ual 分层架构（参考 teco-al 项目设计），将算子实现分为两层：

### Interface 层（用户接口层）

Interface 层提供面向用户的 C API，是算子的统一入口，屏蔽底层实现细节。

- **核心概念**：`tecoopsHandle_t` 作为算子执行的上下文句柄，管理 `spe_num`、`stream` 等设备信息
- **调用方式**：用户通过 `tecoopsMyOp(handle, ...)` 调用算子，接口层负责参数组装并通过 `RUN_OP` 宏分发到 ual 层
- **RUN_OP 宏**：`RUN_OP(OpType, args, patch_args, handle)` 自动完成 `Op.find()`（分支选择）和 `Op.run()`（kernel 执行）

### UAL 层（统一算子库层）

UAL 层是算子的核心实现层，负责分支选择和设备端计算。

- **参数结构体**：每个算子定义 `Args`（运行参数）和 `PatchArgs`（分支选择参数，包含 `data_type` 和 `algo`）
- **Op 类**：继承 `BaseOp<MyOpOp, MyOpType>`，通过 `find()` 根据数据类型和算法选择具体 kernel 实现，`run()` 调用 kernel 执行
- **Kernel**：`.scpp` 文件实现 `__global__` 核函数，是最终的设备端计算逻辑

### 调用流程

```
tecoopsMyOp(handle, ...)     # Interface 层：用户 API
  → 组装 Args + PatchArgs
  → RUN_OP(MyOpOp, args, patch_args, handle)
    → MyOpOp::find(&patch_args)   # UAL ops 层：分支选择
      → findMyOpBranch()          # 根据 data_type/algo 选择 kernel
    → MyOpOp::run(&args, stream)  # UAL ops 层：kernel 执行
      → RUN_KERNEL(instance, stream, args)  # UAL kernel 层：设备计算
```

## 添加新算子

按照以下步骤在项目中添加新算子，实现 interface 层和 ual 层的完整代码：

在 `teco/` 目录下按 interface + ual 分层结构添加算子文件。包括：

1. 在 `teco/interface/include/tecoops.h` 中声明算子 C API
2. 在 `teco/interface/ops/` 中实现接口（参数组装 + `RUN_OP` 分发）
3. 在 `teco/ual/args/` 中定义参数结构体
4. 在 `teco/ual/ops/` 中实现 Op 类（分支分发）
5. 在 `teco/ual/kernel/` 中实现设备端 kernel（`.scpp`）
6. 添加 Proto 参数定义
7. 编写测试代码
8. 添加测试用例
9. 构建并执行测试

### 1. 在 Interface 层声明 API

在 `teco/interface/include/tecoops.h` 中添加算子的 C API 声明：

```cpp
tecoopsStatus_t tecoopsMyOp(
    tecoopsHandle_t handle,
    const void *input, void *output, int size,
    int64_t factor, int64_t offset,
    tecoopsAlgo_t algo);
```

### 2. 在 Interface 层实现接口

在 `teco/interface/ops/` 下创建接口实现文件，完成参数组装和分发：

```cpp
#include "interface/include/tecoops.h"
#include "interface/common/macro.h"
#include "interface/common/convert.h"
#include "ual/args/my_op_args.h"
#include "ual/ops/my_op/my_op.hpp"

using tecoops::ual::args::MyOpArgs;
using tecoops::ual::args::MyOpPatchArgs;
using tecoops::ual::ops::MyOpOp;
using tecoops::Convert;

tecoopsStatus_t tecoopsMyOp(
    tecoopsHandle_t handle,
    const void *input, void *output, int size,
    int64_t factor, int64_t offset,
    tecoopsAlgo_t algo) {
    if (handle == nullptr) {
        return TECOOPS_STATUS_NOT_INITIALIZED;
    }

    MyOpArgs arg;
    arg.spe_num = handle->spe_num;
    arg.input = input;
    arg.output = output;
    arg.size = size;
    arg.factor = factor;
    arg.offset = offset;

    MyOpPatchArgs patch_arg;
    patch_arg.atargs = &arg;
    patch_arg.data_type = Convert::toUALDataType(algo);
    patch_arg.algo = Convert::toUALAlgoType(algo);

    RUN_OP(MyOpOp, arg, patch_arg, handle);

    return TECOOPS_STATUS_SUCCESS;
}
```

### 3. 在 UAL 层定义参数结构体

定义算子的运行参数和分支选择参数：

```cpp
#ifndef TECO_UAL_ARGS_MY_OP_ARGS_H_
#define TECO_UAL_ARGS_MY_OP_ARGS_H_

#include "ual/com/def.h"

namespace tecoops {
namespace ual {
namespace args {

struct MyOpArgs {
    int spe_num;
    const void* input;
    void* output;
    int size;
    int64_t factor;
    int64_t offset;
};

struct MyOpPatchArgs {
    MyOpArgs* atargs;
    UALDataType data_type;
    UALAlgoType algo;
};

}  // namespace args
}  // namespace ual
}  // namespace tecoops

#endif  // TECO_UAL_ARGS_MY_OP_ARGS_H_
```

### 4. 在 UAL 层实现 Op 类（分支分发）

实现 Op 类，通过 find() 方法选择数据类型/算法分支：

```
teco/ual/ops/my_op/
├── my_op.hpp              # Op 类定义
├── find_my_op.h           # 分支选择函数声明
└── find_my_op.cpp         # 分支选择逻辑实现
```

**my_op.hpp**（Op 类定义）：

```cpp
#include "ual/kernel/my_op/my_op.h"
#include "ual/com/log.h"
#include "ual/args/my_op_args.h"
#include "ual/com/def.h"
#include "ual/ops/base_op.hpp"
#include "ual/ops/my_op/find_my_op.h"

using tecoops::ual::args::MyOpArgs;
using tecoops::ual::args::MyOpPatchArgs;
using namespace tecoops::ual::common;
using namespace tecoops::ual::kernel;

namespace tecoops {
namespace ual {
namespace ops {

struct MyOpType {
    using ArgsType = MyOpArgs;
    using PatchType = MyOpPatchArgs;
    using RetType = void;
    using PImplType = void (*)(ArgsType);
};

static MyOpType::PImplType MyOpAlgos[] = {
    tecoKernelMyOpFloat,
};

static const char *MyOpDiscription[] = {
    "tecoKernelMyOpFloat",
};

struct MyOpOp : public BaseOp<MyOpOp, MyOpType> {
    using ArgsType = typename MyOpType::ArgsType;
    using PatchType = typename MyOpType::PatchType;

    static const char *name() { return "my_op"; }

    common::Status findImpl(const PatchType *args) {
        int index = findMyOpBranch(args);
        if (index == -1) {
            ERROR("my_op branch is not exit!");
            return common::Status::NOT_IMPLEMENTED;
        }
        setInstance(MyOpAlgos[index], MyOpDiscription[index]);
        return common::Status::SUCCESS;
    }
};

}  // namespace ops
}  // namespace ual
}  // namespace tecoops
```

**find_my_op.cpp**（分支选择）：

```cpp
#include "ual/ops/my_op/find_my_op.h"
#include "ual/com/convert.hpp"

using tecoops::ual::args::MyOpPatchArgs;

namespace tecoops {
namespace ual {
namespace ops {

int findMyOpBranch(const MyOpPatchArgs *arg) {
    int algo = common::convertAlgoToIndex(arg->algo);
    if (arg->data_type == UALDataType::UAL_DTYPE_FLOAT) {
        return algo;
    }
    return -1;
}

}  // namespace ops
}  // namespace ual
}  // namespace tecoops
```

### 5. 在 UAL 层实现 Kernel（设备端计算）

实现设备端核函数，完成实际计算逻辑：

```
teco/ual/kernel/my_op/
├── my_op.h            # kernel 函数声明
└── my_op.scpp         # SDAA C kernel 实现
```

**my_op.h**（kernel 声明）：

```cpp
#ifndef TECO_UAL_KERNEL_MY_OP_MY_OP_H_
#define TECO_UAL_KERNEL_MY_OP_MY_OP_H_

#include "ual/args/my_op_args.h"

using tecoops::ual::args::MyOpArgs;

void tecoKernelMyOpFloat(MyOpArgs arg);

#endif  // TECO_UAL_KERNEL_MY_OP_MY_OP_H_
```

**my_op.scpp**（kernel 实现）：

```cpp
#include "ual/kernel/my_op/my_op.h"
#include "ual/kernel/macro.h"

template <typename scalar_t>
__global__ void my_op_kernel(const scalar_t *input, scalar_t *output, int size, int64_t factor, int64_t offset) {
    int idx = threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * static_cast<scalar_t>(factor) + static_cast<scalar_t>(offset);
    }
}

void tecoKernelMyOpFloat(MyOpArgs arg) {
    my_op_kernel<float><<<1, arg.spe_num>>>(static_cast<const float*>(arg.input), static_cast<float*>(arg.output), arg.size, arg.factor, arg.offset);
}
```

**SPM 内存限制：** SPM 内存申请时，不能超过 235KB，推荐使用仓库封装的 `rt_spm_malloc()` 与 `rt_spm_free()` 等接口（接口封装占用 128B，上限为 240512B），可以在超量申请时报错提醒。

### 6. 添加 Proto 参数定义

如果算子需要额外参数（除输入输出张量外），需在 proto 中定义：

#### 6.1 创建算子参数 proto 文件

定义算子特有的参数结构：

示例（`test/test_proto/tecokernel/my_op.proto`）：

```proto
syntax = "proto2";
package testpt;

message MyOpParam {
    optional int64 factor = 1 [default = 2];
    optional int64 offset = 2 [default = 0];
}
```

#### 6.2 注册到 tecokernel.proto

将新算子参数注册到统一的 proto 文件中：

```proto
syntax = "proto2";
package testpt;

import "tensor.proto";
import "tecokernel/my_op.proto";  // 导入新算子的 proto

message TecokernelParam {
    optional MyOpParam my_op_param = 3;  // 使用新的 field 号
}
```

### 7. 编写测试代码

继承 TecoExecutor 类，实现算子的测试逻辑：

```
test/zoo/teco/
└── my_op/
    ├── my_op.h
    └── my_op.cpp
```

#### 7.1 头文件

定义测试类，继承 TecoExecutor：

```cpp
#ifndef MY_OP_EXECUTOR_H
#define MY_OP_EXECUTOR_H

#include "zoo/teco/executor.h"

namespace optest {

class MyOpExecutor : public TecoExecutor {
public:
    void paramParse() override;
    void paramGeneration() override;
    void compute() override;
    void cpuCompute() override;
    void gpuCompute() override;
    int64_t getTheoryOps() override;
    int64_t getTheoryIoSize() override;
    void paramCheck() override;
    void destroy() override;

private:
    // 从 proto 或张量获取的参数
    int64_t factor_;
    int64_t offset_;
    
    // 设备指针
    void *input;
    void *output;
};

}  // namespace optest

#endif
```

#### 7.2 实现文件

实现测试类的各个方法：

```cpp
#include "zoo/teco/my_op/my_op.h"
#include "interface/include/tecoops.h"
#include "interface/common/handle.h"
#include "zoo/teco/convert.h"

namespace optest {

void MyOpExecutor::paramParse() {
    // 从张量元数据获取 shape/stride 等
    int64_t size = parser_->inputs()[0].shape_count;
    
    // 从 proto 获取额外参数
    auto param = parser_->getProtoNode()->tecokernel_param().my_op_param();
    factor_ = param.factor();
    offset_ = param.offset();
}

void MyOpExecutor::paramGeneration() {
    input = dev_input[0];
    output = dev_output[0];
}

void MyOpExecutor::compute() {
#ifdef USE_TECO
    static tecoopsHandle_t handle = nullptr;
    if (handle == nullptr) {
        tecoopsCreate(&handle);
    }
    tecoopsMyOp(handle,
                static_cast<const void*>(input),
                static_cast<void*>(output),
                size,
                factor_,
                offset_,
                TECOOPS_ALGO_0);
#endif
}

void MyOpExecutor::cpuCompute() {
    // CPU 参考实现，用于验证结果正确性
    auto* input_ptr = static_cast<float*>(baseline_input[0]);
    auto* output_ptr = static_cast<float*>(baseline_output[0]);
    int64_t size = parser_->inputs()[0].shape_count;
    
    for (int64_t i = 0; i < size; ++i) {
        output_ptr[i] = input_ptr[i] * static_cast<float>(factor_) + offset_;
    }
}

void MyOpExecutor::gpuCompute() {
#ifdef USE_CUDA
    // CUDA 实现（如有）
#endif
}

int64_t MyOpExecutor::getTheoryOps() {
}

int64_t MyOpExecutor::getTheoryIoSize() {
}

void MyOpExecutor::paramCheck() {}
void MyOpExecutor::destroy() {}

}  // namespace optest
```

### 8. 创建测试用例（prototxt）

编写 prototxt 格式的测试用例：

```
test/zoo/teco/my_op/
└── test_case/
    └── 0.prototxt
```

示例（`test/zoo/teco/my_op/test_case/0.prototxt`）：

```protobuf
op_name: "my_op"

input {
  id: "input"
  shape {
    dims: 16
  }
  layout: LAYOUT_ARRAY
  dtype: DTYPE_FLOAT
  random_data {
    seed: 12345
    lower_bound: -1
    upper_bound: 1
    distribution: UNIFORM
  }
  ttype: TENSOR
}

output {
  id: "output"
  shape {
    dims: 16
  }
  layout: LAYOUT_ARRAY
  dtype: DTYPE_FLOAT
  ttype: TENSOR
}

tecokernel_param {
  my_op_param {
    factor: 2
    offset: 0
  }
}
```

### 9. 构建并运行测试

构建算子库及测试，并执行测试。

```bash
# 构建所有算子（TECO 架构）
bash build.sh --build teco

# 构建所有算子测试
cd test
source env.sh
sh build.sh --arch teco

# 运行测试
./build/demo --gid=0
```

## Proto 参数说明

Proto 文件是测试框架的核心配置，主要包含两部分：

1. **张量定义**（`input`/`output`）：定义输入输出张量的 shape、dtype、layout 等
2. **算子参数**（`tecokernel_param`）：定义算子特有的参数

### 测试框架解析流程

```
Parser  --解析-->  MetaTensor (inputs/outputs)
   ↓
Executor (dev_input, dev_output, baseline_input, baseline_output)
   ↓
<OpName>Executor (具体算子的测试实现)
```

### 在测试代码中获取参数

```cpp
// 获取张量信息
auto& input_meta = parser_->inputs()[0];
int64_t size = input_meta.shape_count;
int64_t dtype = input_meta.dtype;

// 获取算子参数
auto param = parser_->getProtoNode()->tecokernel_param().my_op_param();
int64_t factor = param.factor();

// 获取设备指针
void* dev_input_ptr = dev_input[0];
void* dev_output_ptr = dev_output[0];
```

## Executor 类说明

`Executor` 是测试框架的核心类，提供以下重要成员：

| 成员 | 说明 |
|------|------|
| `parser_` | 存储测例解析结果的 Parser 对象 |
| `dev_input[i]` | 第 i 个 input 的设备地址空间 |
| `dev_output[i]` | 第 i 个 output 的设备地址空间 |
| `host_input[i]` | 第 i 个 input 的 host 地址空间 |
| `host_output[i]` | 第 i 个 output 的 host 地址空间 |
| `baseline_input[i]` | baseline 输入数据（测试框架内部使用） |
| `baseline_output[i]` | baseline 输出数据（测试框架内部使用） |

**注意：**
- `dev_input`/`dev_output` 是设备地址，只能在 `__global__` 或 `__device__` 函数中访问
- `baseline_input`/`baseline_output` 由测试框架管理，不要修改

## 参考

- [常见问题](QA.md) — 算子 proto 参数设置及测试框架详细说明
- 示例算子：`test/zoo/teco/flatten_rays/`（使用 `tecoopsFlattenRays` 接口调用）
