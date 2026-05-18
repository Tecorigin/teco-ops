# Teco-Ops

Teco-Ops 算子开发项目，提供基于 SDAA C 编程模型的高性能算子实现、C++ 接口封装、Python API 绑定（PyTorch 扩展）及完整的测试框架。

推荐用户优先阅读以下基础手册，再进行算子开发实操：

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

## 代码架构

本项目采用 interface + ual 分层架构设计（参考 teco-al 项目），主要包含两个接口层：

### Interface 层（用户接口）

Interface 层提供面向用户的 C API，屏蔽底层实现细节，是算子的统一入口。

- **用户 API**（`teco/interface/include/`）：`tecoops.h` 头文件，定义 `tecoopsHandle_t`、`tecoopsStatus_t`、`tecoopsAlgo_t` 等类型及各算子的 C 接口函数（如 `tecoopsFlattenRays`）
- **接口实现**（`teco/interface/ops/`）：各算子的接口实现文件，负责参数组装、调用 `RUN_OP` 宏分发到 ual 层
- **公共组件**（`teco/interface/common/`）：handle 管理、参数转换、宏定义（`RUN_OP`）等

### UAL 层（统一算子库）

UAL 层是算子的核心实现层，负责分支选择和设备端计算。

- **参数结构**（`teco/ual/args/`）：各算子的参数结构体定义（如 `FlattenRaysArgs`、`FlattenRaysPatchArgs`）
- **分支分发**（`teco/ual/ops/`）：各算子的 Op 类（继承 `BaseOp`），通过 `find()` 选择数据类型/算法分支，`run()` 调用对应 kernel
- **设备计算**（`teco/ual/kernel/`）：`.scpp` 核函数实现，包含 `__global__` kernel 和 launch 函数
- **公共组件**（`teco/ual/com/`）：数据类型定义、日志、状态码等

每个算子的典型文件结构：
```
teco/
├── interface/
│   ├── include/
│   │   └── tecoops.h              # 用户 API 头文件（所有算子统一声明）
│   ├── ops/
│   │   └── my_op.cpp              # 接口实现（参数组装 + RUN_OP 分发）
│   └── common/
│       ├── handle.h/cpp           # tecoopsHandle_t 管理
│       ├── convert.h/cpp          # 数据类型转换
│       ├── marco.h                # RUN_OP 宏
│       └── check.h                # 状态检查
├── ual/
│   ├── args/
│   │   └── my_op_args.h           # 参数结构体（Args + PatchArgs）
│   ├── ops/
│   │   └── my_op/
│   │       ├── my_op.hpp          # Op 类定义（继承 BaseOp）
│   │       ├── find_my_op.h/cpp   # 分支选择逻辑
│   │       └── base_op.hpp        # BaseOp 基类
│   ├── kernel/
│   │   └── my_op/
│   │       ├── my_op.h            # kernel 函数声明
│   │       └── my_op.scpp         # SDAA C kernel 实现
│   └── com/
│       ├── def.h                  # UALDataType/UALAlgoType 定义
│       ├── log.h                  # 日志工具
│       └── status.h               # 状态码
└── CMakeLists.txt
```

调用流程：`tecoopsMyOp(handle, ...) → RUN_OP → MyOpOp::find()（分支选择） → MyOpOp::run()（kernel 执行）`

### 命名规则

**重要：** `{opname}` 变量表示算子在目录名或文件名中的名称，是项目自动化构建脚本的索引，所有地方必须完全严格一致。

| 层级 | 路径 | 文件名规则 |
|------|------|-----------|
| Interface 层 | `teco/interface/ops/` | `{opname}.cpp` |
| UAL 参数层 | `teco/ual/args/` | `{opname}_args.h` |
| UAL 分支层 | `teco/ual/ops/{opname}/` | `{opname}.hpp`, `find_{opname}.cpp`, `find_{opname}.h` |
| UAL Kernel 层 | `teco/ual/kernel/{opname}/` | `{opname}.h`, `{opname}.scpp` |

### 编译构建

本项目使用 CMake 构建系统，通过 `build.sh` 脚本简化编译操作：

```bash
# 编译所有算子
bash build.sh

# 编译指定算子（teco 目录下的算子）
bash build.sh -f "flatten_rays"

# 编译指定架构
bash build.sh --arch teco    # TECO 架构
bash build.sh --arch cuda    # CUDA 架构（用于产生精度基线）
bash build.sh --arch all     # 两者都构建
```

**SPM 内存限制：** SPM 内存申请时，不能超过 235KB，推荐使用仓库封装的 `rt_spm_malloc()` 与 `rt_spm_free()` 等接口（接口封装占用 128B，上限为 240512B），可以在超量申请时报错提醒。

### Python API 接口

Python API 基于 PyTorch 扩展机制，将底层 C++ 算子封装为易于使用的 Python 接口，支持 Tensor 直接传入，无缝集成到 PyTorch 训练/推理流程中。内部通过 interface 层（`tecoopsHandle_t`）调用算子。

- **PyTorch 绑定**（`api/`）：
  - `torch_ext.cpp`：使用 `TORCH_LIBRARY` 宏注册算子，通过 `TecoExtension` 编译为独立 C++ 扩展模块
  - `tecoops/__init__.py`：Python 包入口，导出算子接口

- **两种算子绑定方式**：
  - **普通 PyTorch 扩展**（如 `flatten_rays`）：通过 CMake 构建为 `libteco_ops.so`，再由 `torch_ext.cpp` 注册为 PyTorch 扩展
  - **load_inline 动态加载**（如 `axpy`）：使用 `torch.utils.cpp_extension.load_inline` 在运行时动态编译，适合快速原型开发

- **Python API 测试**（`python_api_test/`）：Python 接口的功能自测脚本

Python 使用示例：
```python
import torch
import tecoops

# 使用 PyTorch Tensor 直接调用算子
rays = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32, device='sdaa')
N, M = rays.shape
res = torch.empty(N * M, dtype=torch.int32, device='sdaa')
tecoops.flatten_rays(rays, N, M, res)
```

- **测试框架**（`test/`）：基于 proto 配置的 C++ 测试框架，支持精度校验和性能分析

详见 [Python API 接口说明](doc/README_PYTHON.md) 和 [算子开发指南](doc/README_OP.md)。

## 目录结构

```
Teco-Ops/
├── teco/                   # TECO 算子实现（interface + ual 分层架构）
│   ├── interface/          # Interface 层：用户 API 接口
│   │   ├── include/
│   │   │   └── tecoops.h  #   用户 API 头文件
│   │   ├── ops/            #   各算子接口实现
│   │   │   └── flatten_rays.cpp
│   │   └── common/         #   handle、convert、RUN_OP 宏等
│   ├── ual/                # UAL 层：统一算子库（核心实现）
│   │   ├── args/           #   参数结构体定义
│   │   │   └── flatten_rays_args.h
│   │   ├── ops/            #   Op 类（分支分发）
│   │   │   ├── base_op.hpp
│   │   │   └── flatten_rays/
│   │   ├── kernel/         #   设备端 kernel 实现（.scpp）
│   │   │   └── flatten_rays/
│   │   └── com/            #   数据类型、日志、状态码等
│   └── CMakeLists.txt
├── cuda/                   # CUDA 算子实现（精度基线）
├── common/                 # 公共头文件和工具
├── api/                    # Python API 绑定代码
│   ├── torch_ext.cpp       # PyTorch 扩展绑定（TORCH_LIBRARY 注册）
│   └── tecoops/            # Python 包
│       └── __init__.py
├── test/                   # C++ 测试框架
│   ├── src/               # 测试框架源码
│   ├── test_proto/        # Proto 定义文件
│   │   ├── optest.proto
│   │   ├── tensor.proto
│   │   ├── tecokernel.proto
│   │   └── tecokernel/    # 各算子参数 proto
│   ├── zoo/               # 算子测试用例
│   │   └── teco/
│   │       └── <op_name>/
│   │           ├── <op_name>.cpp  # 测试代码
│   │           └── test_case/      # prototxt 测例
│   ├── CMakeLists.txt
│   └── build.sh
├── python_api_test/        # Python API 接口测试脚本
├── examples/               # 示例脚本
├── doc/                    # 文档
│   ├── README_OP.md        # 算子开发指南
│   ├── README_PYTHON.md    # Python 接口说明
│   └── QA.md               # 常见问题解答
├── build.sh                # 算子库构建脚本
├── setup.py                # Python 绑定构建脚本
├── requirements.txt
└── README.md
```

## 快速开始

### 步骤一：Fork 仓库

将本仓库 Fork 到您的个人空间，点击仓库页面右上方的 Fork 按钮即可。详情可查阅 [Gitee Fork+PullRequest 模式文档](https://help.gitee.com/base/%E5%BC%80%E5%8F%91%E5%8D%8F%E4%BD%9C/Fork+PullRequest%E6%A8%A1%E5%BC%8F)。

### 步骤二：算子功能开发

在 `teco/` 目录下按 interface + ual 分层结构添加算子文件。参考 [算子开发指南](doc/README_OP.md) 了解详细步骤，包括：

1. 在 `teco/interface/include/tecoops.h` 中声明算子 C API
2. 在 `teco/interface/ops/` 中实现接口（参数组装 + `RUN_OP` 分发）
3. 在 `teco/ual/args/` 中定义参数结构体
4. 在 `teco/ual/ops/` 中实现 Op 类（分支分发）
5. 在 `teco/ual/kernel/` 中实现设备端 kernel（`.scpp`）
6. 添加 Proto 参数定义
7. 编写测试代码和测试用例

**开发注意事项：**
- 所有算子目录名和文件名必须保持一致，作为自动化构建脚本的索引
- 新增文件需参考已有文件，在文件头添加 BSD License
- 编码统一使用 [Google C++ 风格](https://zh-google-styleguide.readthedocs.io/en/latest/google-cpp-styleguide/contents.html)
- SPM 内存申请不超过 235KB

### 步骤三：C++ 接口算子自测

使用项目内置的 C++ 测试框架进行算子精度和性能验证：

```bash
cd test
source env.sh

# 构建所有算子测试
sh build.sh --arch teco

# 或构建指定架构
sh build.sh --arch cuda   # CUDA 架构（用于产生精度基线）
sh build.sh --arch all     # 两者都构建

# 运行测试（通过 gid 参数指定核组号）
./build/demo --gid=0
```

详细的测试框架使用说明请查阅 [算子开发指南](doc/README_OP.md) 和 [常见问题](doc/QA.md)。

### 步骤四：Python API 接口绑定构建

C++ 测试通过后，将算子注册到 PyTorch 扩展中：

```bash
# 安装依赖
pip install torch torch-sdaa

# 在 api/torch_ext.cpp 中添加新算子的绑定

# 构建并安装 Python 扩展（本地开发模式）
WITH_TORCH=ON python setup.py build_ext --inplace
```

参考 [Python 接口说明](doc/README_PYTHON.md) 了解绑定方式和接口设计规范。

### 步骤五：Python API 接口自测

运行 Python 测试脚本验证接口功能：

```bash
# 测试 flatten_rays 算子
python python_api_test/test_flatten_rays.py

# 测试 axpy 算子
python python_api_test/test_axpy.py
```

**注意：** 使用 torch 扩展时，需先 `import torch` 再 `import tecoops`。

### 步骤六：提交 PR

完成开发和自测后，提交 Pull Request。提交前请确保：

- **代码格式检查通过**：运行 `pre-commit run --all-files` 确保 cpplint 检查通过
- **一个 PR = 一个算子**：不支持一个 PR 包含多个算子，也不支持单个算子拆分为多个 PR
- **功能兼容**：新增功能不得破坏已有功能
- **代码规范**：删除调试代码、注释代码等非功能性内容
- **Commit 消息**：遵循提交规范，详见 [算子提交规范](#算子提交规范)

每个算子 PR 通常包含以下文件：
- Interface 层：`teco/interface/include/tecoops.h`（新增 API 声明）、`teco/interface/ops/my_op.cpp`（接口实现）
- UAL 层：`teco/ual/args/my_op_args.h`（参数结构体）、`teco/ual/ops/my_op/`（Op 类 + 分支选择）、`teco/ual/kernel/my_op/`（kernel 实现）
- Python 绑定：`api/torch_ext.cpp`（新增绑定代码）
- 测试用例：`test/zoo/teco/my_op/` 目录下的测试代码和 prototxt
- Python 测试：`python_api_test/test_my_op.py`

## 算子提交规范

### Commit 消息格式

```
[type](algo<num>)：<subject>
```

- `type` 必须为以下之一：
  - `feat` — 新增算子或功能
  - `fix` — 修复 bug
  - `perf` — 性能优化
  - `refactor` — 代码重构
  - `ci` — 持续集成配置变更
  - `tool` — 工具脚本变更
  - `docs` — 文档变更
  - `test` — 测试用例变更
- `algo<num>`：算法编号，如 `algo0` 表示默认算法实现，`algo1` 表示优化算法实现等
- `subject`：简短描述本次提交的内容（英文）

示例：
```bash
git commit -m "[feat](algo0)：add sigmoid operator"
git commit -m "[perf](algo1)：optimize memory access pattern for flatten_rays"
git commit -m "[fix](algo0)：correct boundary check in flatten_rays"
```

### PR 规范

- **1 PR = 1 个算子**：每个 PR 对应一个完整的算子实现
- 每个算子开发使用独立的 git 分支，分支名建议使用 `op/<op_name>` 格式
- PR 描述需包含：算子功能说明、接口设计、性能数据（如有）
- PR 提交后，后续 commit 会自动同步到已有 PR 中，无需新建 PR

**PR 自查清单：**
- [ ] 所有新增文件已添加 BSD License 头部
- [ ] Commit 消息符合规范
- [ ] 新功能不破坏原有功能
- [ ] SPM 内存申请未超过 235KB
- [ ] 测例精度和性能测试通过
- [ ] 已更新或新增算子设计文档（`doc/op_docs/`）

### 代码风格

- 遵循 [Google C++ Style Guide](https://zh-google-styleguide.readthedocs.io/en/latest/google-cpp-styleguide/contents.html)
- 所有新增文件需添加 BSD License 头部
- SPM 内存申请上限 235KB

### 代码格式化

贡献者可以使用 `tools/format2google` 脚本将代码规范化为 Google C++ 风格：

```bash
# 格式化单个文件
./tools/format2google path/to/file.cpp

# 格式化整个目录
./tools/format2google path/to/directory
```

### 代码风格检查

项目使用 cpplint 进行代码风格检查。在 `source env.sh` 后，git hooks 会自动安装，提交时会自动检查暂存区的 C++ 文件：

```bash
# 安装依赖
pip install cpplint

# 首次使用需要 source env.sh 来安装 git hooks
source env.sh

# 提交时自动检查
git add <files>
git commit -m "message"

# 跳过检查（不推荐）
git commit -n -m "message"
```

检查配置文件位于 `tools/CPPLINT.cfg`。

## 注意事项

- 非代码说明的注释代码，请删除（例如开发过程中的功能调试、打印代码等）
- PR 中的新功能不能破坏原有功能，需要兼容原有功能，只能新增代码，不能删除原有代码
- SPM 空间申请时，不要超过 235KB。推荐使用仓库封装的 `rt_spm_malloc()` 与 `rt_spm_free()` 等接口（上限 240512B）
- 提交 PR 前，请确保本地自测通过

## 文档

- [算子开发指南](doc/README_OP.md) — 算子实现、测例编写及测试步骤
- [算子设计文档](doc/op_docs/) — 各算子的设计文档
- [Python 接口说明](doc/README_PYTHON.md) — Python API 使用指南
- [常见问题](doc/QA.md) — 算子 proto 参数设置及测试框架说明

## License

请参考项目根目录的 LICENSE 文件。
