# Teco-Ops

Teco-Ops 算子开发项目，提供基于 SDAA C 编程模型的高性能算子实现、C++ 接口封装、Python API 绑定（PyTorch 扩展）、Plugin 自定义算子接口（Teco-Inference 推理框架）及完整的测试框架。通过本项目，您可以高效地开发和优化自定义算子，将其封装为 C++/Python 接口并无缝集成到 PyTorch 或Teco-Inference中，同时利用内置测试框架全面验证算子的正确性与性能。

## 代码架构

本项目分为算子代码（采用 interface + ual 分层架构设计）和 Python API 接口两部分：

### 算子代码

算子代码采用 interface + ual 分层架构，详见 [算子开发指南](doc/README_OP.md)：

- **Interface 层**：用户 C API 入口
- **UAL 层**：核心实现层，负责分支选择和设备端计算

### Python API 接口

Python API 基于 PyTorch 扩展机制，将底层 C++ 算子封装为易于使用的 Python 接口：

- **PyTorch 绑定**（`api/`）：
  - `torch_ext.cpp`：使用 `TORCH_LIBRARY` 宏注册算子，通过 `TecoExtension` 编译为独立 C++ 扩展模块
  - `tecoops/__init__.py`：Python 包入口，导出算子接口

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

详见 [Python API 接口说明](doc/README_PYTHON.md)。

### Plugin 自定义算子接口

Plugin 自定义算子接口基于 Teco-Inference 推理框架，支持通过 TVM Relay IR 注册自定义算子并在推理引擎中执行。适用于推理场景中的自定义算子需求。

- **Plugin 算子实现**（`teco/plugin/`）：继承 `AbstractPluginOp` 基类，实现 `InferOutputShape` 和 `Enqueue` 方法，编译产物为 `libteco_ops_plugin.so` 和 `libTecoInferPlugin.so`
- **Plugin 算子测试**（`plugin_test/`）：基于 ONNX + TVM Relay 的推理测试脚本

详见 [Plugin 自定义算子接口说明](doc/README_PLUGIN.md)。

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
│   ├── plugin/              # Plugin 自定义算子实现（基于 Teco-Inference）
│   │   └── pluginFlattenRays/
│   │       └── plugin_flatten_rays.cc
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
├── plugin_test/            # Plugin 自定义算子推理测试脚本
├── examples/               # 示例脚本
├── doc/                    # 文档
│   ├── README_OP.md        # 算子开发指南
│   ├── README_PYTHON.md    # Python 接口说明
│   ├── README_PLUGIN.md    # Plugin 自定义算子接口说明
│   ├── README_DEBUG.md    # Debug手册
│   └── QA.md               # 常见问题解答
├── build.sh                # 算子库构建脚本
├── setup.py                # Python 绑定构建脚本
├── requirements.txt
└── README.md
```

## 快速开始

### 步骤一：Fork 仓库

将本仓库 Fork 到您的个人空间，点击仓库页面右上方的 Fork 按钮即可。详情可查阅 [GitHub Fork 文档](https://docs.github.com/en/get-started/quickstart/fork-a-repo)。

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
- 新增文件需参考已有文件，在文件头添加 [BSD License](LICENSE)
- 编码统一使用 [Google C++ 风格](https://zh-google-styleguide.readthedocs.io/en/latest/google-cpp-styleguide/contents.html)
- SPM 内存申请不超过 235KB

### 步骤三：C++ 接口算子自测

使用项目内置的 C++ 测试框架进行算子精度和性能验证：

```bash
cd test
source env.sh

# 构建所有算子测试
sh build.sh --arch teco

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
```

**注意：** 使用 torch 扩展时，需先 `import torch` 再 `import tecoops`。

### 步骤六：Plugin 自定义算子开发与自测

如需在推理场景中使用自定义算子，可在 `teco/plugin/` 目录下开发 Plugin 算子，并在 `plugin_test/` 目录下编写测试脚本。详见 [Plugin 自定义算子接口说明](doc/README_PLUGIN.md)。

```bash
# 设置 TVM 路径环境变量
export TECO_INFER_PLUGIN_UTIL_PATH=<tvm_package_path>

# 构建（setup.py 自动执行两阶段构建）
python setup.py build_ext --inplace

# 测试前设置库路径（需包含 libteco_ops_plugin.so 和 libTecoInferPlugin.so 所在目录）
export LD_LIBRARY_PATH=<lib_path>:${LD_LIBRARY_PATH}

# 测试 Plugin 算子
python plugin_test/test_plugin_flatten_rays.py
```

### 步骤七：提交 PR

完成开发和自测后，提交 Pull Request。详细规范见 [算子提交规范](doc/PR.md)。

## 注意事项

- 非代码说明的注释代码，请删除（例如开发过程中的功能调试、打印代码等）
- PR 中的新功能不能破坏原有功能，需要兼容原有功能，只能新增代码，不能删除原有代码
- SPM 空间申请时，不要超过 235KB。推荐使用仓库封装的 `rt_spm_malloc()` 与 `rt_spm_free()` 等接口（上限 240512B）
- 提交 PR 前，请确保本地自测通过

## 文档

- [算子提交规范](doc/PR.md) — Commit 消息格式、PR 规范、代码风格检查
- [算子开发指南](doc/README_OP.md) — 算子实现、测例编写及测试步骤
- [算子设计文档](doc/op_docs/) — 各算子的设计文档
- [Python 接口说明](doc/README_PYTHON.md) — Python API 使用指南
- [Plugin 自定义算子接口说明](doc/README_PLUGIN.md) — Plugin 算子开发与推理使用指南
- [常见问题](doc/QA.md) — 算子 proto 参数设置及测试框架说明
- [算子硬件知识](doc/teco-ops-hardware.md) — 算子开发相关的硬件知识
- [Debug 手册](doc/README_DEBUG.md) — SDAAC 程序调试手段及精度问题排查

## License

请参考项目根目录的 [LICENSE](LICENSE) 文件。
