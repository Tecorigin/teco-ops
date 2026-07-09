# Teco-Ops PyTorch 扩展绑定

Teco-Ops 算子的 PyTorch 扩展绑定，提供基于 PyTorch 接口的高性能算子。

## 添加新算子绑定

在 `api/torch_ext.cpp` 中添加新算子的 PyTorch 绑定，步骤如下：

### 1. 创建包装函数

将 PyTorch Tensor 转换为 interface 层参数，调用 C API：

```cpp
void my_op_torch(torch::Tensor input, torch::Tensor output, int param) {
    tecoopsHandle_t handle = getGlobalHandle();
    tecoopsMyOp(handle,
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                param,
                TECOOPS_ALGO_0);
}
```

### 2. 注册算子

在 PYBIND11_MODULE 中添加算子绑定：
```cpp
m.def("my_op", &my_op_torch, "my_op (SDAA)");
```

### 3. 暴露 Python 接口

在 Python 包中导出算子：
```python
from _torch_ext import flatten_rays, my_op
```

### 完整示例

```cpp
#include <torch/extension.h>
#include <torch_sdaa/sdaa_extension.h>

#include "interface/include/tecoops.h"
#include "interface/common/handle.h"

static tecoopsHandle_t g_handle = nullptr;

static tecoopsHandle_t getGlobalHandle() {
    if (g_handle == nullptr) {
        tecoopsCreate(&g_handle);
    }
    return g_handle;
}

void my_op_torch(torch::Tensor input, torch::Tensor output, int param) {
    tecoopsHandle_t handle = getGlobalHandle();
    tecoopsMyOp(handle,
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                param,
                TECOOPS_ALGO_0);
}

PYBIND11_MODULE(_torch_ext, m) {
    m.def("flatten_rays", &flatten_rays_torch, "flatten_rays (SDAA)");
    m.def("my_op", &my_op_torch, "my_op (SDAA)");
}
```

## 构建

### 环境要求

环境准备和依赖安装说明：

**必需依赖：**
- Python 3.8+
- torch
- torch-sdaa
- CMake >= 3.24
- tecocc 编译器

### 构建步骤

编译和安装 Python 扩展的命令：

```bash
# 安装依赖
pip install torch torch-sdaa

# 构建并安装（本地开发模式）
WITH_TORCH=ON python setup.py build_ext --inplace

# 构建wheel包
WITH_TORCH=ON python setup.py bdist_wheel
```

**注意：** 使用 torch 扩展时，需要先 `import torch` 再 `import tecoops`，以确保 torch 库路径正确设置。

### 清理构建

清理构建产物和 CMake 缓存的命令：

```bash
# 清理构建文件
python setup.py clean

# 完全清理（包括 CMake 构建目录）
rm -rf build api/tecoops/*.so
```

## 使用示例

### 使用 PyTorch 扩展算子

通过 Python API 调用算子的示例：

```python
import torch
import tecoops

# flatten_rays 算子
rays = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32, device='sdaa')
N, M = rays.shape
res = torch.empty(N * M, dtype=torch.int32, device='sdaa')

tecoops.flatten_rays(rays, N, M, res)
```

### 示例脚本

```bash
# 测试 flatten_rays
python examples/test_flatten_rays.py
```

## 算子列表

| 算子 | 描述 | 接口类型 |
|------|------|----------|
| `flatten_rays` | 光线扁平化处理 | PyTorch Tensor |

**注：**
- **普通 PyTorch 扩展算子**（如 `flatten_rays`）：通过 `TecoExtension` 编译独立的 C++ 扩展模块，算子实现位于 `teco/` 目录下的 interface + ual 分层架构中，通过 CMake 构建为 `libteco_ops.so` 库。Python API 内部通过 interface 层（`tecoopsHandle_t`）调用算子，即 `tecoops.flatten_rays → tecoopsFlattenRays → RUN_OP → ual` 各层。

## 项目结构

```
Teco-Ops/
├── api/                    # Python 绑定代码
│   ├── torch_ext.cpp      # PyTorch 扩展绑定
│   └── tecoops/           # Python 包
│       └── __init__.py
├── teco/                   # TECO 算子实现（interface + ual 分层架构）
│   ├── interface/          # Interface 层：用户 API
│   │   ├── include/        #   tecoops.h（C API 声明）
│   │   ├── ops/            #   接口实现（RUN_OP 分发）
│   │   └── common/         #   handle、convert、宏等
│   ├── ual/                # UAL 层：统一算子库
│   │   ├── args/           #   参数结构体
│   │   ├── ops/            #   Op 类（分支分发）
│   │   ├── kernel/         #   设备端 kernel（.scpp）
│   │   └── com/            #   数据类型、日志等
│   └── CMakeLists.txt
├── examples/               # 示例脚本
├── setup.py               # 构建脚本
└── doc/README_PYTHON.md   # 本文件
```

> **分层架构说明**：Python API 用户接口保持不变（如 `tecoops.flatten_rays`），但内部调用链路为：`tecoops.flatten_rays → torch_ext.cpp → tecoopsFlattenRays(handle, ...) → RUN_OP → ual ops（分支选择） → ual kernel（设备计算）`。这种分层设计使得用户只需关注 interface 层 API，而算子实现细节由 ual 层管理。

## 常见问题

常见问题及解决方案：

### ImportError：libtorch.so：cannot open shared object file

**原因：** torch 库路径未正确设置。

**解决：** 确保先 `import torch` 再 `import tecoops`：

```python
import torch      # 先导入 torch
import tecoops    # 再导入 tecoops
```

### AttributeError：module 'tecoops' has no attribute 'flatten_rays'

**原因：** 未启用 `WITH_TORCH` 构建。

**解决：** 使用 `WITH_TORCH=ON` 重新构建：

```bash
WITH_TORCH=ON python setup.py build_ext --inplace
```

## License

请参考项目根目录的 [LICENSE](LICENSE) 文件。
