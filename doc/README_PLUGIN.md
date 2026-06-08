# Teco-Ops Plugin 自定义算子接口

Plugin 自定义算子接口基于 Teco-Inference 推理框架，支持用户通过 TVM Relay IR 注册自定义算子，并在推理引擎中执行。适用于需要在推理场景中使用自定义算子的场景。

## 架构概述

Plugin 算子的工作流程如下：

1. **C++ 端**：在 `teco/plugin/<OpName>/` 目录下实现 `AbstractPluginOp` 子类，注册算子实现和规格
2. **Python 端**：使用 `tvm.plugin.plugins.register_op()` 注册算子元信息，构建 ONNX 模型并转换为 Relay IR
3. **推理执行**：通过 `tecoinference.Engine` 加载编译后的模型，执行推理

编译产物为 `libteco_ops_plugin.so`（算子核心库）和 `libTecoInferPlugin.so`（Plugin 算子库，链接 `libteco_ops_plugin.so`），与普通算子库 `libteco_ops.so` 分离构建。

## 环境要求

**必需依赖：**
- Python 3.8+
- TVM（含 Teco-Inference 扩展）
- tecoinference
- tvm.contrib.teco_infer_dyn
- onnx 1.13.0
- numpy

**构建依赖：**
- g++（支持 C++17）
- CMake >= 3.10.2
- tecocc 编译器
- libsdaart.so
- libTecoInferPluginUtil.so

## 开发流程

### 1. 定义属性结构体

在 `teco/plugin/<OpName>/plugin_<op_name>.cc` 中，定义算子属性结构体，继承 `tvm::AttrsNode`：

```cpp
struct plugin_flatten_raysAttrs
    : public tvm::AttrsNode<plugin_flatten_raysAttrs> {
  int64_t M;
  TVM_DECLARE_ATTRS(plugin_flatten_raysAttrs,
                     "relay.attrs.plugin_flatten_raysAttrs") {
    TVM_ATTR_FIELD(M).set_default(0).describe("Total number of steps");
  }
};
TVM_REGISTER_NODE_TYPE(plugin_flatten_raysAttrs);
```

**说明：**
- 属性结构体命名规则：`plugin_<op_name>Attrs`
- `TVM_DECLARE_ATTRS` 的参数为结构体类型名和全局注册名（格式为 `relay.attrs.plugin_<op_name>Attrs`）
- 使用 `TVM_ATTR_FIELD` 声明每个属性字段，可设置默认值和描述
- 必须调用 `TVM_REGISTER_NODE_TYPE` 注册节点类型

### 2. 实现 AbstractPluginOp 子类

继承 `AbstractPluginOp` 基类，实现两个核心方法：

```cpp
class PluginFlattenRaysImpl : public AbstractPluginOp {
 public:
  PluginFlattenRaysImpl() = default;

  void InferOutputShape(
      const std::vector<std::vector<int>>& total_input_shape, int n_input,
      std::vector<std::vector<int>>& total_output_shape, int n_output,
      const OpAttr& attr) {
    int64_t M = attr.GetAttr<int64_t>("M");
    total_output_shape[0] = {static_cast<int>(M)};
  }

  void Enqueue(std::shared_ptr<ComputeContext>& ctx) {
    void* rays_dev = ctx->GetInputDataPtr("rays");
    void* res_dev = ctx->GetOutputDataPtr(0);

    int64_t M;
    ctx->GetAttr("M", M);

    std::vector<int> rays_shape;
    ctx->GetInputShape("rays", rays_shape);
    uint32_t N = rays_shape[0];

    sdaaStream_t stream = ctx->GetStream();

    tecoopsHandle_t handle;
    tecoopsCreate(&handle);
    tecoopsSetStream(handle, stream);

    tecoopsFlattenRays(handle,
                       static_cast<const int*>(rays_dev),
                       N,
                       static_cast<uint32_t>(M),
                       static_cast<int*>(res_dev),
                       TECOOPS_ALGO_0);

    tecoopsDestroy(handle);
  }
};
```

#### InferOutputShape 方法

根据输入形状和属性推导输出形状：

- **参数**：
  - `total_input_shape`：所有输入张量的形状
  - `n_input`：输入数量
  - `total_output_shape`：输出形状数组（需填充）
  - `n_output`：输出数量
  - `attr`：算子属性，通过 `attr.GetAttr<T>("name")` 获取属性值

#### Enqueue 方法

执行实际计算，通过 `ComputeContext` 获取运行时信息：

- **获取输入数据指针**：`ctx->GetInputDataPtr("input_name")` — 按名称获取
- **获取输出数据指针**：`ctx->GetOutputDataPtr(index)` — 按索引获取
- **获取属性值**：`ctx->GetAttr("name", value)` — 获取属性并写入变量
- **获取输入形状**：`ctx->GetInputShape("input_name", shape_vec)` — 获取指定输入的形状
- **获取设备流**：`ctx->GetStream()` — 获取 sdaaStream_t

### 3. 注册算子

使用两个宏完成算子注册：

```cpp
REGISTER_PLUGIN_OP_IMPL(plugin_flatten_rays, PluginFlattenRaysImpl)

PLUGIN_REGISTER_OP("plugin_flatten_rays")
    .Input("rays")
    .Type("Tensor")
    .Desc("Rays input with shape [N, 2]")
    .AttrType<plugin_flatten_raysAttrs>()
    .Register();
```

**说明：**
- `REGISTER_PLUGIN_OP_IMPL`：注册算子实现类，将算子名与实现类绑定
- `PLUGIN_REGISTER_OP`：注册算子规格，声明输入（名称、类型、描述）和属性类型
- 所有注册代码需放在 `TECO_INFER` 命名空间内

### 4. 必需的头文件

```cpp
#include <plugin/register_op.h>
#include <sdaa_runtime.h>
#include <memory>
#include <string>
#include <vector>
#include "interface/include/tecoops.h"
```

## 构建步骤

Plugin 算子库通过 `setup.py` 构建，启用 `WITH_INFERENCE_PLUGIN=ON` 时产出两个库：

1. **`libteco_ops_plugin.so`**：由 `tecocc` 编译 `ops_objs` 生成，包含算子核心实现
2. **`libTecoInferPlugin.so`**：由 `gcc` 编译 `PLUGIN_OBJECTS` 并链接 `libteco_ops_plugin.so` + `PLUGIN_LIBS` 生成，包含 Plugin 算子实现

```bash
# 设置 TVM 路径环境变量（必需）
export TECO_INFER_PLUGIN_UTIL_PATH=<tvm_package_path>

# 构建（setup.py 自动构建两个库）
WITH_TORCH=ON python setup.py build_ext --inplace
```

构建完成后，`libteco_ops_plugin.so` 和 `libTecoInferPlugin.so` 会自动复制到 `api/tecoops/` 目录。

### CMake 构建细节

`teco/CMakeLists.txt` 中 Plugin 相关的构建配置：

- Plugin 源码使用 `g++` 编译（而非 `tecocc`），因为需要兼容 TVM C++ 头文件
- 编译选项：`-O3 -std=c++17 -fPIC -DWITH_INFERENCE_PLUGIN`
- 构建分两个目标：
  - `ops_lib`：`tecocc` 编译 `ops_objs` 生成 `libteco_ops_plugin.so`
  - `plugin_lib`：`gcc` 编译 `PLUGIN_OBJECTS`，链接 `-lteco_ops_plugin` + `PLUGIN_LIBS`（`libsdaart.so`、`libTecoInferPluginUtil.so`）生成 `libTecoInferPlugin.so`
- `plugin_lib` 依赖 `ops_lib`（确保 `libteco_ops_plugin.so` 先构建完成）和 `plugin_compile`

## Python 端使用

### 1. 注册算子

在 Python 端使用 `tvm.plugin.plugins.register_op()` 注册算子元信息：

```python
from tvm.plugin import plugins

plugins.register_op(
    op_name="plugin_flatten_rays",
    inputs=["rays"],
    attrs={"M": "int"}
)
```

**参数说明：**
- `op_name`：算子名称，需与 C++ 端 `PLUGIN_REGISTER_OP` 注册的名称一致
- `inputs`：输入名称列表，需与 C++ 端 `.Input()` 注册的名称一致
- `attrs`：属性字典，键为属性名，值为类型字符串

### 2. 构建 ONNX 模型

使用 ONNX helper 构建包含自定义算子的模型：

```python
from onnx import helper, TensorProto

def create_plugin_onnx_model(op_type, input_shapes, output_shape, attributes=None):
    input_names = list(input_shapes.keys())
    inputs = [
        helper.make_tensor_value_info(name, TensorProto.INT32, shape)
        for name, shape in zip(input_names, input_shapes)
    ]
    output_name = "output"
    output = helper.make_tensor_value_info(output_name, TensorProto.INT32, output_shape)

    attributes = attributes or {}
    node = helper.make_node(
        op_type,
        input_names,
        [output_name],
        **attributes,
        domain="my_custom_ops",
        version=1
    )

    graph = helper.make_graph([node], op_type, inputs, [output])
    model = helper.make_model(graph)
    model.opset_import.append(helper.make_opsetid("my_custom_ops", 1))
    return model
```

**关键点：**
- 自定义算子节点必须指定 `domain="my_custom_ops"`
- 模型需添加对应的 opset import：`helper.make_opsetid("my_custom_ops", 1)`
- 属性值通过 `**attributes` 传入节点

### 3. 转换为 Relay IR 并编译

```python
import tvm
from tvm import relay
from tvm.contrib.teco_infer_dyn import dyn
import tecoinference

model = create_plugin_onnx_model(
    op_type="plugin_flatten_rays",
    input_shapes={"rays": (N, 2)},
    output_shape=(M,),
    attributes={"M": M}
)

mod, params = tvm.relay.frontend.from_onnx(model, {"rays": (N, 2)})

fbs_model = dyn.to_teco_infer_dyn(mod, {}, "teco_dyn")
engine = tecoinference.Engine(fbs_model)
```

### 4. 执行推理

```python
ctx = engine.create_context()

ctx.set_input(0, input_data)
ctx.executor_run()
output = ctx.get_output(0)
```

**接口说明：**
- `engine.create_context()`：创建推理上下文
- `ctx.set_input(index, data)`：设置输入数据（按索引或名称）
- `ctx.executor_run()`：执行推理
- `ctx.get_output(index)`：获取输出数据（按索引）

## 完整测试示例

以 `plugin_flatten_rays` 为例，完整测试脚本参考 `plugin_test/test_plugin_flatten_rays.py`：

```python
import numpy as np
from tvm.plugin import plugins
import tecoinference
from tvm.contrib.teco_infer_dyn import dyn
from onnx import helper, TensorProto
import tvm
from tvm import relay

plugins.register_op(
    op_name="plugin_flatten_rays",
    inputs=["rays"],
    attrs={"M": "int"}
)

rays_data = np.array([[0, 3], [3, 2], [5, 4]], dtype=np.int32)
N = 3
M = 9

model = create_plugin_onnx_model(
    op_type="plugin_flatten_rays",
    input_shapes={"rays": (N, 2)},
    output_shape=(M,),
    attributes={"M": M}
)

mod, params = tvm.relay.frontend.from_onnx(model, {"rays": (N, 2)})

fbs_model = dyn.to_teco_infer_dyn(mod, {}, "teco_dyn")
engine = tecoinference.Engine(fbs_model)
ctx = engine.create_context()

ctx.set_input(0, rays_data)
ctx.executor_run()
out = ctx.get_output(0)

expected = compute_expected_flatten_rays(rays_data, M)
assert np.array_equal(out, expected)
```

运行测试：

```bash
python plugin_test/test_plugin_flatten_rays.py
```

## 目录结构

```
teco/plugin/
└── pluginFlattenRays/              # Plugin 算子目录（目录名与算子名对应）
    └── plugin_flatten_rays.cc      # 算子实现文件

plugin_test/
└── test_plugin_flatten_rays.py     # Plugin 算子推理测试脚本
```

**命名规则：**
- 算子目录名：`<OpName>`（首字母大写的驼峰命名）
- 算子实现文件：`plugin_<op_name>.cc`（全小写，下划线分隔）
- 测试脚本：`test_plugin_<op_name>.py`

## ComputeContext 接口参考

| 方法 | 说明 |
|------|------|
| `GetInputDataPtr(name)` | 按名称获取输入数据指针 |
| `GetOutputDataPtr(index)` | 按索引获取输出数据指针 |
| `GetAttr(name, value)` | 获取属性值，写入 `value` |
| `GetInputShape(name, shape_vec)` | 获取指定输入的形状 |
| `GetStream()` | 获取 sdaaStream_t 设备流 |

## OpAttr 接口参考

| 方法 | 说明 |
|------|------|
| `GetAttr<T>(name)` | 获取指定名称的属性值，类型为 `T` |

## 动态库路径

编译完动态库路径在api/tecoops目录下。如果编译成wheel包安装后，动态库在tecoops安装路径下。使用前确保动态库的依赖都能够被找到，可通过ldd命令查看。

## 常见问题

### 构建失败：找不到 libTecoInferPluginUtil.so

**原因：** 未设置 `TECO_INFER_PLUGIN_UTIL_PATH` 环境变量。

**解决：** 设置 TVM 包路径：

```bash
export TECO_INFER_PLUGIN_UTIL_PATH=<tvm_package_path>
```

### ImportError：找不到 libTecoInferPlugin.so 或 libteco_ops_plugin.so

**原因：** Plugin 算子库未正确构建或未复制到 `api/tecoops/` 目录，或运行时 `LD_LIBRARY_PATH` 未包含库路径。

**解决：** 重新构建并设置库路径：

```bash
WITH_TORCH=ON python setup.py build_ext --inplace

# 设置库路径（需包含 libteco_ops_plugin.so 和 libTecoInferPlugin.so 所在目录）
export LD_LIBRARY_PATH=<lib_path>:${LD_LIBRARY_PATH}
```

### ONNX 模型转换失败

**原因：** 自定义算子未在 Python 端注册，或 ONNX 模型缺少 `my_custom_ops` opset。

**解决：** 确保：
1. 在 `from_onnx` 之前调用 `plugins.register_op()`
2. ONNX 模型节点指定了 `domain="my_custom_ops"`
3. 模型添加了 `helper.make_opsetid("my_custom_ops", 1)`

## License

请参考项目根目录的 LICENSE 文件。
