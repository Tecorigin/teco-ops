## Q：如何补充算子 proto 的参数设置

算子的 proto 参数设置，是根据测试代码中实际需要进行配置的，一般是根据算子原型进行设置。

以 flatten\_rays 为例，说明补充思路。没有固定的步骤，这里给出参数选择的思路。

**1）测试代码是从 prototxt 中获取接口所需要的变量，因此需要先了解测例（prototxt）的组成**

prototxt 最主要的两个部分：张量（input/output）+ 接口参数，如下图所示：

![example.png](./prototxt-example.png)

对于张量：

张量的 proto 设置每个算子是通用的，因此，不用再额外编写（test/test\_proto/tensor.proto）。

在测试框架中，会自动解析并保存成 MetaTensor 类（test/src/case/parser.h），并分配对应的空间，保存在 executor 类中（test/src/suite/executor.h），因此，对于接口中的张量指针就可以从这里获取。同时关于张量的 shape/dtype 等也可以从张量参数中获取。

对于接口参数：

接口中，除了传入张量的地址和 shape 外，还有一些其他的参数，对于这部分参数，每个接口都不一样，因此，需要单独设置 proto，存放路径：test/test\_proto/tecokernel。总的来说，这部分是为了获取接口所需的其他参数，如果接口中，只有张量的传入，没有其他参数的传入，那么接口参数这部分都可以不用设置。

**2）了解接口的 proto 用处，那么就需要知道接口原型，从而知道哪些参数是需要的**

flatten\_rays 算子原型：

```shell
tecoopsStatus_t tecoopsFlattenRays(
    tecoopsHandle_t handle,
    const int *rays, uint32_t N, uint32_t M, int *res,
    tecoopsAlgo_t algo);
```

从原型可以看到，接口的变量共 6 个，分析如下：

|  变量名  |  分析  |  是否需要  |
| --- | --- | --- |
|  handle  |  tecoops 句柄，自动管理  |  否  |
|  rays  |  张量指针，自动获取  |  否  |
|  N  |  rays 数量，可以从 prototxt 中的张量参数获取  |  否  |
|  M  |  总步数，可以从 prototxt 中的张量参数获取  |  否  |
|  res  |  张量指针，自动获取  |  否  |
|  algo  |  算法类型，使用默认值  |  否  |

（这里给的是思路，测试框架中只给了一个 demo，和这里有些出入）


## Q：prototxt 的参数和测试代码中 input/output 的关系

这里讲解下 prototxt 中的参数和测试代码的关系。

以 flatten\_rays 这个算子的测试举例，对应的一个 prototxt 如下：
![example.png](./prototxt-example.png)

在测试框架中，和测试代码相关的，比较重要的类有三个：Executor、Parser、Meta 三个类，他们的关系如下图所示。

```mermaid
classDiagram
    class MetaTensor {
        -testpt::DataType dtype
        -testpt::TensorLayout layout
        -testpt::TensorType ttype
        -std::vector<int> shape
        -std::vector<int> stride
    }

    class Parser {
        -testpt::Node *proto_node_
        -std::vector<MetaTensor> inputs_
        -std::vector<MetaTensor> outputs_
        +testpt::Node *node()
        +const std::vector<MetaTensor>& inputs()
        +const std::vector<MetaTensor>& outputs()
    }

    class Executor {
        #std::shared_ptr<Parser> parser_
        #std::vector<void *> workspace_
        #std::vector<void *> dev_input
        #std::vector<void *> dev_output
        #std::vector<void *> host_input
        #std::vector<void *> host_output
        #std::vector<void *> baseline_input
        #std::vector<void *> baseline_output
        #std::vector<void *> gpu_input
        #std::vector<void *> gpu_output
    }

    class TecoExecutor {
    }

    class FlattenRaysExecutor {
    }

    Parser "1" --> "*" MetaTensor: contains
    Executor "1" --> "1" Parser: parser_
    TecoExecutor --|> Executor
    FlattenRaysExecutor --|> TecoExecutor
```

**Executor 类：**

```shell
class Executor {
public:
        std::shared_ptr parser_
        std::vector<void *> dev_input
        std::vector<void *> dev_output
        std::vector<void *> host_input
        std::vector<void *> host_output
        std::vector<void *> baseline_input
        std::vector<void *> baseline_output
        std::vector<void *> gpu_input
        std::vector<void *> gpu_output
    }
```

FlattenRaysExecutor 继承 TecoExecutor，继承 Executor 类。Executor 类是测试框架中，算子的测试执行类，因此很多公共的属性都在这个类中，在写测试代码中，最常用的属性上述代码已经列出来了，说明如下：

- parser\_：存储测例解析结果的变量
- dev\_input：根据测例中 input 的参数分配的设备地址空间，根据测例中 input 的顺序存放，使用 dev\_input\[i\] 可以获取第 i 个 input 的设备地址。（设备地址空间，只能 global、device 函数访问）
- dev\_output：根据测例中 output 的参数分配的设备地址空间，根据测例中 output 的顺序存放，使用 dev\_output\[i\] 可以获取第 output 的设备地址。（设备地址空间，只能 global、device 函数访问）
- host\_input：根据测例中 input 的参数分配的 host 地址空间，使用同 dev\_input，只不过对应的是 host 的地址（尽量不要改变，测试框架中，会使用 memcpy 同步 dev\_input 和 host\_input 的数据）
- dev\_output：根据测例中 output 的参数分配的 host 地址空间，使用同 dev\_output，只不过对应的是 host 的地址（尽量不要改变，测试框架中，会使用 memcpy 同步 dev\_output 和 host\_output 的数据）
- baseline\_input：根据测例中 input 的参数分配的 host 的 baseline 地址空间，**测试框架中使用，不要去改变**。在跑 teco 代码的时候，会从 baseline 文件中，读取对应的 baseline 结果。
- baseline\_output：根据测例中 output 的参数分配的 host 的 baseline 地址空间，**测试框架中使用，不要去改变**。在跑 teco 代码的时候，会从 baseline 文件中，读取对应的 baseline 结果。

对于上面的属性，在扩充算子测试代码时，一般使用 parser\_ 获取对应的参数，然后将 dev\_input/dev\_output 传到调用的接口中。（对于 host\_input/host\_output/baseline\_input/baseline\_output 不需要去改变）。

prototxt 的参数获取，可以根据 prototxt 参数的结构，获取，代码示例如下：

```shell
// flatten_rays 算子无需额外参数，rays 和 res 通过 parser_->inputs()/outputs() 获取
N = parser_->inputs()[0].shape[0];
M = parser_->outputs()[0].shape[0];
```

**Parser 类：**

```shell
class Parser {
 public:
    inline testpt::Node *node() { return proto_node_; }
    inline const std::vector<MetaTensor> &inputs() { return inputs_; }
    inline const std::vector<MetaTensor> &outputs() { return outputs_; }

 private:
    testpt::Node *proto_node_ = nullptr;
    std::vector<MetaTensor> inputs_;
    std::vector<MetaTensor> outputs_;
};
```

Parser 类是用于保存解析测例后的结果，比较重要的属性如上代码，解释如下：

- node()：根据 proto 的配置解析 prototxt 文件，因此，proto 的支持的函数，node() 都可以使用。最常用的就是获取参数
- inputs()：根据 prototxt 中的 input 参数解析成测试框架公共的 MetaTensor 类，方便测试框架使用。属性和方法可以见 MetaTensor 类
- outputs()：根据 prototxt 中的 input 参数解析成测试框架公共的 MetaTensor 类，方便测试框架使用。属性和方法可以见 MetaTensor 类

对于 protoxt 中的参数，都可以使用 node() 方法一层一层获取，具体可以参考 executor 上面给出的方法。

**MetaTensor 类：**

```shell
struct MetaTensor {
public
    testpt::DataType dtype = testpt::DTYPE_INVALID;
    testpt::TensorLayout layout = testpt::LAYOUT_ARRAY;
    testpt::TensorType ttype = testpt::TENSOR;
    std::vector<int> shape;
    std::vector<int> stride;
};
```

MetaTensor 是解析 prototxt 中的 Input/output 转换的类，对应 proto 配置中的 tensor 变量。该类中，保存基本的一些变量，如果额外给 tensor 加变量，可能会识别不到，基本的属性如下：

- dtype：tensor 对应的数据类型
- layout：tensor 对应的存储格式（一般都是 array）
- ttype：tensor 对应的存储类型（一些特殊的库会使用，该库不会使用）
- shape：tensor 对应的形状
- stride：tensor 对应的跨步

（其他的方法，可以见代码）
