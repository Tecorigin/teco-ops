## 简介

SDAAC_examples是一个独立的文件夹，目的是介绍SDAA C编程模型接口的使用方法，文件夹内的示例会向您展示各个接口的使用方式。该目录下的AlgorithmExample是算子示例文件夹，目的是指导初次接触SDAA C的开发者学习算子开发，而SDAAC_examples之外的其他文件夹主要是 太初社区 为了算子活动比赛准备的内容，对代码合入有一定规范要求（主要为了比赛的自动化测试等）。

您可以通过以下文档系统性学习SDAA C编程模型：

- [SDAA C编程指南](http://docs.tecorigin.com/release/sdaac/)：介绍SDAA编程模型、语言规范、函数接口、数学函数、程序编译、程序调试及性能调优等内容。

如何进行算子开发，您可以阅读以下内容：

- [SDAA C零基础入门](http://docs.tecorigin.com/release/sdaac_beginner_guide/)：介绍算子的基本概念、算子开发的流程、并以加法和矩阵乘为例，介绍如何进行算子开发。**注意:** 本目录下的add和gemm算子示例，与《SDAA C零基础入门》的示例基本一致。

如果您对开发的算子性能有更高的要求，推荐阅读：

- [性能优化手册-SDAAC篇](http://docs.tecorigin.com/release/sddac_perf_opt/)：介绍程序并行、函数接口、数学函数、程序编译过程中的性能优化方法。
- [性能优化手册-算子篇](http://docs.tecorigin.com/release/op_perf_opt/)：介绍计算与访存优化方法。计算相关优化包括向量指令、指令流水线、矩阵乘法加速单元等优化方法；访存相关优化包括双缓冲、广播等优化方法。

本文档主要介绍SDAAC_examples目录结构，算子的开发流程您可以阅读AlgorithmExample目录下的README.md。


## 目录结构
SDAAC_examples目录结构如下：

```bash
.
├── algorithm_example                             # 算子文件夹
│   ├── add                                       # add 算法示例文件夹
│   ├── convolution                               # convolution 算法示例文件夹
│   ├── gemm                                      # gemm 算法示例文件夹
│   ├── summa                                     # summa 算法示例文件夹
│   └── README.md                                 # README文件，包含文件夹介绍、环境依赖、编译运行方式、算子开发步骤等内容
├── atomic                                        # 原子操作接口示例文件夹
│   └── README.md                                 # README文件，包含文件夹介绍、环境依赖、编译运行方式等内容
├── broadcast                                     # 广播接口示例文件夹
│   └── README.md                                 # README文件，包含文件夹介绍、环境依赖、编译运行方式等内容
├── DMA                                           # DMA数据搬运接口示例文件夹
│   ├── memcpy                                    # 阻塞型接口DMA数据搬运接口示例文件夹
│   ├── memcpy_async                              # 非阻塞型接口DMA数据搬运接口示例文件夹
│   └── README.md                                 # README文件，包含文件夹介绍、环境依赖、编译运行方式等内容
├── malloc_free                                   # SPM内存分配和释放接口示例文件夹
│   └── README.md                                 # README文件，包含文件夹介绍、环境依赖、编译运行方式等内容
├── math_funcs                                    # 数学函数示例文件夹
│   ├── double_scalar                             # 双精度数学函数接口示例文件夹
│   ├── single_scalar                             # 单精度数学函数接口示例文件夹
│   ├── vector                                    # 向量数学函数接口示例文件夹
│   └── README.md                                 # README文件，包含文件夹介绍、环境依赖、编译运行方式等内容
├── matmul                                        # 矩阵乘接口示例文件夹
│   └── README.md                                 # README文件，包含文件夹介绍、环境依赖、编译运行方式等内容
├── RMA                                           # RMA数据搬运接口示例文件夹
│   ├── rma_async                                 # 非阻塞型RMA数据搬运接口示例文件夹
│   │   ├── mix_RmaNormalMode_RmaCustomizeMode    # 非阻塞型RMA数据搬运常规模式与自定义模式混合使用示例文件夹
│   │   ├── RmaCustomizeMode                      # 非阻塞型RMA数据搬自定义模式使用示例文件夹
│   │   │   ├── async_get                         # 非阻塞型RMA数据搬自定义模式下使用get接口获取数据示例文件夹
│   │   │   ├── async_put                         # 非阻塞型RMA数据搬自定义模式下使用put接口发送数据示例文件夹
│   │   │   └── mix_async_get_and_async_put       # 非阻塞型RMA数据搬自定义模式下混合使用get接口与put接口示例文件夹
│   │   └── RmaNormalMode                         # 非阻塞型RMA数据搬常规模式使用示例文件夹
│   │       ├── async_get                         # 非阻塞型RMA数据搬常规模式下使用get接口获取数据示例文件夹
│   │       ├── async_put                         # 非阻塞型RMA数据搬常规模式下使用put接口发送数据示例文件夹
│   │       └── mix_async_get_and_async_put       # 非阻塞型RMA数据搬常规模式下混合使用get接口与put接口示例文件夹
│   └── rma_sync                                  # 阻塞型RMA数据搬运接口示例文件夹
│       ├── sync_get                              # 阻塞型RMA数据搬运get接口示例文件夹
│       └── sync_put                              # 阻塞型RMA数据搬运put接口示例文件夹
├── simd                                          # 向量接口示例文件夹
│   ├── simd_concat                               # simd_concat接口示例文件夹
│   ├── simd_copy_sign                            # simd_copy_sign接口示例文件夹
│   ├── simd_cvt                                  # simd_cvt接口示例文件夹
│   ├── simd_ins                                  # simd_ins接口示例文件夹
│   ├── simd_load                                 # simd_load接口示例文件夹
│   ├── simd_loadu                                # simd_loadu接口示例文件夹
│   ├── simd_seleq                                # simd_seleq接口示例文件夹
│   ├── simd_selle                                # simd_selle接口示例文件夹
│   ├── simd_sellt                                # simd_sellt接口示例文件夹
│   ├── simd_store                                # simd_store接口示例文件夹
│   ├── simd_storeu                               # simd_storeu接口示例文件夹
│   ├── simd_stretch                              # simd_stretch接口示例文件夹
│   └── README.md                                 # README文件，包含文件夹介绍、环境依赖、编译运行方式等内容
├── simple_examples                               # 入门示例文件夹，包含SPMD编程范式示例和printf函数打印
│   └── README.md                                 # README文件，包含文件夹介绍、环境依赖、编译运行方式等内容
├── sync_thread                                   # 计算核心SPE同步接口示例文件夹
│   └── README.md                                 # README文件，包含文件夹介绍、环境依赖、编译运行方式等内容
└── transpose                                     # 转置接口示例文件夹
    ├── async                                     # 非阻塞型转置接口示例文件夹
    ├── sync                                      # 阻塞型转置接口示例文件夹
    └── README.md                                 # README文件，包含文件夹介绍、环境依赖、编译运行方式等内容

```

## 环境依赖

SDAAC_examples的运行依赖以下组件，可以参考[环境安装手册](http://docs.tecorigin.com/release/software_installation/)进行安装。

- TecoDriver
- TecoToolKit

### 步骤一：编译运行

在SDAAC_examples目录下，运行以下命令，编译并运行算子。

```
bash ../env_dev.sh
bash run_customops.sh
```
**执行结果**

系统根据您测试代码的运行情况，打印测试结果。

### 步骤二：提交PR

PR（Pull Request）提交原则：一个PR需要包括一个完整的算子代码或改进代码，详情可以查阅GitHub官方使用文档：[《Creating a pull request》](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests)。

**注意:** PR提交的目标分支是：**tecorigin/teco-ops:SDAAC_examples**

提交PR需要按照以下提示信息，填写commit内容：

- please input standard format for commit: {[type]: <subject> }  
- the type should be one of: 'feat', 'fix', 'perf', 'refactor', 'ci', 'tool', 'docs', 'test'   

例如：

```
git commit -m "[feat]: add new operator sigmoid"
```

## 贡献指南

将算子合入SDAAC_examples目录的主要流程如下：

1. **创建文件夹**：创建用于存放算子代码和编译运行脚本的文件夹，并复制示例算子的编译运行脚本到此文件夹下。
2. **开发算子**：根据需要实现的算法功能开发算子。
3. **编译运行算子**：编译并运行算子，获取算子的计算结果。
4. **提交PR**：提交代码至SDAAC_examples目录。

## 免责声明

- 使用限制：本开源仓库旨在促进生态交流，不得用于非法或未经授权的目的。用户需遵守相关法律法规，不得利用本仓库进行任何违法活动，如发生任何违法情形的，本仓库开发者和贡献者不承担任何法律责任。

- 责任限制：本仓库的开发者和贡献者对使用本仓库的结果不承担任何责任。用户需自行承担使用本仓库所带来的所有风险，包括但不限于财产、性能、安全性、兼容性等方面的问题。

- 知识产权声明：本仓库不侵犯任何第三方知识产权。后续贡献者应自行保证其贡献内容享有相关知识产权并在允许的范围内进行合法的发布、传播和使用，本仓库开发者不负责鉴别或审查。若因侵犯他人知识产权而造成法律责任（包括但不限于民事赔偿和刑事责任），由违约者自行承担。任何用户如发现任何侵权行为，请及时联系我们，我们将尽快删除相关内容。

## 许可认证

Teco-Ops采用The 3-Clause BSD License。具体内容，请参见[LICENSE](../../LICENSE)文件。
