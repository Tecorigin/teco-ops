## 简介

algorithm_example文件夹为用户提供了一些简单的算子实现，包括卷积算子、summa算子、gemm算子以及add算子实现。目的是为用户展示如何通过SDAA C编程模型完成一个简单算子的开发。
algorithm_example相较于Teco-Ops仓库提供的比赛示例代码，学习成本低，您仅需要了解[SDAA C编程语言](http://docs.tecorigin.com/release/sdaac/)，使用较少的步骤，即可完成算子的开发和验证。

## 支持的Teco芯片

T1

## 支持的操作系统

Linux

## 支持的CPU架构

x86_64，Loongnix，aarch64，SW_64

## 环境依赖

SDAAC_examples的运行依赖以下组件，可以参考[环境安装手册](http://docs.tecorigin.com/release/software_installation/)进行安装。

- TecoDriver
- TecoToolKit

## 涉及的API接口

算子实现示例是SDAA C编程模型的综合实践，使用到了大部分SDAA C接口，要求用户对SDAA C接口有一定的基础。您可通过以下链接学习SDAA C编程模型：
- [SDAA C编程指南](http://docs.tecorigin.com/release/sdaac/)：介绍SDAA编程模型、语言规范、函数接口、数学函数、程序编译、程序调试及性能调优等内容。**注意：** 本目录下的convolution和summa示例，与《SDAA C编程指南》中的最佳实践章节中的示例基本一致。
- [SDAA C零基础入门](http://docs.tecorigin.com/release/sdaac_beginner_guide/)：介绍算子的基本概念、算子开发的流程、并以加法和矩阵乘为例，介绍如何进行算子开发。**注意：** 本目录下的add和gemm算子示例，与《SDAA C零基础入门》的示例基本一致。

如果您对开发的算子性能有更高的要求，推荐阅读：

- [性能优化手册-SDAAC篇](http://docs.tecorigin.com/release/sddac_perf_opt/)：介绍程序并行、函数接口、数学函数、程序编译过程中的性能优化方法。
- [性能优化手册-算子篇](http://docs.tecorigin.com/release/op_perf_opt/)：介绍计算与访存优化方法。计算相关优化包括向量指令、指令流水线、矩阵乘法加速单元等优化方法；访存相关优化包括双缓冲、广播等优化方法。

## 编译运行

编译运行方式是在SDAAC_examples目录下执行run_customops.sh，具体步骤如下：

```
bash run_customops.sh algorithm_example
```

## 算子开发步骤
### 步骤一：创建文件夹

1. 在algorithm_example目录下，创建文件夹，用于存放您的算子代码和编译运行脚本。例如，您要开发的是sigmoid算法，可以创建一个名称为sigmoid的文件夹。
2. 复制示例目录add下的makefile编译脚本，到新创建的文件夹下。

### 步骤二：开发算子

1. 在步骤一创建的目录下，创建后缀为.scpp的文件，例如：sigmoid.scpp。

2. 根据需要实现的算法功能开发算子，可以参考[SDAA C编程指南](http://docs.tecorigin.com/release/sdaac/)、[SDAA C零基础入门](http://docs.tecorigin.com/release/sdaac_beginner_guide/)进行算子开发。

   **注意:** 算子代码和测试代码均需开发。

### 步骤三：编译运行

编译运行方式是在SDAAC_examples目录下执行run_customops.sh，具体步骤如下：

```
bash run_customops.sh algorithm_example
```
**执行结果**

系统根据您测试代码的运行情况，打印测试结果。

### 步骤四：提交PR

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
