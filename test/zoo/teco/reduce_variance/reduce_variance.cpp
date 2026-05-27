#include <stdio.h>
#include <iostream>
#include <string>
#include "zoo/teco/convert.h"
#include "common/time.hpp"
#include "zoo/teco/reduce_variance/reduce_variance.h"


namespace optest {

void ReduceVarianceExecutor::paramCheck() {
    if (parser_->inputs().size() != 1) {
        ALLOG(ERROR) << "input num is wrong.";
        throw std::invalid_argument(std::string(__FILE__) + ":" + std::to_string(__LINE__));  // NOLINT
    }

    if (parser_->outputs().size() != 1) {
        ALLOG(ERROR) << "output num is wrong.";
        throw std::invalid_argument(std::string(__FILE__) + ":" + std::to_string(__LINE__));  // NOLINT
    }
}

void ReduceVarianceExecutor::paramParse() {
    auto reduce_variance_param = parser_->getProtoNode()->tecokernel_param().reduce_variance_param();
    axis_ = reduce_variance_param.axis();
    correction_ = reduce_variance_param.correction();
}

void ReduceVarianceExecutor::paramGeneration() {
    xDesc_ = getInputDesc<tecoopsTensorDescriptor_t>(0);
    x_ = dev_input[0];
    yDesc_ = getOutputDesc<tecoopsTensorDescriptor_t>(0);
    y_ = dev_output[0];
}

void ReduceVarianceExecutor::compute() {
    checkTECOOPS(tecoopsReduceVariance(handle_, axis_, correction_, xDesc_, x_, yDesc_, y_));
}

int64_t ReduceVarianceExecutor::getTheoryOps() {
    int64_t theory_ops = parser_->input(0)->shape_count * 3;
    return theory_ops;
}

int64_t ReduceVarianceExecutor::getTheoryIoSize() { return getIoSize(); }

void ReduceVarianceExecutor::cpuCompute() { pythonComputeCPU("cpu"); }

void ReduceVarianceExecutor::gpuCompute() { /*pythonComputeGPU("cuda"); */ }

}  // namespace optest
