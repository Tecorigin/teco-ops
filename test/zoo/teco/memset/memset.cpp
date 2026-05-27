#include <stdio.h>
#include <iostream>
#include <string>
#include "zoo/teco/convert.h"
#include "common/time.hpp"
#include "zoo/teco/memset/memset.h"
#include "interface/include/tecoops.h"

namespace optest {

void MemsetExecutor::paramCheck() {
    if (parser_->inputs().size() != 1) {
        ALLOG(ERROR) << "input num is wrong.";
        throw std::invalid_argument(std::string(__FILE__) + ":" + std::to_string(__LINE__)); // NOLINT
    }

    if (parser_->outputs().size() != 0) {
        ALLOG(ERROR) << "output num is wrong.";
        throw std::invalid_argument(std::string(__FILE__) + ":" + std::to_string(__LINE__));  // NOLINT
    }
}

void MemsetExecutor::paramParse() {
    auto memset_param = parser_->getProtoNode()->tecokernel_param().memset_param();
    value_ = memset_param.value();
    size_ = memset_param.size();
}

void MemsetExecutor::paramGeneration() { x_ = dev_input[0]; }

void MemsetExecutor::compute() { checkTECOOPS(tecoopsMemset(handle_, x_, value_, size_)); }

int64_t MemsetExecutor::getTheoryOps() {
    int64_t theory_ops = parser_->input(0)->shape_count;
    return theory_ops;
}

int64_t MemsetExecutor::getTheoryIoSize() { return (int64_t)size_; }

void MemsetExecutor::cpuCompute() { memset(baseline_input[0], value_, size_); }

}  // namespace optest
