#include <stdio.h>
#include <iostream>
#include <string>
#include <tensor.h>
#include "zoo/teco/convert.h"
#include "common/time.hpp"

#include "zoo/teco/flatten_rays/flatten_rays.h"
#include "interface/include/tecoops.h"


namespace optest {

void FlattenRaysExecutor::destroy() {
}

void FlattenRaysExecutor::paramCheck() {
}

void FlattenRaysExecutor::paramParse() {
    // rays shape: [N, 2], where each row is (offset, num_steps)
    N = parser_->inputs()[0].shape[0];
    M = parser_->outputs()[0].shape[0];
}

void FlattenRaysExecutor::paramGeneration() {
    rays = dev_input[0];
    res = dev_output[0];
}

void FlattenRaysExecutor::compute() {
#ifdef USE_TECO
    // static tecoopsHandle_t handle = nullptr;
    // if (handle == nullptr) {
    //     tecoopsCreate(&handle);
    // }
    checkTECOOPS(tecoopsFlattenRays(handle_,
                       static_cast<const int*>(rays), N, M,
                       static_cast<int*>(res),
                       TECOOPS_ALGO_0));
#endif
}

int64_t FlattenRaysExecutor::getTheoryOps() {
    int64_t theory_ops = M;
    return theory_ops;
}

int64_t FlattenRaysExecutor::getTheoryIoSize() {
    int64_t theoryIo_ops = N * 2 + M;
    return theoryIo_ops;
}

void FlattenRaysExecutor::cpuCompute() {
    auto* rays_ptr = static_cast<int*>(baseline_input[0]);
    auto* res_ptr = static_cast<int*>(baseline_output[0]);

    // rays is [N, 2], each row is (offset, num_steps)
    // For each ray i, fill res[offset:offset+num_steps] with i
    for (uint32_t i = 0; i < N; ++i) {
        int offset = rays_ptr[i * 2];
        int num_steps = rays_ptr[i * 2 + 1];
        for (int j = 0; j < num_steps; ++j) {
            res_ptr[offset + j] = static_cast<int>(i);
        }
    }
}

void FlattenRaysExecutor::gpuCompute() {
    // No GPU compute needed for this test
}

}  // namespace optest
