// BSD 3-Clause License
//
// Copyright (c) 2024, Tecorigin Co., Ltd.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <stdio.h>
#include <iostream>
#include <string>
#include <tensor.h>
#include "zoo/teco/convert.h"
#include "common/time.hpp"

#include "zoo/teco/morton3D_invert/morton3D_invert.h"


namespace optest {

void Morton3dInvertExecutor::destroy() {
}

void Morton3dInvertExecutor::paramCheck() {
}

void Morton3dInvertExecutor::paramParse() {
    // indices shape: [N], coords shape: [N, 3]
    N = parser_->inputs()[0].shape[0];
}

void Morton3dInvertExecutor::paramGeneration() {
    indices = dev_input[0];
    coords = dev_output[0];
}

void Morton3dInvertExecutor::compute() {
#ifdef USE_TECO
    checkTECOOPS(tecoopsMorton3DInvert(handle_,
                       static_cast<const int*>(indices), N,
                       static_cast<int*>(coords)));
#endif
}

int64_t Morton3dInvertExecutor::getTheoryOps() {
    int64_t theory_ops = 3 * N;
    return theory_ops;
}

int64_t Morton3dInvertExecutor::getTheoryIoSize() {
    int64_t theoryIo_ops = N * 4;
    return theoryIo_ops;
}

void Morton3dInvertExecutor::cpuCompute() {
    auto* indices_ptr = static_cast<int*>(baseline_input[0]);
    auto* coords_ptr = static_cast<int*>(baseline_output[0]);

    // For each index i, compute the Morton inverse and store in coords
    auto morton3D_invert = [](uint32_t x) -> uint32_t {
        x = x & 0x49249249;
        x = (x | (x >> 2)) & 0xc30c30c3;
        x = (x | (x >> 4)) & 0x0f00f00f;
        x = (x | (x >> 8)) & 0xff0000ff;
        x = (x | (x >> 16)) & 0x0000ffff;
        return x;
    };
    for (uint32_t i = 0; i < N; ++i) {
        int ind = indices_ptr[i];
        coords_ptr[i * 3 + 0] = morton3D_invert(ind >> 0);
        coords_ptr[i * 3 + 1] = morton3D_invert(ind >> 1);
        coords_ptr[i * 3 + 2] = morton3D_invert(ind >> 2);
    }
}


void Morton3dInvertExecutor::gpuCompute() {
    // No GPU compute needed for this test
}

}  // namespace optest
