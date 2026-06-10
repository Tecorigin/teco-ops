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

#ifndef ZOO_TECO_MORTON3D_INVERT_MORTON3D_INVERT_H_  // NOLINT
#define ZOO_TECO_MORTON3D_INVERT_MORTON3D_INVERT_H_

#include "interface/include/tecoops.h"
#include "zoo/teco/executor.h"

namespace optest {
class Morton3dInvertExecutor : public TecoExecutor {
 public:
    Morton3dInvertExecutor() {}
    ~Morton3dInvertExecutor() {}

    void paramCheck();
    void paramParse();
    void paramGeneration();
    void compute();
    void cpuCompute();
    void gpuCompute();
    int64_t getTheoryOps() override;
    int64_t getTheoryIoSize() override;
    void destroy();

 private:
    void *indices, *coords;
    uint32_t N;
};

}  // namespace optest

#endif  // ZOO_TECO_MORTON3D_INVERT_MORTON3D_INVERT_H_
