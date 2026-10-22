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

#ifndef ZOO_TECO_MS_DEFORM_ATTN_BACKWARD_H_
#define ZOO_TECO_MS_DEFORM_ATTN_BACKWARD_H_
#include "zoo/teco/executor.h"

namespace optest {
class MsDeformAttnBackwardExecutor : public TecoExecutor {
 public:
    ~MsDeformAttnBackwardExecutor() override;
    void destroy() override;
    void paramCheck();
    void paramParse();
    void paramGeneration();
    void compute();
    void cpuCompute();
    int64_t getTheoryOps() override;
    int64_t getTheoryIoSize() override;
 private:
    void *list_workspace_ = nullptr;
    int32_t *value_heads_ = nullptr, *value_next_ = nullptr;
    float *node_wx_ = nullptr, *node_wy_ = nullptr;
    int batch_, value_len_, heads_, dim_, queries_, levels_, points_;
};
}  // namespace optest
#endif  // ZOO_TECO_MS_DEFORM_ATTN_BACKWARD_H_
