// BSD 3- Clause License Copyright (c) 2024, Tecorigin Co., Ltd. All rights
// reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
// OF SUCH DAMAGE.

#ifndef TECOOPS_UAL_OPS_REDUCE_VARIANCE_REDUCE_VARIANCE_HPP_
#define TECOOPS_UAL_OPS_REDUCE_VARIANCE_REDUCE_VARIANCE_HPP_

#include "ual/ops/base_op.hpp"
#include "ual/com/log.h"
#include "ual/kernel/reduce_variance/reduce_variance.h"
#include "ual/ops/reduce_variance/find_reduce_variance.h"

namespace tecoops {
namespace ual {
namespace ops {

using tecoops::ual::args::ReduceVarianceArgs;
using tecoops::ual::args::ReduceVariancePatchArgs;
using tecoops::ual::common::Status;

struct ReduceVarianceType {
    using ArgsType = ReduceVarianceArgs;
    using PatchType = ReduceVariancePatchArgs;
    using RetType = void;
    using PImplType = void (*)(ArgsType);
};

// array of function pointers to different reduce variance operator algorithm
static const ReduceVarianceType::PImplType ReduceVarianceAlgos[] = {
    /* 00 */
    teco_slave_reduce_variance_axis_right_float,
    /* 01 */
    teco_slave_reduce_variance_axis_middle_float,
    /* 02 */
    teco_slave_reduce_variance_axis_left_float,
};

// array of strings describied the names of reduce variance operator algorithm
static const char *ReduceVarianceAlgosDiscription[] = {
    /* 00 */
    "teco_slave_reduce_variance_axis_right_float",
    /* 01 */
    "teco_slave_reduce_variance_axis_middle_float",
    /* 02 */
    "teco_slave_reduce_variance_axis_left_float",
};
struct ReduceVarianceOp : public BaseOp<ReduceVarianceOp, ReduceVarianceType> {
 public:
    using ArgsType = typename ReduceVarianceType::ArgsType;
    using PatchType = typename ReduceVarianceType::PatchType;
    using RetType = typename ReduceVarianceType::RetType;
    using PImplType = typename ReduceVarianceType::PImplType;

    ReduceVarianceOp() = default;
    ~ReduceVarianceOp() = default;

    static const char *name() { return "reduce_variance"; }

    Status findImpl(const PatchType *args) {
        int index = findReduceVarianceBranch(args);
        if (index == -1) {
            ERROR("reduce_variance branch is not exit!");
            return Status::NOT_IMPLEMENTED;
        }
        setInstance(ReduceVarianceAlgos[index], ReduceVarianceAlgosDiscription[index]);
        return Status::SUCCESS;
    }
};

}  // namespace ops
}  // namespace ual
}  // namespace tecoops

#endif  // TECOOPS_UAL_OPS_REDUCE_VARIANCE_REDUCE_VARIANCE_HPP_
