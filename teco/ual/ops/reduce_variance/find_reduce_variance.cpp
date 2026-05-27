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

// #include "ual/ops/ops_com/discriptor_finder.h"
#include "ual/ops/reduce_variance/find_reduce_variance.h"
#include "ual/kernel/reduce_variance/reduce_variance.h"
#include "ual/com/log.h"

namespace tecoops {
namespace ual {
namespace ops {

using tecoops::ual::args::ReduceVariancePatchArgs;


int findReduceVarianceBranch(const ReduceVariancePatchArgs *args) {
    auto data_type = args->data_type;
    auto axis = args->axis;
    auto dims = args->rvargs->dims;
    auto dimA = args->dimA;
    int algo = 0;
    if (data_type == UALDataType::UAL_DTYPE_FLOAT) {
        if (axis == dims - 1) {
            // "teco_slave_reduce_variance_axis_right_float",
            algo = 0;
        } else if (axis > 0 && axis < dims - 1) {
            // "teco_slave_reduce_variance_axis_middle_float"
            algo = 1;
        } else {
            // "teco_slave_reduce_variance_axis_left_float"
            algo = 2;
        }
    } else {
        ERROR("not support this datatype \n");
    }
    return algo;
}

}  // namespace ops
}  // namespace ual
}  // namespace tecoops
