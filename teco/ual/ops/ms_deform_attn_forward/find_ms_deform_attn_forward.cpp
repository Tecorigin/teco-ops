// BSD 3-Clause License
// Copyright (c) 2024, Tecorigin Co., Ltd.

#include "ual/ops/ms_deform_attn_forward/find_ms_deform_attn_forward.h"

namespace tecoops {
namespace ual {
namespace ops {

int findMsDeformAttnForwardBranch(
    const args::MsDeformAttnForwardPatchArgs *arg) {
    if (arg->data_type == common::UALDataType::UAL_DTYPE_FLOAT) {
        return 0;
    }
    if (arg->data_type == common::UALDataType::UAL_DTYPE_HALF) {
        return 1;
    }
    return -1;
}

}  // namespace ops
}  // namespace ual
}  // namespace tecoops
