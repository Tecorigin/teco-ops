// BSD 3-Clause License
// Copyright (c) 2024, Tecorigin Co., Ltd.

#ifndef TECOOPS_UAL_OPS_FIND_MS_DEFORM_ATTN_FORWARD_H_
#define TECOOPS_UAL_OPS_FIND_MS_DEFORM_ATTN_FORWARD_H_

#include "ual/args/ms_deform_attn_forward_args.h"

namespace tecoops {
namespace ual {
namespace ops {

int findMsDeformAttnForwardBranch(
    const args::MsDeformAttnForwardPatchArgs *arg);

}  // namespace ops
}  // namespace ual
}  // namespace tecoops

#endif  // TECOOPS_UAL_OPS_FIND_MS_DEFORM_ATTN_FORWARD_H_
