// BSD 3-Clause License
// Copyright (c) 2024, Tecorigin Co., Ltd.

#ifndef TECOOPS_UAL_OPS_MS_DEFORM_ATTN_FORWARD_H_
#define TECOOPS_UAL_OPS_MS_DEFORM_ATTN_FORWARD_H_

#include "ual/kernel/ms_deform_attn_forward/ms_deform_attn_forward.h"
#include "ual/ops/base_op.hpp"
#include "ual/ops/ms_deform_attn_forward/find_ms_deform_attn_forward.h"

namespace tecoops {
namespace ual {
namespace ops {

using args::MsDeformAttnForwardArgs;
using args::MsDeformAttnForwardPatchArgs;

struct MsDeformAttnForwardType {
    using ArgsType = MsDeformAttnForwardArgs;
    using PatchType = MsDeformAttnForwardPatchArgs;
    using RetType = void;
    using PImplType = void (*)(ArgsType);
};

static MsDeformAttnForwardType::PImplType MsDeformAttnForwardAlgos[] = {
    teco_slave_ms_deform_attn_forward_fp32,
    teco_slave_ms_deform_attn_forward_fp16,
};

static const char *MsDeformAttnForwardDescriptions[] = {
    "teco_slave_ms_deform_attn_forward_fp32",
    "teco_slave_ms_deform_attn_forward_fp16",
};

struct MsDeformAttnForwardOp
    : public BaseOp<MsDeformAttnForwardOp, MsDeformAttnForwardType> {
    static const char *name() { return "ms_deform_attn_forward"; }

    common::Status findImpl(const MsDeformAttnForwardPatchArgs *args) {
        int index = findMsDeformAttnForwardBranch(args);
        if (index < 0) {
            ERROR("ms_deform_attn_forward dtype is not supported!");
            return common::Status::NOT_IMPLEMENTED;
        }
        setInstance(MsDeformAttnForwardAlgos[index],
                    MsDeformAttnForwardDescriptions[index]);
        return common::Status::SUCCESS;
    }
};

}  // namespace ops
}  // namespace ual
}  // namespace tecoops

#endif  // TECOOPS_UAL_OPS_MS_DEFORM_ATTN_FORWARD_H_
