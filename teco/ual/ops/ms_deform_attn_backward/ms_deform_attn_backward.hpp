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

#ifndef TECOOPS_UAL_OPS_MS_DEFORM_ATTN_BACKWARD_H_
#define TECOOPS_UAL_OPS_MS_DEFORM_ATTN_BACKWARD_H_

#include "ual/kernel/ms_deform_attn_backward/ms_deform_attn_backward.h"
#include "ual/ops/base_op.hpp"
#include "ual/ops/ms_deform_attn_backward/find_ms_deform_attn_backward.h"

namespace tecoops {
namespace ual {
namespace ops {

using args::MsDeformAttnBackwardArgs;
using args::MsDeformAttnBackwardPatchArgs;

struct MsDeformAttnBackwardType {
    using ArgsType = MsDeformAttnBackwardArgs;
    using PatchType = MsDeformAttnBackwardPatchArgs;
    using RetType = void;
    using PImplType = void (*)(ArgsType);
};

static MsDeformAttnBackwardType::PImplType MsDeformAttnBackwardAlgos[] = {
    teco_slave_ms_deform_attn_backward_fp32,
    teco_slave_ms_deform_attn_backward_fp16,
    teco_slave_ms_deform_attn_backward_fp32_grain,
};

static const char *MsDeformAttnBackwardDescriptions[] = {
    "teco_slave_ms_deform_attn_backward_fp32",
    "teco_slave_ms_deform_attn_backward_fp16",
    "teco_slave_ms_deform_attn_backward_fp32_grain",
};

struct MsDeformAttnBackwardOp
    : public BaseOp<MsDeformAttnBackwardOp, MsDeformAttnBackwardType> {
    static const char *name() { return "ms_deform_attn_backward"; }

    common::Status findImpl(const MsDeformAttnBackwardPatchArgs *args) {
        int index = findMsDeformAttnBackwardBranch(args);
        if (index < 0) {
            ERROR("ms_deform_attn_backward dtype is not supported!");
            return common::Status::NOT_IMPLEMENTED;
        }
        setInstance(MsDeformAttnBackwardAlgos[index],
                    MsDeformAttnBackwardDescriptions[index]);
        return common::Status::SUCCESS;
    }
};

// Native reset is a separate kernel on the same handle stream as accumulation.
struct MsDeformAttnBackwardZeroOp
    : public BaseOp<MsDeformAttnBackwardZeroOp, MsDeformAttnBackwardType> {
    static const char *name() { return "ms_deform_attn_backward_zero_fp32"; }
    common::Status findImpl(const MsDeformAttnBackwardPatchArgs *) {
        setInstance(teco_slave_ms_deform_attn_backward_zero_fp32,
                    "teco_slave_ms_deform_attn_backward_zero_fp32");
        return common::Status::SUCCESS;
    }
};

// Separate conversion kernel keeps FP16 subnormal gradients intact.
struct MsDeformAttnBackwardCastOp
    : public BaseOp<MsDeformAttnBackwardCastOp, MsDeformAttnBackwardType> {
    static const char *name() { return "ms_deform_attn_backward_cast_fp16"; }
    common::Status findImpl(const MsDeformAttnBackwardPatchArgs *) {
        setInstance(teco_slave_ms_deform_attn_backward_cast_fp16,
                    "teco_slave_ms_deform_attn_backward_cast_fp16");
        return common::Status::SUCCESS;
    }
};

// List entry points are independent of the legacy atomic implementation.
struct MsDeformAttnBackwardListInitOp
    : public BaseOp<MsDeformAttnBackwardListInitOp, MsDeformAttnBackwardType> {
    static const char *name() { return "ms_deform_attn_backward_list_init"; }
    common::Status findImpl(const MsDeformAttnBackwardPatchArgs *) {
        setInstance(teco_slave_ms_deform_attn_backward_list_init,
                    "teco_slave_ms_deform_attn_backward_list_init");
        return common::Status::SUCCESS;
    }
};

static MsDeformAttnBackwardType::PImplType MsDeformAttnBackwardListProducerAlgos[] = {
    teco_slave_ms_deform_attn_backward_list_producer_fp32,
    teco_slave_ms_deform_attn_backward_list_producer_fp16,
};
static const char *MsDeformAttnBackwardListProducerDescriptions[] = {
    "teco_slave_ms_deform_attn_backward_list_producer_fp32",
    "teco_slave_ms_deform_attn_backward_list_producer_fp16",
};
struct MsDeformAttnBackwardListProducerOp
    : public BaseOp<MsDeformAttnBackwardListProducerOp, MsDeformAttnBackwardType> {
    static const char *name() { return "ms_deform_attn_backward_list_producer"; }
    common::Status findImpl(const MsDeformAttnBackwardPatchArgs *args) {
        const int index = findMsDeformAttnBackwardBranch(args);
        if (index < 0) {
            ERROR("ms_deform_attn_backward_list dtype is not supported!");
            return common::Status::NOT_IMPLEMENTED;
        }
        setInstance(MsDeformAttnBackwardListProducerAlgos[index],
                    MsDeformAttnBackwardListProducerDescriptions[index]);
        return common::Status::SUCCESS;
    }
};

static MsDeformAttnBackwardType::PImplType MsDeformAttnBackwardListReduceAlgos[] = {
    teco_slave_ms_deform_attn_backward_list_reduce_fp32,
    teco_slave_ms_deform_attn_backward_list_reduce_fp16,
};
static const char *MsDeformAttnBackwardListReduceDescriptions[] = {
    "teco_slave_ms_deform_attn_backward_list_reduce_fp32",
    "teco_slave_ms_deform_attn_backward_list_reduce_fp16",
};
struct MsDeformAttnBackwardListReduceOp
    : public BaseOp<MsDeformAttnBackwardListReduceOp, MsDeformAttnBackwardType> {
    static const char *name() { return "ms_deform_attn_backward_list_reduce"; }
    common::Status findImpl(const MsDeformAttnBackwardPatchArgs *args) {
        const int index = findMsDeformAttnBackwardBranch(args);
        if (index < 0) {
            ERROR("ms_deform_attn_backward_list dtype is not supported!");
            return common::Status::NOT_IMPLEMENTED;
        }
        setInstance(MsDeformAttnBackwardListReduceAlgos[index],
                    MsDeformAttnBackwardListReduceDescriptions[index]);
        return common::Status::SUCCESS;
    }
};

}  // namespace ops
}  // namespace ual
}  // namespace tecoops

#endif  // TECOOPS_UAL_OPS_MS_DEFORM_ATTN_BACKWARD_H_
