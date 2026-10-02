// BSD 3-Clause License
// Copyright (c) 2024, Tecorigin Co., Ltd.

#ifndef TECOOPS_UAL_KERNEL_MS_DEFORM_ATTN_FORWARD_H_
#define TECOOPS_UAL_KERNEL_MS_DEFORM_ATTN_FORWARD_H_

#include "ual/args/ms_deform_attn_forward_args.h"

using tecoops::ual::args::MsDeformAttnForwardArgs;

__global__ void teco_slave_ms_deform_attn_forward_fp32(MsDeformAttnForwardArgs args);
__global__ void teco_slave_ms_deform_attn_forward_fp16(MsDeformAttnForwardArgs args);

#endif  // TECOOPS_UAL_KERNEL_MS_DEFORM_ATTN_FORWARD_H_
