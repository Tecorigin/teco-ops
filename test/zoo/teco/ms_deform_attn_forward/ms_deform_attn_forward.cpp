// BSD 3-Clause License
//
// Copyright (c) 2024, Tecorigin Co., Ltd.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED.

#include <stdexcept>
#include <string>

#include "zoo/teco/convert.h"
#include "zoo/teco/ms_deform_attn_forward/ms_deform_attn_forward.h"
#include "interface/include/tecoops.h"

namespace optest {

void MsDeformAttnForwardExecutor::paramCheck() {
    if (parser_->inputs().size() != 4 || parser_->outputs().size() != 1) {
        ALLOG(ERROR) << "ms_deform_attn_forward expects 4 inputs and 1 output, got "
                     << parser_->inputs().size() << " inputs and "
                     << parser_->outputs().size() << " outputs.";
        throw std::invalid_argument(std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
}

void MsDeformAttnForwardExecutor::paramParse() {
    const auto *value = parser_->input(0);
    const auto *spatial_shapes = parser_->input(1);
    const auto *locations = parser_->input(2);
    const auto *weights = parser_->input(3);
    if (value->shape.size() != 4 || spatial_shapes->shape.size() != 2 ||
        spatial_shapes->shape[1] != 2 || locations->shape.size() != 6 ||
        locations->shape[5] != 2 || weights->shape.size() != 5) {
        throw std::invalid_argument("invalid ms_deform_attn_forward test shapes");
    }
    batch_ = value->shape[0];
    value_len_ = value->shape[1];
    num_heads_ = value->shape[2];
    head_dim_ = value->shape[3];
    num_queries_ = locations->shape[1];
    num_levels_ = locations->shape[3];
    num_points_ = locations->shape[4];
}

void MsDeformAttnForwardExecutor::paramGeneration() {
    value_ = dev_input[0];
    spatial_shapes_ = static_cast<const int64_t *>(dev_input[1]);
    sampling_locations_ = dev_input[2];
    attention_weights_ = dev_input[3];
    output_ = dev_output[0];
}

void MsDeformAttnForwardExecutor::compute() {
    const auto dtype = convert::toTecoopsDataType(parser_->input(0)->dtype);
    checkTECOOPS(tecoopsMsDeformAttnForward(
        handle_, value_, spatial_shapes_, sampling_locations_, attention_weights_, output_,
        batch_, value_len_, num_heads_, head_dim_, num_queries_, num_levels_, num_points_,
        dtype, TECOOPS_ALGO_0));
}

void MsDeformAttnForwardExecutor::cpuCompute() {
    pythonComputeCPU("cpu");
}

int64_t MsDeformAttnForwardExecutor::getTheoryOps() {
    return static_cast<int64_t>(batch_) * num_queries_ * num_heads_ * num_levels_ *
           num_points_ * head_dim_ * 12;
}

int64_t MsDeformAttnForwardExecutor::getTheoryIoSize() {
    const int64_t element_size = parser_->input(0)->dtype == testpt::DTYPE_FLOAT ? 4 : 2;
    const int64_t value_bytes = static_cast<int64_t>(batch_) * value_len_ * num_heads_ *
                                 head_dim_ * element_size;
    const int64_t location_bytes = static_cast<int64_t>(batch_) * num_queries_ * num_heads_ *
                                    num_levels_ * num_points_ * 2 * element_size;
    const int64_t weight_bytes = static_cast<int64_t>(batch_) * num_queries_ * num_heads_ *
                                 num_levels_ * num_points_ * element_size;
    const int64_t output_bytes = static_cast<int64_t>(batch_) * num_queries_ * num_heads_ *
                                 head_dim_ * element_size;
    return value_bytes + location_bytes + weight_bytes + output_bytes +
           static_cast<int64_t>(num_levels_) * 2 * sizeof(int64_t);
}

}  // namespace optest
