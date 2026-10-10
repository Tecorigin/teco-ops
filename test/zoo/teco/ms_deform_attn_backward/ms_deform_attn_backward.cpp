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

#include <stdexcept>
#include "zoo/teco/ms_deform_attn_backward/ms_deform_attn_backward.h"
#include "zoo/teco/convert.h"
#include "interface/include/tecoops.h"
#include "interface/common/ms_deform_attn_list_capacity.h"
#include "device/device.h"

namespace optest {
void MsDeformAttnBackwardExecutor::paramCheck() {
    if (parser_->inputs().size() != 5 || parser_->outputs().size() != 3) {
        throw std::invalid_argument("MSDA backward expects five inputs and three gradient outputs");
    }
}

void MsDeformAttnBackwardExecutor::paramParse() {
    const auto *v = parser_->input(0);
    const auto *s = parser_->input(1);
    const auto *l = parser_->input(2);
    const auto *w = parser_->input(3);
    const auto *g = parser_->input(4);
    if (v->shape.size() != 4 || s->shape.size() != 2 || s->shape[1] != 2 ||
        l->shape.size() != 6 || l->shape[5] != 2 || w->shape.size() != 5 ||
        g->shape.size() != 3 || v->dtype != testpt::DTYPE_FLOAT ||
        s->dtype != testpt::DTYPE_INT64 || l->dtype != v->dtype || w->dtype != v->dtype ||
        g->dtype != v->dtype) {
        throw std::invalid_argument("invalid MSDA backward input layout/dtype");
    }
    batch_ = v->shape[0]; value_len_ = v->shape[1]; heads_ = v->shape[2]; dim_ = v->shape[3];
    queries_ = l->shape[1]; levels_ = l->shape[3]; points_ = l->shape[4];
    if (batch_ <= 0 || value_len_ <= 0 || heads_ <= 0 || dim_ <= 0 || dim_ > 128 ||
        queries_ <= 0 || levels_ <= 0 || levels_ > 8 || points_ <= 0 ||
        l->shape[0] != batch_ || l->shape[2] != heads_ || s->shape[0] != levels_ ||
        w->shape != std::vector<int>({batch_, queries_, heads_, levels_, points_}) ||
        g->shape != std::vector<int>({batch_, queries_, heads_ * dim_}) ||
        parser_->output(0)->shape != v->shape || parser_->output(0)->dtype != testpt::DTYPE_FLOAT ||
        parser_->output(1)->shape != l->shape || parser_->output(1)->dtype != v->dtype ||
        parser_->output(2)->shape != w->shape || parser_->output(2)->dtype != v->dtype) {
        throw std::invalid_argument("invalid MSDA backward shape or gradient output dtype");
    }
    // The CPU evaluator gates DIFF1 using max_error, not error_threshold.
    // Configure this operator's actual gate from the declared test threshold.
    const auto parsed = parser_->criterions();
    if (parsed.size() != 1 || parsed.begin()->formula != DIFF1 ||
        parsed.begin()->error_threshold <= 0) {
        throw std::invalid_argument("MSDA backward accuracy fixture requires one positive DIFF1 threshold");
    }
    auto criterion = *parsed.begin();
    criterion.max_error = criterion.error_threshold;
    criterions_.clear();
    criterions_.insert(criterion);
    ALLOG(INFO) << "MSDA backward DIFF1 max_error gate: " << criterion.max_error;
}

MsDeformAttnBackwardExecutor::~MsDeformAttnBackwardExecutor() {
    if (list_workspace_) scdaFree(list_workspace_);
}

void MsDeformAttnBackwardExecutor::destroy() {
    if (list_workspace_) {
        void *allocation = list_workspace_;
        list_workspace_ = nullptr;
        value_heads_ = value_next_ = nullptr;
        node_wx_ = node_wy_ = nullptr;
        if (!scdaFree(allocation)) {
            throw std::runtime_error("MSDA list workspace memory guard failed");
        }
    }
}

void MsDeformAttnBackwardExecutor::paramGeneration() {
    // FP32-only fixture: framework output[0] is the value gradient. ListReduce
    // writes all elements, and the private list storage is reset by ListInit.
    int64_t head_count = 0, node_count = 0;
    if (!tecoops::msdaListCount({batch_, value_len_, heads_}, &head_count) ||
        !tecoops::msdaListCount({4, batch_, queries_, heads_, levels_, points_}, &node_count)) {
        throw std::invalid_argument("MSDA list workspace counts must fit int32");
    }
    const size_t head_bytes = static_cast<size_t>(head_count) * sizeof(int32_t);
    const size_t node_bytes = static_cast<size_t>(node_count) * sizeof(int32_t);
    scdaMalloc(&list_workspace_, head_bytes + 3 * node_bytes);
    if (!list_workspace_) throw std::runtime_error("MSDA list workspace allocation failed");
    auto *base = static_cast<unsigned char *>(list_workspace_);
    value_heads_ = reinterpret_cast<int32_t *>(base);
    value_next_ = reinterpret_cast<int32_t *>(base + head_bytes);
    node_wx_ = reinterpret_cast<float *>(base + head_bytes + node_bytes);
    node_wy_ = reinterpret_cast<float *>(base + head_bytes + 2 * node_bytes);
}

void MsDeformAttnBackwardExecutor::compute() {
    const auto status = tecoopsMsDeformAttnBackwardList(
        handle_, dev_input[0], static_cast<const int64_t *>(dev_input[1]),
        dev_input[2], dev_input[3], dev_input[4], static_cast<float *>(dev_output[0]), nullptr,
        dev_output[1], dev_output[2], value_heads_, value_next_, node_wx_, node_wy_,
        batch_, value_len_, heads_, dim_, queries_, levels_, points_,
        convert::toTecoopsDataType(parser_->input(0)->dtype), TECOOPS_ALGO_0);
    if (status != TECOOPS_STATUS_SUCCESS) {
        throw std::runtime_error("MSDA backward native call failed");
    }
}

void MsDeformAttnBackwardExecutor::cpuCompute() { pythonComputeCPU("cpu"); }
int64_t MsDeformAttnBackwardExecutor::getTheoryOps() {
    return static_cast<int64_t>(batch_) * queries_ * heads_ * levels_ * points_ * dim_ * 24;
}
int64_t MsDeformAttnBackwardExecutor::getTheoryIoSize() {
    const int64_t element = parser_->input(0)->dtype == testpt::DTYPE_FLOAT ? 4 : 2;
    const int64_t v = static_cast<int64_t>(batch_) * value_len_ * heads_ * dim_;
    const int64_t samples = static_cast<int64_t>(batch_) * queries_ * heads_ * levels_ * points_;
    const int64_t g = static_cast<int64_t>(batch_) * queries_ * heads_ * dim_;
    return (v + samples * 6 + g) * element + v * 4 + levels_ * 2 * sizeof(int64_t);
}
}  // namespace optest
