// BSD 3-Clause License
//
// Copyright (c) 2024, Tecorigin Co., Ltd.

#ifndef TECOOPS_UAL_ARGS_MS_DEFORM_ATTN_FORWARD_ARGS_H_
#define TECOOPS_UAL_ARGS_MS_DEFORM_ATTN_FORWARD_ARGS_H_

#include <cstdint>

#include "ual/com/def.h"

namespace tecoops {
namespace ual {
namespace args {

// Inference-only forward for MultiScaleDeformableAttention.
//
// value:               [N, S, M, D]
// spatial_shapes:      [L, 2] int64, (H_l, W_l)
// sampling_locations:  [N, Lq, M, L, P, 2]
// attention_weights:   [N, Lq, M, L, P]
// output:              [N, Lq, M, D]
//
// The kernel implements the same align_corners=False, zero-padding bilinear
// interpolation as torch.nn.functional.grid_sample.  It is deliberately a
// forward-only UAL primitive: autograd training continues to use the existing
// grid_sample compatibility path in the model adapter.
struct MsDeformAttnForwardArgs {
    int spe_num;
    const void *value;
    const int64_t *spatial_shapes;
    const void *sampling_locations;
    const void *attention_weights;
    void *output;
    int batch;
    int value_len;
    int num_heads;
    int head_dim;
    int num_queries;
    int num_levels;
    int num_points;
};

struct MsDeformAttnForwardPatchArgs {
    MsDeformAttnForwardArgs *atargs;
    common::UALDataType data_type;
    common::UALAlgoType algo;
};

}  // namespace args
}  // namespace ual
}  // namespace tecoops

#endif  // TECOOPS_UAL_ARGS_MS_DEFORM_ATTN_FORWARD_ARGS_H_
