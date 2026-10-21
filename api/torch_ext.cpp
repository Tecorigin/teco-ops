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

#include <torch/extension.h>
#include <torch_sdaa/sdaa_extension.h>

#include "interface/include/tecoops.h"

static tecoopsHandle_t g_handle = nullptr;

static tecoopsHandle_t getGlobalHandle() {
    if (g_handle == nullptr) {
        tecoopsCreate(&g_handle);
    }
    return g_handle;
}

void flatten_rays_torch(torch::Tensor rays, uint32_t N, uint32_t M, torch::Tensor res) {
    tecoopsHandle_t handle = getGlobalHandle();
    tecoopsFlattenRays(handle,
                       rays.data_ptr<int>(),
                       N, M,
                       res.data_ptr<int>(),
                       TECOOPS_ALGO_0);
}

void morton3D_invert_torch(torch::Tensor indices, uint32_t N, torch::Tensor coords) {
    tecoopsHandle_t handle = getGlobalHandle();
    tecoopsMorton3DInvert(handle,
                          indices.data_ptr<int>(),
                          N,
                          coords.data_ptr<int>());
}

void reshape_and_cache_torch(
    torch::Tensor key, torch::Tensor value,
    torch::Tensor slot_mapping,
    torch::Tensor key_cache, torch::Tensor value_cache) {
    tecoopsHandle_t handle = getGlobalHandle();
    int num_tokens = key.size(0);
    int num_kv_heads = key.size(1);
    int head_size = key.size(2);
    int num_blocks = key_cache.size(0);
    int block_size = key_cache.size(2);

    tecoopsReshapeAndCache(
        handle,
        key.data_ptr(), value.data_ptr(),
        slot_mapping.data_ptr<int64_t>(),
        key_cache.data_ptr(), value_cache.data_ptr(),
        num_tokens, num_kv_heads, head_size,
        num_blocks, block_size);
}

void rms_norm_torch(
    torch::Tensor input, torch::Tensor weight,
    c10::optional<torch::Tensor> residual,
    torch::Tensor output, c10::optional<torch::Tensor> residual_out,
    double eps) {
    tecoopsHandle_t handle = getGlobalHandle();
    int num_tokens = input.size(0);
    int hidden_size = input.size(1);

    const void *residual_ptr = residual.has_value() ? residual.value().data_ptr() : nullptr;
    void *res_out_ptr = residual_out.has_value() ? residual_out.value().data_ptr() : nullptr;

    tecoopsRmsNorm(handle,
                   input.data_ptr(), weight.data_ptr(),
                   residual_ptr, output.data_ptr(), res_out_ptr,
                   num_tokens, hidden_size, static_cast<float>(eps));
}

void rotary_embedding_torch(
    torch::Tensor q, torch::Tensor k, torch::Tensor cos_sin_cache, torch::Tensor positions,
    torch::Tensor q_out, torch::Tensor k_out, bool is_neox) {
    tecoopsHandle_t handle = getGlobalHandle();
    // Keep the operator on the caller's current SDAA stream, like rms_norm.
    tecoopsSetStream(handle, torch::sdaa::getCurrentSDAAStream());

    TORCH_CHECK(q.dim() == 2 && k.dim() == 2,
                "rotary_embedding: q and k must be 2-D [num_tokens, heads * head_dim]");
    TORCH_CHECK(cos_sin_cache.dim() == 2,
                "rotary_embedding: cos_sin_cache must be 2-D [max_position, head_dim]");
    TORCH_CHECK(q.scalar_type() == torch::kHalf && k.scalar_type() == torch::kHalf &&
                    cos_sin_cache.scalar_type() == torch::kHalf,
                "rotary_embedding: q, k and cos_sin_cache must be fp16");
    TORCH_CHECK(q_out.scalar_type() == torch::kHalf && k_out.scalar_type() == torch::kHalf,
                "rotary_embedding: q_out and k_out must be fp16");
    TORCH_CHECK(q.stride(-1) == 1 && k.stride(-1) == 1 && cos_sin_cache.stride(-1) == 1,
                "rotary_embedding: head dimension must be contiguous in q, k, cos_sin_cache");
    // NeoX (half-split) layout only: fail closed instead of rotating the wrong pairs.
    TORCH_CHECK(is_neox, "rotary_embedding: only the NeoX (half-split) layout is implemented");

    const int64_t num_tokens = q.size(0);
    const int64_t head_dim = cos_sin_cache.size(1);
    TORCH_CHECK(k.size(0) == num_tokens && positions.numel() == num_tokens,
                "rotary_embedding: q, k and positions must agree on num_tokens");
    TORCH_CHECK(q_out.size(0) == num_tokens && k_out.size(0) == num_tokens,
                "rotary_embedding: q_out and k_out must agree on num_tokens");
    TORCH_CHECK(head_dim > 0 && (head_dim % 2) == 0,
                "rotary_embedding: head_dim must be positive and even");
    TORCH_CHECK(q.size(1) % head_dim == 0 && k.size(1) % head_dim == 0,
                "rotary_embedding: head_dim must divide the fused head width");
    TORCH_CHECK(q_out.size(1) == q.size(1) && k_out.size(1) == k.size(1),
                "rotary_embedding: out tensors must keep the in-tensor head layout");

    // positions is int64 in the model runner; .to(kLong) is a no-op when it already is.
    auto pos = positions.is_contiguous() ? positions : positions.contiguous();
    if (pos.scalar_type() != torch::kLong) {
        pos = pos.to(torch::kLong);
    }

    tecoopsRotaryEmbedding(
        handle,
        q.data_ptr(), k.data_ptr(), cos_sin_cache.data_ptr(), pos.data_ptr<int64_t>(),
        q_out.data_ptr(), k_out.data_ptr(),
        static_cast<int>(num_tokens),
        static_cast<int>(q.size(1) / head_dim),
        static_cast<int>(k.size(1) / head_dim),
        static_cast<int>(head_dim),
        static_cast<int>(q.stride(0)), static_cast<int>(k.stride(0)),
        static_cast<int>(q_out.stride(0)), static_cast<int>(k_out.stride(0)),
        static_cast<int>(cos_sin_cache.stride(0)),
        is_neox ? 1 : 0);
}

void flash_attn_varlen_func_torch(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    int max_seqlen_q,
    torch::Tensor cu_seqlens_q,
    int max_seqlen_k,
    torch::Tensor cu_seqlens_k,
    torch::Tensor seqused_k,
    double softmax_scale,
    bool causal,
    torch::Tensor window_size,
    torch::Tensor block_table,
    bool return_softmax_lse,
    torch::Tensor out) {

    int batch_size = seqused_k.size(0);
    int max_block_num = k.size(0);

    // parse q_seq_lens
    auto cu_seqlens_q_cpu = cu_seqlens_q.cpu();
    auto cu_ptr = cu_seqlens_q_cpu.data_ptr<int>();
    auto q_lens_cpu = torch::empty({batch_size}, torch::kInt32);
    auto ql_cpu_ptr = q_lens_cpu.data_ptr<int>();
    for (int b = 0; b < batch_size; b++) {
        ql_cpu_ptr[b] = cu_ptr[b + 1] - cu_ptr[b];
    }
    auto q_lens = q_lens_cpu.to("sdaa");

    if (!out.defined()) {
        out = torch::zeros_like(q);
    } else {
        out.zero_();
    }

    tecoopsHandle_t handle = getGlobalHandle();
    tecoopsTensorDescriptor_t blockTableDesc, qDataDesc, kCacheDesc, vCacheDesc, oDataDesc;
    auto make_desc = [&](tecoopsTensorDescriptor_t &desc, tecoopsDataType_t dtype,
                         torch::Tensor &t) {
        tecoopsCreateTensorDescriptor(&desc);
        int ndim = t.dim();
        std::vector<int> dims(ndim);
        for (int i = 0; i < ndim; i++) dims[i] = t.size(i);
        tecoopsSetTensorNdDescriptor(desc, dtype, ndim, dims.data(), nullptr);
    };
    make_desc(blockTableDesc, TECOOPS_DATA_INT32, block_table);
    make_desc(qDataDesc, TECOOPS_DATA_HALF, q);
    make_desc(kCacheDesc, TECOOPS_DATA_HALF, k);
    make_desc(vCacheDesc, TECOOPS_DATA_HALF, v);
    make_desc(oDataDesc, TECOOPS_DATA_HALF, out);

    tecoopsFlashAttention(handle,
                          max_seqlen_q, max_seqlen_k, max_block_num,
                          (const int*)q_lens.data_ptr(), (const int *)seqused_k.data_ptr(),
                          blockTableDesc, block_table.data_ptr(),
                          qDataDesc, q.data_ptr(),
                          kCacheDesc, k.data_ptr(),
                          vCacheDesc, v.data_ptr(),
                          oDataDesc, out.data_ptr(),
                          /*workspace=*/nullptr);

    tecoopsDestroyTensorDescriptor(blockTableDesc);
    tecoopsDestroyTensorDescriptor(qDataDesc);
    tecoopsDestroyTensorDescriptor(kCacheDesc);
    tecoopsDestroyTensorDescriptor(vCacheDesc);
    tecoopsDestroyTensorDescriptor(oDataDesc);
}

void causal_conv1d_fn_torch(
    torch::Tensor out,          // [dim, totalSeqLen]  (modified in-place)
    torch::Tensor conv_states,  // [num_entries, convStateLen, dim]  (modified in-place)
    torch::Tensor x,            // [dim, totalSeqLen]
    torch::Tensor weight,       // [width, dim]
    torch::Tensor query_start_loc,  // [batch+1] int32
    torch::Tensor cache_indices,    // [batch] int32
    torch::Tensor has_initial_state, // [batch] int8
    int64_t pad_slot_id) {
    tecoopsHandle_t handle = getGlobalHandle();

    int dim = x.size(0);
    int totalSeqLen = x.size(1);
    int batch = query_start_loc.size(0) - 1;
    int width = weight.size(0);
    int convStateStride = conv_states.stride(0);
    int xSeqStride = dim;
    int outSeqStride = dim;

    // C API expects [totalSeqLen, dim]; torch passes [dim, totalSeqLen]
    auto x_t = x.t().contiguous();
    auto out_t = out.t().contiguous();

    tecoopsCausalConv1d(
        handle,
        batch, totalSeqLen, dim, width,
        convStateStride, xSeqStride, outSeqStride,
        pad_slot_id,
        /*api_mode=*/0, /*actMode=*/0,
        x_t.data_ptr(),
        conv_states.data_ptr(),
        weight.data_ptr(),
        out_t.data_ptr(),
        query_start_loc.data_ptr<int>(),
        cache_indices.data_ptr<int>(),
        has_initial_state.data_ptr<int8_t>());

    // Copy result from [totalSeqLen, dim] back to original out [dim, totalSeqLen]
    out.copy_(out_t.t());
}

PYBIND11_MODULE(_torch_ext, m) {
    m.def("flatten_rays", &flatten_rays_torch, "flatten_rays (SDAA)");
    m.def("morton3D_invert", &morton3D_invert_torch, "morton3D_invert (SDAA)");
    m.def("reshape_and_cache", &reshape_and_cache_torch, "reshape_and_cache (SDAA)");
    m.def("rms_norm", &rms_norm_torch, "rms_norm (SDAA)");
    m.def("flash_attn_varlen_func", &flash_attn_varlen_func_torch, "flash_attn_varlen_func (SDAA)");
    m.def("rotary_embedding", &rotary_embedding_torch, "rotary_embedding (SDAA, fp16 NeoX)");
    m.def("causal_conv1d_fn_torch", &causal_conv1d_fn_torch, "causal_conv1d_fn_torch (SDAA)");
}
