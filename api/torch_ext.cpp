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

    // The kernel reads the length metadata directly from device memory, so a
    // host tensor would be silently reinterpreted as a device address
    // (undefined behaviour). Validate the documented device-tensor contract up
    // front so misuse fails loudly instead.
    //
    // is_privateuseone() means "any PrivateUse1 backend", not "the SDAA
    // backend". SDAA is currently the only PrivateUse1 backend registered in
    // this stack, so the two are equivalent here; the TORCH_CHECK right below
    // (cu_seqlens_q.device() == q.device()) also pins the tensor to q's device,
    // so one from a hypothetical second PrivateUse1 backend cannot slip through
    // silently. If such a backend is ever added, replace this predicate with an
    // explicit backend check instead of relying on is_privateuseone().
    // (C++ ATen has no is_sdaa(); SDAA tensors are the PrivateUse1 backend,
    // hence is_privateuseone().)
    TORCH_CHECK(cu_seqlens_q.defined() && cu_seqlens_q.is_privateuseone(),
                "flash_attn_varlen_func: cu_seqlens_q must be an SDAA device "
                "tensor; host/CPU tensors are reinterpreted as device "
                "addresses by the kernel and violate the device-tensor "
                "contract (see doc/op_docs/flash_attention.md)");
    TORCH_CHECK(cu_seqlens_q.scalar_type() == torch::kInt32,
                "flash_attn_varlen_func: cu_seqlens_q must have dtype int32, got ",
                cu_seqlens_q.scalar_type());
    TORCH_CHECK(cu_seqlens_q.is_contiguous(),
                "flash_attn_varlen_func: cu_seqlens_q must be contiguous");
    TORCH_CHECK(cu_seqlens_q.device() == q.device(),
                "flash_attn_varlen_func: cu_seqlens_q must reside on the same "
                "device as q (", q.device(), "), got ", cu_seqlens_q.device());
    TORCH_CHECK(cu_seqlens_q.dim() == 1 &&
                cu_seqlens_q.numel() >= static_cast<int64_t>(batch_size) + 1,
                "flash_attn_varlen_func: cu_seqlens_q must be a 1-D tensor with "
                "at least batch_size + 1 elements (", batch_size + 1,
                "); got shape ", cu_seqlens_q.sizes());
    // cu_seqlens_k is part of the historical ABI but is NOT dereferenced by the
    // kernel: the KV lengths come from seqused_k (device side) instead.  It is
    // therefore deliberately not validated -- existing callers (including the
    // python tests) pass an empty/default-constructed tensor here, and rejecting
    // that would break compatibility for a tensor the kernel never reads.  Any
    // future kernel that starts consuming it must validate it with the same
    // device/dtype/contiguity contract as cu_seqlens_q.
    (void)cu_seqlens_k;

    // The kernel consumes q_seq_lens on device. Derive it from the device-side
    // cumulative lengths without a D2H sync followed by an H2D copy. The
    // subtraction produces a contiguous [batch_size] tensor for the ABI.
    auto q_lens = (
        cu_seqlens_q.narrow(0, 1, batch_size) -
        cu_seqlens_q.narrow(0, 0, batch_size)).contiguous();

    tecoopsHandle_t handle = getGlobalHandle();
    // Keep the Teco-Ops launch ordered with the device-side metadata
    // subtraction and the caller's other PyTorch work.
    tecoopsSetStream(handle, torch::sdaa::getCurrentSDAAStream());

    if (!out.defined()) {
        out = torch::zeros_like(q);
    } else {
        out.zero_();
    }

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
    m.def("causal_conv1d_fn_torch", &causal_conv1d_fn_torch, "causal_conv1d_fn_torch (SDAA)");
}
