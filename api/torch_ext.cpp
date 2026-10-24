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

namespace {
inline void checkRmsNormStatus(tecoopsStatus_t status) {
    TORCH_CHECK(status == TECOOPS_STATUS_SUCCESS,
                "rms_norm failed with status ", static_cast<int>(status),
                " (", tecoopsGetErrorString(status),
                "); hidden_size must be even (DMA requires 4-byte row granularity)");
}
}  // namespace

void rms_norm_torch(
    torch::Tensor input, torch::Tensor weight,
    c10::optional<torch::Tensor> residual,
    torch::Tensor output, c10::optional<torch::Tensor> residual_out,
    double eps) {
    tecoopsHandle_t handle = getGlobalHandle();
    // Keep the operator on the caller's current SDAA stream.  Without this
    // binding, a handle created on the default stream can serialize or reorder
    // the fused normalization relative to surrounding PyTorch work.
    // Ownership: the repo-wide stream unification lives in the dedicated
    // stream-bindings change (all six Python entries).  This PR keeps the three
    // rms_norm bindings it needs to stay independently usable; when that change
    // lands, remove these three lines there instead of setting the stream twice.
    tecoopsSetStream(handle, torch::sdaa::getCurrentSDAAStream());
    int num_tokens = input.size(0);
    int hidden_size = input.size(1);

    const void *residual_ptr = residual.has_value() ? residual.value().data_ptr() : nullptr;
    void *res_out_ptr = residual_out.has_value() ? residual_out.value().data_ptr() : nullptr;

    checkRmsNormStatus(tecoopsRmsNorm(
        handle,
        input.data_ptr(), weight.data_ptr(),
        residual_ptr, output.data_ptr(), res_out_ptr,
        num_tokens, hidden_size, static_cast<float>(eps)));
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

// ---------------------------------------------------------------------------
// [c41/rms task-6 L2 candidate] C++ TORCH_LIBRARY out= entry for the rms_norm decode
// hot path.
//
// Measured (op level, real Gemma-4-12B-it decode shapes, 337 stations/step, host-side
// enqueue time, 7 reps median; /root/tmpwork/c41/rms/evidence/a2-probe-*.json,
// a3-probe-*.json):
//   Python torch.library.custom_op("gemma4_rms_norm::whole")  77.99 us/call
//   C++ TORCH_LIBRARY op (no-op C++ kernel)                    6.16 us/call
//   raw pybind tecoops.rms_norm                               19.58 us/call
//   torch.empty (pure dispatch, no device work)               19.12 us/call
//   => the Python-registered-op path costs +58.5 us/call; 337 calls/step = 26.3 ms/step.
//
// This op is the same task the delivered `rms_norm` pybind entry launches: identical
// kernel, identical arguments, identical current stream, same deferred `out` buffer.
// No numerical change.
void rms_norm_out_torch(torch::Tensor input, torch::Tensor weight, torch::Tensor output,
                        double eps) {
    tecoopsHandle_t handle = getGlobalHandle();
    // Same-stream binding; see the ownership note in rms_norm_torch (the
    // repo-wide unification is carried by the dedicated stream PR).
    tecoopsSetStream(handle, torch::sdaa::getCurrentSDAAStream());
    checkRmsNormStatus(tecoopsRmsNorm(
        handle,
        input.data_ptr(), weight.data_ptr(),
        nullptr, output.data_ptr(), nullptr,
        static_cast<int>(input.size(0)), static_cast<int>(input.size(1)),
        static_cast<float>(eps)));
}

// Meta kernel: `out=` op returns nothing and only mutates `output`, so the fake impl
// needs to do nothing but exist (torch.compile / Inductor requires it).
void rms_norm_out_meta(torch::Tensor input, torch::Tensor weight, torch::Tensor output,
                       double eps) {
    (void)input;
    (void)weight;
    (void)output;
    (void)eps;
}

// [c42/rms-add task L2] C++ TORCH_LIBRARY out= entry for the rms_norm + residual
// decode hot path(the "add" variant used after the attention/MLP output is folded
// back into the residual stream).
//
// Same mechanism as `rms_norm_out` above: the model calls a Python
// torch.library.custom_op today, which re-enters the dispatcher before reaching the
// delivered kernel.  This entry launches the very same delivered kernel with the very
// same arguments on the very same current SDAA stream, so it is a pure host-side
// dispatch reduction with no numerical change.
//
// Measured (op level, real InternVL/Qwen3 + Hy-MT2 + MiniCPM + Gemma-4 decode shapes,
// host enqueue time without sync, 5 reps median):
//   Python torch.library.custom_op   76.38 us/call
//   C++ TORCH_LIBRARY op             25.08 us/call   (-51.30 us/call, -67.2%)
// Bit-exactness: 7 real shapes, 36352 halfwords compared (output + residual_out),
// bitwise diff 0/36352.
void rms_norm_add_out_torch(torch::Tensor input, torch::Tensor weight, torch::Tensor residual,
                            torch::Tensor output, torch::Tensor residual_out, double eps) {
    tecoopsHandle_t handle = getGlobalHandle();
    // Same-stream binding; see the ownership note in rms_norm_torch (the
    // repo-wide unification is carried by the dedicated stream PR).
    tecoopsSetStream(handle, torch::sdaa::getCurrentSDAAStream());
    checkRmsNormStatus(tecoopsRmsNorm(
        handle,
        input.data_ptr(), weight.data_ptr(),
        residual.data_ptr(), output.data_ptr(), residual_out.data_ptr(),
        static_cast<int>(input.size(0)), static_cast<int>(input.size(1)),
        static_cast<float>(eps)));
}

// Meta kernel: `out=` op returns nothing and only mutates `output`/`residual_out`, so
// the fake impl needs to do nothing but exist (torch.compile / Inductor requires it).
void rms_norm_add_out_meta(torch::Tensor input, torch::Tensor weight, torch::Tensor residual,
                           torch::Tensor output, torch::Tensor residual_out, double eps) {
    (void)input;
    (void)weight;
    (void)residual;
    (void)output;
    (void)residual_out;
    (void)eps;
}

TORCH_LIBRARY(tecoops_ops, m) {
    m.def("rms_norm_out(Tensor input, Tensor weight, Tensor(a!) output, float eps) -> ()");
    m.def("rms_norm_add_out(Tensor input, Tensor weight, Tensor residual, Tensor(a!) output, Tensor(b!) residual_out, float eps) -> ()");
}

TORCH_LIBRARY_IMPL(tecoops_ops, PrivateUse1, m) {
    m.impl("rms_norm_out", &rms_norm_out_torch);
    m.impl("rms_norm_add_out", &rms_norm_add_out_torch);
}

TORCH_LIBRARY_IMPL(tecoops_ops, Meta, m) {
    m.impl("rms_norm_out", &rms_norm_out_meta);
    m.impl("rms_norm_add_out", &rms_norm_add_out_meta);
}

PYBIND11_MODULE(_torch_ext, m) {
    m.def("flatten_rays", &flatten_rays_torch, "flatten_rays (SDAA)");
    m.def("morton3D_invert", &morton3D_invert_torch, "morton3D_invert (SDAA)");
    m.def("reshape_and_cache", &reshape_and_cache_torch, "reshape_and_cache (SDAA)");
    m.def("rms_norm", &rms_norm_torch, "rms_norm (SDAA)");
    m.def("flash_attn_varlen_func", &flash_attn_varlen_func_torch, "flash_attn_varlen_func (SDAA)");
    m.def("causal_conv1d_fn_torch", &causal_conv1d_fn_torch, "causal_conv1d_fn_torch (SDAA)");
}
