// BSD 3-Clause License
//
// Copyright (c) 2026, Tecorigin Co., Ltd.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include "zoo/teco/convert.h"
#include "common/time.hpp"
#include "zoo/teco/rotary_embedding/rotary_embedding.h"
#include "interface/include/tecoops.h"

namespace optest {

void RotaryEmbeddingExecutor::paramCheck() {
    int nin = parser_->inputs().size();
    int nout = parser_->outputs().size();
    if (nin != 4 || nout != 2) {
        ALLOG(ERROR) << "rotary_embedding requires 4 inputs (q, k, cos_sin_cache, positions) "
                     << "and 2 outputs (q_out, k_out), got " << nin << " in, " << nout << " out.";
        throw std::invalid_argument(std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
}

void RotaryEmbeddingExecutor::paramParse() {
    num_tokens_ = parser_->input(0)->shape[0];
    q_stride_ = parser_->input(0)->shape[1];
    k_stride_ = parser_->input(1)->shape[1];
    cache_stride_ = parser_->input(2)->shape[1];
    max_position_ = parser_->input(2)->shape[0];
    head_dim_ = cache_stride_;
    q_out_stride_ = parser_->output(0)->shape[1];
    k_out_stride_ = parser_->output(1)->shape[1];

    auto re_param = parser_->getProtoNode()->tecokernel_param().rotary_embedding_param();
    num_q_heads_ = re_param.num_q_heads();
    num_kv_heads_ = re_param.num_kv_heads();
    is_neox_ = re_param.is_neox();
}

void RotaryEmbeddingExecutor::paramGeneration() {
    q_ = dev_input[0];
    k_ = dev_input[1];
    cos_sin_cache_ = dev_input[2];
    positions_ = static_cast<int64_t *>(dev_input[3]);
    q_out_ = dev_output[0];
    k_out_ = dev_output[1];

    // Positions are overwritten with a deterministic sequence so that the device
    // run and cpuCompute() index the cache identically.  (Same intent as the
    // slot_mapping overwrite in the reshape_and_cache case.)
    std::vector<int64_t> pos(num_tokens_);
    for (int t = 0; t < num_tokens_; ++t) {
        pos[t] = PositionAt(t);
    }
    sdaaMemcpy(const_cast<int64_t *>(positions_), pos.data(),
               static_cast<size_t>(num_tokens_) * sizeof(int64_t), sdaaMemcpyHostToDevice);
}

void RotaryEmbeddingExecutor::compute() {
    checkTECOOPS(tecoopsRotaryEmbedding(
        handle_,
        q_, k_, cos_sin_cache_, positions_,
        q_out_, k_out_,
        num_tokens_, num_q_heads_, num_kv_heads_, head_dim_,
        q_stride_, k_stride_, q_out_stride_, k_out_stride_, cache_stride_,
        is_neox_));
}

int64_t RotaryEmbeddingExecutor::getTheoryOps() {
    // per (token, head, d in [0, D/2)): 4 mults + 1 sub + 1 add = 6 flops
    int64_t heads = num_q_heads_ + num_kv_heads_;
    return static_cast<int64_t>(num_tokens_) * heads * (head_dim_ / 2) * 6;
}

int64_t RotaryEmbeddingExecutor::getTheoryIoSize() {
    constexpr int kHalfSize = 2;
    int64_t size = 0;
    size += static_cast<int64_t>(num_tokens_) * (q_stride_ + k_stride_) * kHalfSize;
    size += static_cast<int64_t>(num_tokens_) * cache_stride_ * kHalfSize;
    size += static_cast<int64_t>(num_tokens_) * sizeof(int64_t);
    size += static_cast<int64_t>(num_tokens_) * (q_out_stride_ + k_out_stride_) * kHalfSize;
    return size;
}

static inline float half_to_float(uint16_t h) {
    uint32_t sign = (static_cast<uint32_t>(h & 0x8000)) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;

    uint32_t fbits = 0;
    if (exp == 0) {
        if (mant == 0) {
            fbits = sign;
        } else {
            int e = -14;
            while ((mant & 0x0400) == 0) {
                mant <<= 1;
                e--;
            }
            mant &= 0x03FF;
            uint32_t fexp = static_cast<uint32_t>(e + 127);
            fbits = sign | (fexp << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        fbits = sign | 0x7F800000u | (mant << 13);
        if (mant != 0) fbits |= 0x00400000u;
    } else {
        uint32_t fexp = exp - 15 + 127;
        fbits = sign | (fexp << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &fbits, sizeof(f));
    return f;
}

static inline uint32_t round_shift_right(uint32_t value, int shift) {
    if (shift <= 0) return value << (-shift);
    if (shift >= 32) return 0;
    uint32_t result = value >> shift;
    uint32_t remainder = value & ((1u << shift) - 1u);
    uint32_t halfway = 1u << (shift - 1);
    if (remainder > halfway || (remainder == halfway && (result & 1u))) result++;
    return result;
}

static inline uint16_t float_to_half(float f) {
    uint32_t fbits;
    std::memcpy(&fbits, &f, sizeof(fbits));
    uint16_t sign = static_cast<uint16_t>((fbits >> 16) & 0x8000);
    uint32_t exp = (fbits >> 23) & 0xFF;
    uint32_t mant = fbits & 0x7FFFFF;

    if (exp == 0xFF) {
        if (mant == 0) return static_cast<uint16_t>(sign | 0x7C00);
        uint16_t nan_mant = static_cast<uint16_t>(mant >> 13);
        if (nan_mant == 0) nan_mant = 1;
        return static_cast<uint16_t>(sign | 0x7C00 | nan_mant | 0x0200);
    }
    if (exp == 0) return sign;

    int e = static_cast<int>(exp) - 127;
    if (e > 15) return static_cast<uint16_t>(sign | 0x7C00);

    if (e >= -14) {
        uint32_t half_exp = static_cast<uint32_t>(e + 15);
        uint32_t half_mant = mant >> 13;
        uint32_t remainder = mant & 0x1FFF;
        uint32_t halfway = 0x1000;
        if (remainder > halfway || (remainder == halfway && (half_mant & 1u))) half_mant++;
        if (half_mant == 0x400) {
            half_mant = 0;
            half_exp++;
        }
        if (half_exp >= 0x1F) return static_cast<uint16_t>(sign | 0x7C00);
        return static_cast<uint16_t>(sign | (half_exp << 10) | half_mant);
    }

    if (e >= -25) {
        uint32_t sig = mant | 0x800000u;
        int shift = -e - 1;
        uint32_t half_mant = round_shift_right(sig, shift);
        if (half_mant >= 0x400) return static_cast<uint16_t>(sign | 0x0400);
        return static_cast<uint16_t>(sign | half_mant);
    }
    return sign;
}

// Host reference: identical rounding sequence to teco_slave_rotary_embedding_fp16
// -- t1 = fp16(f32(x1)*f32(c)), t2 = fp16(f32(x2)*f32(s)), o1 = fp16(f32(t1)-f32(t2)),
//    t3 = fp16(f32(x2)*f32(c)), t4 = fp16(f32(x1)*f32(s)), o2 = fp16(f32(t3)+f32(t4)).
void RotaryEmbeddingExecutor::cpuCompute() {
    auto *q_in = static_cast<uint16_t *>(baseline_input[0]);
    auto *k_in = static_cast<uint16_t *>(baseline_input[1]);
    auto *cache = static_cast<uint16_t *>(baseline_input[2]);
    auto *pos = static_cast<int64_t *>(baseline_input[3]);
    auto *q_out = static_cast<uint16_t *>(baseline_output[0]);
    auto *k_out = static_cast<uint16_t *>(baseline_output[1]);

    for (int t = 0; t < num_tokens_; ++t) {
        pos[t] = PositionAt(t);  // keep the host copy authoritative too
    }

    const int H = head_dim_ / 2;
    for (int t = 0; t < num_tokens_; ++t) {
        const int64_t p = pos[t];
        const uint16_t *c_row = cache + static_cast<size_t>(p) * cache_stride_;
        const uint16_t *s_row = c_row + H;
        for (int tensor = 0; tensor < 2; ++tensor) {
            const int heads = (tensor == 0) ? num_q_heads_ : num_kv_heads_;
            const int in_stride = (tensor == 0) ? q_stride_ : k_stride_;
            const int out_stride = (tensor == 0) ? q_out_stride_ : k_out_stride_;
            const uint16_t *in = (tensor == 0 ? q_in : k_in) + static_cast<size_t>(t) * in_stride;
            uint16_t *out = (tensor == 0 ? q_out : k_out) + static_cast<size_t>(t) * out_stride;
            for (int h = 0; h < heads; ++h) {
                const uint16_t *x = in + static_cast<size_t>(h) * head_dim_;
                uint16_t *o = out + static_cast<size_t>(h) * head_dim_;
                for (int d = 0; d < H; ++d) {
                    const float x1 = half_to_float(x[d]);
                    const float x2 = half_to_float(x[d + H]);
                    const float c = half_to_float(c_row[d]);
                    const float s = half_to_float(s_row[d]);
                    const uint16_t t1 = float_to_half(x1 * c);
                    const uint16_t t2 = float_to_half(x2 * s);
                    const uint16_t t3 = float_to_half(x2 * c);
                    const uint16_t t4 = float_to_half(x1 * s);
                    o[d] = float_to_half(half_to_float(t1) - half_to_float(t2));
                    o[d + H] = float_to_half(half_to_float(t3) + half_to_float(t4));
                }
            }
        }
    }
}

}  // namespace optest
