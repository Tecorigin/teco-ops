// BSD 3- Clause License Copyright (c) 2024, Tecorigin Co., Ltd. All rights
// reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
// OF SUCH DAMAGE.

#include <stdio.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include "zoo/teco/convert.h"
#include "common/time.hpp"
#include "zoo/teco/rms_norm/rms_norm.h"
#include "interface/include/tecoops.h"

namespace optest {

void RmsNormExecutor::paramCheck() {
    int nin = parser_->inputs().size();
    int nout = parser_->outputs().size();

    // case_0: pure RMSNorm (2 inputs: input, weight; 1 output: output)
    // case_1: RMSNorm+Add  (3 inputs: input, weight, residual; 2 outputs: output, residual_out)
    if (!(nin == 2 && nout == 1) && !(nin == 3 && nout == 2)) {
        ALLOG(ERROR) << "rms_norm requires (2 in, 1 out) or (3 in, 2 out), got "
                     << nin << " in, " << nout << " out.";
        throw std::invalid_argument(std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
}

void RmsNormExecutor::paramParse() {
    auto meta_input = parser_->input(0);
    // input shape: [N, D]
    num_tokens_  = meta_input->shape[0];
    hidden_size_ = meta_input->shape[1];

    has_residual_ = (parser_->inputs().size() == 3);

    auto rms_norm_param = parser_->getProtoNode()->tecokernel_param().rms_norm_param();
    eps_ = rms_norm_param.eps();
}

void RmsNormExecutor::paramGeneration() {
    input_   = dev_input[0];
    weight_  = dev_input[1];
    output_  = dev_output[0];

    if (has_residual_) {
        residual_     = dev_input[2];
        residual_out_ = dev_output[1];
    } else {
        residual_     = nullptr;
        residual_out_ = nullptr;
    }
}

void RmsNormExecutor::compute() {
    checkTECOOPS(tecoopsRmsNorm(
        handle_,
        input_, weight_,
        residual_,
        output_, residual_out_,
        num_tokens_, hidden_size_,
        eps_));
}

int64_t RmsNormExecutor::getTheoryOps() {
    // sum_sq: D mul + D-1 add ≈ 2D
    // rstd:   1 div + 1 sqrt = 2
    // norm:   D mul
    // scale:  D mul
    // Total per row: ~4D
    int64_t ops = static_cast<int64_t>(num_tokens_) * hidden_size_ * 4;
    if (has_residual_) {
        ops += static_cast<int64_t>(num_tokens_) * hidden_size_;  // add
    }
    return ops;
}

int64_t RmsNormExecutor::getTheoryIoSize() {
    constexpr int kHalfSize = 2;
    int64_t size = 0;
    // read: input [N,D] + weight [D]
    size += num_tokens_ * hidden_size_ * kHalfSize;
    size += hidden_size_ * kHalfSize;
    // write: output [N,D]
    size += num_tokens_ * hidden_size_ * kHalfSize;
    if (has_residual_) {
        // read: residual [N,D]
        size += num_tokens_ * hidden_size_ * kHalfSize;
        // write: residual_out [N,D]
        size += num_tokens_ * hidden_size_ * kHalfSize;
    }
    return size;
}

static inline float half_to_float(uint16_t h) {
    uint32_t sign = (static_cast<uint32_t>(h & 0x8000)) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;

    uint32_t fbits = 0;
    if (exp == 0) {
        if (mant == 0) {
            fbits = sign;
        } else {
            int e = -14;
            while ((mant & 0x0400) == 0) { mant <<= 1; e--; }
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
    uint32_t exp  = (fbits >> 23) & 0xFF;
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
        if (half_mant == 0x400) { half_mant = 0; half_exp++; }
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

void RmsNormExecutor::cpuCompute() {
    auto *in_raw  = static_cast<uint16_t *>(baseline_input[0]);
    auto *wt_raw  = static_cast<uint16_t *>(baseline_input[1]);
    auto *out_raw = static_cast<uint16_t *>(baseline_output[0]);

    uint16_t *res_raw    = nullptr;
    uint16_t *res_out_raw = nullptr;
    if (has_residual_) {
        res_raw     = static_cast<uint16_t *>(baseline_input[2]);
        res_out_raw = static_cast<uint16_t *>(baseline_output[1]);
    }

    for (int r = 0; r < num_tokens_; ++r) {
        float sum_sq = 0.0f;

        if (has_residual_) {
            for (int d = 0; d < hidden_size_; ++d) {
                float val = half_to_float(in_raw[r * hidden_size_ + d])
                            + half_to_float(res_raw[r * hidden_size_ + d]);
                sum_sq += val * val;
                unsigned short hval = float_to_half(val);
                res_out_raw[r * hidden_size_ + d] = hval;
                out_raw[r * hidden_size_ + d] = hval;
            }
        } else {
            unsigned short *row = in_raw + r * hidden_size_;
            for (int d = 0; d < hidden_size_; ++d) {
                sum_sq += half_to_float(row[d]) * half_to_float(row[d]);
            }
        }

        float rstd = 1.0f / sqrtf(sum_sq / static_cast<float>(hidden_size_) + eps_);
        unsigned short *row = has_residual_ ? out_raw + r * hidden_size_
                                            : in_raw + r * hidden_size_;
        for (int d = 0; d < hidden_size_; ++d) {
            float val = half_to_float(row[d]) * rstd * half_to_float(wt_raw[d]);
            out_raw[r * hidden_size_ + d] = float_to_half(val);
        }
    }
}

}  // namespace optest
