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

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include "zoo/teco/convert.h"
#include "common/time.hpp"
#include "zoo/teco/rms_norm_compat/rms_norm_compat.h"
#include "interface/include/tecoops.h"

namespace optest {

void RmsNormCompatExecutor::paramCheck() {
    int nin = parser_->inputs().size();
    int nout = parser_->outputs().size();

    // case_0: epilogue only      (3 inputs: input, weight, rstd; 1 output: output)
    // case_1: fused residual add (4 inputs: input, weight, rstd, residual;
    //                             2 outputs: output, residual_out)
    if (!(nin == 3 && nout == 1) && !(nin == 4 && nout == 2)) {
        ALLOG(ERROR) << "rms_norm_compat requires (3 in, 1 out) or (4 in, 2 out), got "
                     << nin << " in, " << nout << " out.";
        throw std::invalid_argument(std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
}

void RmsNormCompatExecutor::paramParse() {
    auto meta_input = parser_->input(0);
    // input shape: [N, D]
    num_tokens_  = meta_input->shape[0];
    hidden_size_ = meta_input->shape[1];
    has_residual_ = (parser_->inputs().size() == 4);
}

void RmsNormCompatExecutor::paramGeneration() {
    input_  = dev_input[0];
    weight_ = dev_input[1];
    rstd_   = dev_input[2];
    output_ = dev_output[0];

    if (has_residual_) {
        residual_     = dev_input[3];
        residual_out_ = dev_output[1];
    } else {
        residual_     = nullptr;
        residual_out_ = nullptr;
    }
}

void RmsNormCompatExecutor::compute() {
    checkTECOOPS(tecoopsRmsNormCompat(
        handle_,
        input_, weight_, rstd_, residual_,
        output_, residual_out_,
        num_tokens_, hidden_size_));
}

int64_t RmsNormCompatExecutor::getTheoryOps() {
    // per element: residual add (fused form) + one fp32 multiply + one fp16
    // multiply; the fp32 -> fp16 rounding steps are not counted as ops.
    int64_t ops = static_cast<int64_t>(num_tokens_) * hidden_size_ * 2;
    if (has_residual_) {
        ops += static_cast<int64_t>(num_tokens_) * hidden_size_;
    }
    return ops;
}

int64_t RmsNormCompatExecutor::getTheoryIoSize() {
    constexpr int kHalfSize = 2;
    constexpr int kFloatSize = 4;
    int64_t size = 0;
    // read: input [N,D] half + weight [D] half + rstd [N] float
    size += static_cast<int64_t>(num_tokens_) * hidden_size_ * kHalfSize;
    size += static_cast<int64_t>(hidden_size_) * kHalfSize;
    size += static_cast<int64_t>(num_tokens_) * kFloatSize;
    // write: output [N,D] half
    size += static_cast<int64_t>(num_tokens_) * hidden_size_ * kHalfSize;
    if (has_residual_) {
        // read: residual [N,D]; write: residual_out [N,D]
        size += static_cast<int64_t>(num_tokens_) * hidden_size_ * kHalfSize;
        size += static_cast<int64_t>(num_tokens_) * hidden_size_ * kHalfSize;
    }
    return size;
}

static inline float compat_half_to_float(uint16_t h) {
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

static inline uint32_t compat_round_shift_right(uint32_t value, int shift) {
    if (shift <= 0) return value << (-shift);
    if (shift >= 32) return 0;
    uint32_t result = value >> shift;
    uint32_t remainder = value & ((1u << shift) - 1u);
    uint32_t halfway = 1u << (shift - 1);
    if (remainder > halfway || (remainder == halfway && (result & 1u))) result++;
    return result;
}

static inline uint16_t compat_float_to_half(float f) {
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
        uint32_t half_mant = compat_round_shift_right(sig, shift);
        if (half_mant >= 0x400) return static_cast<uint16_t>(sign | 0x0400);
        return static_cast<uint16_t>(sign | half_mant);
    }
    return sign;
}

// Reference order (bit-identical to the device kernel's documented order):
//   plain:    s = fp16( fp32(x) * rstd );        y = fp16( fp32(s) * fp32(w) )
//   residual: xr = fp32(x) + fp32(res);
//             residual_out = fp16(xr);
//             s = fp16( xr * rstd );             y = fp16( fp32(s) * fp32(w) )
void RmsNormCompatExecutor::cpuCompute() {
    auto *in_raw  = static_cast<uint16_t *>(baseline_input[0]);
    auto *wt_raw  = static_cast<uint16_t *>(baseline_input[1]);
    auto *rstd_raw = static_cast<float *>(baseline_input[2]);
    auto *out_raw = static_cast<uint16_t *>(baseline_output[0]);

    uint16_t *res_raw     = nullptr;
    uint16_t *res_out_raw = nullptr;
    if (has_residual_) {
        res_raw     = static_cast<uint16_t *>(baseline_input[3]);
        res_out_raw = static_cast<uint16_t *>(baseline_output[1]);
    }

    for (int r = 0; r < num_tokens_; ++r) {
        const float rstd = rstd_raw[r];
        for (int d = 0; d < hidden_size_; ++d) {
            const int idx = r * hidden_size_ + d;
            float xr = compat_half_to_float(in_raw[idx]);
            if (has_residual_) {
                xr += compat_half_to_float(res_raw[idx]);
                res_out_raw[idx] = compat_float_to_half(xr);
            }
            const float scaled = compat_half_to_float(compat_float_to_half(xr * rstd));
            out_raw[idx] = compat_float_to_half(scaled * compat_half_to_float(wt_raw[d]));
        }
    }
}

}  // namespace optest
