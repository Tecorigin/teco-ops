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

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include "interface/include/tecoops.h"
#include "zoo/teco/convert.h"
#include "zoo/teco/weight_stream_gemv/weight_stream_gemv.h"

namespace optest {

namespace {

float halfToFloat(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x03ff;
    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int e = -14;
            while ((mant & 0x0400) == 0) {
                mant <<= 1;
                --e;
            }
            mant &= 0x03ff;
            bits = sign | (static_cast<uint32_t>(e + 127) << 23) | (mant << 13);
        }
    } else if (exp == 0x1f) {
        bits = sign | 0x7f800000u | (mant << 13);
        if (mant != 0) bits |= 0x00400000u;
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

uint16_t floatToHalf(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000);
    const uint32_t exp = (bits >> 23) & 0xff;
    uint32_t mant = bits & 0x7fffff;
    if (exp == 0xff) {
        if (mant == 0) return static_cast<uint16_t>(sign | 0x7c00);
        uint16_t nan_mant = static_cast<uint16_t>(mant >> 13);
        if (nan_mant == 0) nan_mant = 1;
        return static_cast<uint16_t>(sign | 0x7c00 | nan_mant | 0x0200);
    }
    if (exp == 0) return sign;
    const int e = static_cast<int>(exp) - 127;
    if (e > 15) return static_cast<uint16_t>(sign | 0x7c00);
    if (e >= -14) {
        uint32_t half_exp = static_cast<uint32_t>(e + 15);
        uint32_t half_mant = mant >> 13;
        const uint32_t remainder = mant & 0x1fff;
        if (remainder > 0x1000 || (remainder == 0x1000 && (half_mant & 1u))) {
            ++half_mant;
        }
        if (half_mant == 0x400) {
            half_mant = 0;
            ++half_exp;
        }
        if (half_exp >= 0x1f) return static_cast<uint16_t>(sign | 0x7c00);
        return static_cast<uint16_t>(sign | (half_exp << 10) | half_mant);
    }
    if (e >= -25) {
        const uint32_t sig = mant | 0x800000u;
        const int shift = -e - 1;
        uint32_t half_mant = shift >= 32 ? 0 : sig >> shift;
        if (shift > 0 && shift < 32) {
            const uint32_t remainder = sig & ((1u << shift) - 1u);
            const uint32_t halfway = 1u << (shift - 1);
            if (remainder > halfway || (remainder == halfway && (half_mant & 1u))) {
                ++half_mant;
            }
        }
        if (half_mant >= 0x400) return static_cast<uint16_t>(sign | 0x0400);
        return static_cast<uint16_t>(sign | half_mant);
    }
    return sign;
}

}  // namespace

void WeightStreamGemvExecutor::paramCheck() {
    if (parser_->inputs().size() != 2 || parser_->outputs().size() != 1) {
        throw std::invalid_argument(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                    ": weight_stream_gemv expects 2 inputs and 1 output");
    }
}

void WeightStreamGemvExecutor::paramParse() {
    const auto *x = parser_->input(0);
    const auto *weight = parser_->input(1);
    const auto *output = parser_->output(0);
    if (x->shape.size() != 1 || weight->shape.size() != 2 || output->shape.size() != 1 ||
        weight->shape[1] != x->shape[0] || output->shape[0] != weight->shape[0] ||
        weight->shape[0] % 64 != 0 || weight->shape[1] % 16 != 0 ||
        x->dtype != testpt::DTYPE_HALF || weight->dtype != testpt::DTYPE_HALF ||
        output->dtype != testpt::DTYPE_HALF) {
        throw std::invalid_argument(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                    ": unsupported weight_stream_gemv shape or dtype");
    }
    k_ = static_cast<int>(x->shape[0]);
    n_ = static_cast<int>(weight->shape[0]);
}

void WeightStreamGemvExecutor::paramGeneration() {
    x_ = dev_input[0];
    weight_ = dev_input[1];
    output_ = dev_output[0];
}

void WeightStreamGemvExecutor::compute() {
    checkTECOOPS(tecoopsWeightStreamGemv(handle_, x_, weight_, output_, n_, k_));
}

void WeightStreamGemvExecutor::cpuCompute() {
    const auto *x = static_cast<const uint16_t *>(baseline_input[0]);
    const auto *weight = static_cast<const uint16_t *>(baseline_input[1]);
    auto *output = static_cast<uint16_t *>(baseline_output[0]);
    for (int row = 0; row < n_; ++row) {
        float acc = 0.0f;
        for (int col = 0; col < k_; ++col) {
            acc += halfToFloat(x[col]) * halfToFloat(weight[row * k_ + col]);
        }
        output[row] = floatToHalf(acc);
    }
}

int64_t WeightStreamGemvExecutor::getTheoryOps() {
    return static_cast<int64_t>(n_) * k_ * 2;
}

int64_t WeightStreamGemvExecutor::getTheoryIoSize() {
    return static_cast<int64_t>(k_ + n_ * k_ + n_) * sizeof(uint16_t);
}

}  // namespace optest
