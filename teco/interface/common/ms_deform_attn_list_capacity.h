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

#ifndef TECOOPS_INTERFACE_COMMON_MS_DEFORM_ATTN_LIST_CAPACITY_H_
#define TECOOPS_INTERFACE_COMMON_MS_DEFORM_ATTN_LIST_CAPACITY_H_

#include <cstdint>
#include <initializer_list>
#include <limits>

namespace tecoops {
// Check before every multiplication: neither intermediate arithmetic nor node IDs overflow.
inline bool msdaListCount(std::initializer_list<int64_t> factors, int64_t *count) {
    int64_t result = 1;
    for (const int64_t factor : factors) {
        if (factor <= 0 || result > std::numeric_limits<int32_t>::max() / factor) {
            return false;
        }
        result *= factor;
    }
    *count = result;
    return true;
}
}  // namespace tecoops

#endif  // TECOOPS_INTERFACE_COMMON_MS_DEFORM_ATTN_LIST_CAPACITY_H_
