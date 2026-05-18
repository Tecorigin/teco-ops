// MIT License
// 
// Copyright (c) 2024, Tecorigin Co., Ltd.
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
 
#include "device/sdaa/helper.h"
#include <sdaa_runtime.h>
#include "device/sdaa/common.h"

namespace optest {
namespace sdaa {
void initLDM(sdaaStream_t stream) { 
    // init_ldm(stream); 
}
void calcHash(sdaaStream_t stream, void *in_data, size_t size, unsigned int *out_hash) {
    // calc_hash(stream, in_data, size, out_hash);
}
void checkHBM(sdaaStream_t stream, void *in_data, size_t size, unsigned char value, int *result) {
    // check_hbm(stream, in_data, size, value, result);
}

void initHBM() {
    size_t mem_free_size = 0, mem_total_size = 0;
    checkScdaErrors(sdaaMemGetInfo(&mem_free_size, &mem_total_size));
    void *ptr;
    size_t size = mem_free_size - 128 * 1024 * 1024;
    checkScdaErrors(sdaaMalloc((void **)&ptr, size));
    checkScdaErrors(sdaaMemset(ptr, 0xff, size));
    checkScdaErrors(sdaaFree(ptr));
}
}  // namespace sdaa
}  // namespace optest
