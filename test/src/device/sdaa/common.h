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
 
#ifndef DEVICE_SDAA_COMMON_H_  // NOLINT
#define DEVICE_SDAA_COMMON_H_

#include <stdlib.h>
#include <sdaa_runtime.h>

typedef sdaaStream_t scdaStream_t;

#define scdaSetDevice(id) sdaaSetDevice(id)
#define scdaGetDeviceCount(count) sdaaGetDeviceCount(count)
#define scdaStreamCreate(stream) sdaaStreamCreate(stream)
#define scdaStreamSynchronize(stream) sdaaStreamSynchronize(stream)
#define scdaStreamDestroy(stream) sdaaStreamDestroy(stream)
#define scdaMemcpy(dst, src, size, direction) sdaaMemcpy(dst, src, size, sdaa##direction)
#define scdaMemset(src, value, size) sdaaMemset(src, value, size)

#define TKNRM "\x1B[0m"
#define TKRED "\x1B[31m"
#define checkScdaErrors(error)                                                                 \
    {                                                                                          \
        sdaaError_t localError = error;                                                        \
        if ((localError != sdaaSuccess)) {                                                     \
            printf("%serror: '%s'(%d) from %s at %s:%d%s\n", TKRED,                            \
                   sdaaGetErrorString(localError), localError, #error, __FUNCTION__, __LINE__, \
                   TKNRM);                                                                     \
            exit(EXIT_FAILURE);                                                                \
        }                                                                                      \
    }

#endif  // DEVICE_SDAA_COMMON_H_  // NOLINT
