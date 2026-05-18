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

#ifndef TECOOPS_UAL_COM_RT_H_
#define TECOOPS_UAL_COM_RT_H_

#define CHECK_SPM_OVERFLOW
// #define NOT_KILL_WHEN_SPM_OVERFLOW

#ifdef CHECK_SPM_OVERFLOW
#ifdef NOT_KILL_WHEN_SPM_OVERFLOW
#define RAISE_EXCEPTION (void)0
#else
#define RAISE_EXCEPTION                                     \
    do {                                                    \
        asm volatile("rcsr %0,5\n" : "=&r"(tmp)::"memory"); \
    } while (0)
#endif
#endif

#ifdef SPM_MALLOC_MEMSET
#include <stdlib.h>
#define __safe_spm_malloc(size, section, _filename, _lineno)                             \
    ({                                                                                   \
        void *ptr = (void *)malloc(size, AddressLowToHigh);                              \
        if (ptr == NULL || (int64_t)ptr < 0) {                                           \
            ai_printf("[%s:%d] spm malloc failed, size %u\n", _filename, _lineno, size); \
            return;                                                                      \
        }                                                                                \
        memset(ptr, 0, size);                                                            \
        ptr;                                                                             \
    })
#elif defined(CHECK_SPM_OVERFLOW)
#define __safe_spm_malloc(size, section, _filename, _lineno)                             \
    ({                                                                                   \
        void *ptr = (void *)malloc((size) + 128, section);                               \
        void *real_ptr = NULL;                                                           \
        if (ptr == NULL || (int64_t)ptr < 0) {                                           \
            ai_printf("[%s:%d] spm malloc failed, size %u\n", _filename, _lineno, size); \
            return;                                                                      \
        } else {                                                                         \
            for (int __ik = 0; __ik < 64; __ik++) {                                      \
                ((unsigned char *)ptr)[__ik] = 0xFF;                                     \
                ((unsigned char *)ptr)[(size) + 64 + __ik] = 0xFF;                       \
            }                                                                            \
            ((size_t *)ptr)[0] = size;                                                   \
            real_ptr = (void *)((unsigned long)ptr + 64);                                \
        }                                                                                \
        real_ptr;                                                                        \
    })
#else
#define __safe_spm_malloc(size, section, _filename, _lineno)                             \
    ({                                                                                   \
        void *ptr = (void *)malloc(size, section);                                       \
        if (ptr == NULL || (int64_t)ptr < 0) {                                           \
            ai_printf("[%s:%d] spm malloc failed, size %u\n", _filename, _lineno, size); \
            return;                                                                      \
        }                                                                                \
        ptr;                                                                             \
    })
#endif

#define rt_spm_malloc_try_left(size) __safe_spm_malloc(size, AddressLowToHigh, __FILE__, __LINE__)
#define rt_spm_malloc_try_right(size) __safe_spm_malloc(size, AddressHighToLow, __FILE__, __LINE__)
#define rt_spm_malloc(size) rt_spm_malloc_try_left(size)

#if defined(CHECK_SPM_OVERFLOW)
#define __safe_spm_free(ptr, _filename, _lineno)                                           \
    ({                                                                                     \
        if (ptr != NULL) {                                                                 \
            int __sum_pre = 0, __sum_suf = 0;                                              \
            constexpr int pre_count = 64 - sizeof(size_t);                                 \
            constexpr int suf_count = 64;                                                  \
            size_t size = ((size_t *)ptr)[-(64 / sizeof(size_t))];                         \
            for (int __ik = -pre_count; __ik < 0; __ik++) {                                \
                __sum_pre += ((unsigned char *)ptr)[__ik];                                 \
            }                                                                              \
            for (int __ik = 0; __ik < suf_count; __ik++) {                                 \
                __sum_suf += ((unsigned char *)ptr)[size + __ik];                          \
            }                                                                              \
            if (__sum_pre != ((unsigned char)0xFF) * pre_count ||                          \
                __sum_suf != ((unsigned char)0xFF) * suf_count) {                          \
                ai_printf("[%s:%d]: ERROR!!!!!!! spm overflow detected ptr %p size %lu\n", \
                          _filename, _lineno, ptr, size);                                  \
                unsigned int tmp;                                                          \
                RAISE_EXCEPTION;                                                           \
            }                                                                              \
            free(((unsigned char *)ptr) - 64);                                             \
        }                                                                                  \
    })
#else
#define __safe_spm_free(ptr, _filename, _lineno) \
    ({                                           \
        if (ptr != NULL) {                       \
            free((void *)ptr);                   \
        }                                        \
    })
#endif
#define rt_spm_free(ptr) __safe_spm_free(ptr, __FILE__, __LINE__);

#endif  // TECOOPS_UAL_COM_RT_H_