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

#ifndef TECOOPS_UAL_COM_LOG_H_
#define TECOOPS_UAL_COM_LOG_H_

#include <cstdio>

#define __LOG_INFO_PREFIX "INFO"
#define __LOG_WARN_PREFIX "WARNING"
#define __LOG_ERR_PREFIX "ERROR"

#define LOG(format, ...)                                                                  \
    do {                                                                                  \
        printf("%s [%s %d]: => " format " \n", __LOG_INFO_PREFIX, __FUNCTION__, __LINE__, \
               ##__VA_ARGS__);                                                            \
    } while (0)

#define WARNING(format, ...)                                                              \
    do {                                                                                  \
        printf("%s [%s %d]: => " format " \n", __LOG_WARN_PREFIX, __FUNCTION__, __LINE__, \
               ##__VA_ARGS__);                                                            \
    } while (0)

#define ERROR(format, ...)                                                              \
    do {                                                                                \
        printf("%s [%s %d]: %s => " format " \n", __LOG_ERR_PREFIX, __FILE__, __LINE__, \
               __FUNCTION__, ##__VA_ARGS__);                                            \
    } while (0)

#endif  // TECOOPS_UAL_COM_LOG_H_