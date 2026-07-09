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


#ifndef CUSTOM_OPS_COMMON_H_
#define CUSTOM_OPS_COMMON_H_

#include <string>
#include "../../thirdparty/accuracy/include/tecoaccuracy.h"

uint32_t teco_pass;

#define CHECK()                                \
    if (teco_pass == 0) {                      \
        printf("%s: TECO FAILED\n", __FILE__); \
        return -1;                             \
    } else {                                   \
        printf("%s: TECO PASS\n", __FILE__);   \
    }


/**
typedef enum {
    TECOACCURACY_DATA_CHAR = 0,
    TECOACCURACY_DATA_INT16 = 1,
    TECOACCURACY_DATA_INT32 = 2,
    TECOACCURACY_DATA_INT64 = 3,
    TECOACCURACY_DATA_UCHAR = 4,
    TECOACCURACY_DATA_UINT16 = 5,
    TECOACCURACY_DATA_UINT32 = 6,
    TECOACCURACY_DATA_UINT64 = 7,
    TECOACCURACY_DATA_HALF = 8,
    TECOACCURACY_DATA_BFLOAT16 = 9,
    TECOACCURACY_DATA_FLOAT = 10,
    TECOACCURACY_DATA_DOUBLE = 11,
} tecoaccuracyDataType_t;
**/

// 回调函数，提供给tecoaccuracySetCallback，设置处理报错信息的方法
void callback(void *udata, const tecoaccuracyReport_t *dbconst, const char *msg) {
        printf("%s\n", msg);
        teco_pass = dbconst->is_passed;
}

// 封装teco 提供的精度测试接口。简洁化，方便用户调用。
// data_teco， 参数为 sdaaMemcpy 从teco device端copy 出来的数据地址。
// data_golden， 参数为CPU端计算出的作为对比用的精准数据的地址。
// compare_type， 要比对的数据的类型
// num， 要比对多少个数
void tecoAccuracyValidate(void* data_teco, void* data_golden,
                tecoaccuracyDataType_t compare_type, size_t num) {
    tecoaccuracyHandle_t handle;
    tecoaccuracyCreate(&handle);

    tecoaccuracyAttribute_t att;
    att.form = TECOACCURACY_JSON;

    tecoaccuracySetAttribute(handle, att);
    tecoaccuracySetCallback(handle, 0, callback);

    tecoaccuracyTwoSideValidate(handle, data_teco, data_golden, compare_type, num);
}

#endif  // CUSTOM_OPS_COMMON_H_
