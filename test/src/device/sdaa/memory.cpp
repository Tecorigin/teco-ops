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
 
#include <sdaa_runtime.h>
#include <string>
#include "common/md5.h"
#include "common/tools.h"
#include "common/tecoallog.h"
#include "device/sdaa/memory.h"
#include "device/sdaa/helper.h"
#include "device/sdaa/common.h"

namespace optest {
namespace sdaa {

void DeviceMemory::deviceMalloc(void **p, size_t size) {
    size = (size + 511) / 512 * 512;
    void *ori_p;
    checkScdaErrors(sdaaMalloc(&ori_p, size + padding_size_ * 2 + addr_size_));
    *p = (char *)ori_p + addr_size_ + padding_size_;

    // padding using md5
    MD5 md5;
    std::string md5_value = md5.getMD5((size_t)(*p));
    char *padding_mem = (char *)malloc(padding_size_);
    for (int i = 0; i < padding_size_; i++) {
        padding_mem[i] = md5_value[i % md5_length_];
    }
    checkScdaErrors(sdaaMemcpy(ori_p, &size, sizeof(size_t), sdaaMemcpyHostToDevice));
    checkScdaErrors(
        sdaaMemcpy((char *)ori_p + addr_size_, padding_mem, padding_size_, sdaaMemcpyHostToDevice));
    checkScdaErrors(sdaaMemcpy((char *)ori_p + +addr_size_ + padding_size_ + size, padding_mem,
                               padding_size_, sdaaMemcpyHostToDevice));
    free(padding_mem);

    left_malloc_times++;
    ALLOG(INFO) << "====[malloc] ptr=" << *p << ", size=" << size;
}

bool DeviceMemory::deviceFree(void *p) {
    bool flag = true;

    // check padding
    void *ori_p = (char *)p - addr_size_ - padding_size_;

    char *size_mem = (char *)malloc(addr_size_);
    checkScdaErrors(sdaaMemcpy(size_mem, (char *)ori_p, sizeof(size_t), sdaaMemcpyDeviceToHost));
    size_t data_size = ((size_t *)size_mem)[0];

    char *padding_mem = (char *)malloc(padding_size_ * 2 + 1);
    checkScdaErrors(
        sdaaMemcpy(padding_mem, (char *)ori_p + addr_size_, padding_size_, sdaaMemcpyDeviceToHost));
    checkScdaErrors(sdaaMemcpy(padding_mem + padding_size_,
                               (char *)ori_p + addr_size_ + padding_size_ + data_size,
                               padding_size_, sdaaMemcpyDeviceToHost));

    MD5 md5;
    std::string md5_value = md5.getMD5((size_t)p);
    for (int i = 0; i < padding_size_ * 2; i++) {
        if (padding_mem[i] != md5_value[i % md5_length_]) {
            flag = false;
        }
    }

    // // show padding
    // if (!flag) {
    //     char *_tmp = (char *)malloc(md5_length_ + 1);
    //     _tmp[md5_length_] = '\0';
    //     int fault_count = 0;
    //     for (int i = 0; i < padding_size_ * 2 / md5_length_; i++) {
    //         if (fault_count < 20) {
    //             memcpy(_tmp, padding_mem + i * md5_length_, md5_length_);
    //             std::string str = _tmp;
    //             if (str != md5_value) {
    //                 fault_count++;
    //                 std::cout << i << ": " << str << std::endl;
    //             }
    //         }
    //     }
    //     free(_tmp);
    // }

    free(size_mem);
    free(padding_mem);
    checkScdaErrors(sdaaFree(ori_p));

    left_malloc_times--;
    ALLOG(INFO) << "====[free  ] ptr=" << p << ", size=" << data_size;
    return flag;
}

int DeviceMemory::check(sdaaStream_t stream) {
    int res = 0;
    if (left_malloc_times != 0) res = 1;
    left_malloc_times = 0;
    return res;
}

inline size_t align(size_t ptr, size_t n) { return (ptr + n - 1) / n * n; }

DeviceMemoryPool::DeviceMemoryPool() {
    ALLOG(VLOG) << "we will check all device memory";
    init();
}
DeviceMemoryPool::~DeviceMemoryPool() { /*destroy();*/
}  // can not destroy here, for sdaa Destructor early

void DeviceMemoryPool::init() {
    size_t mem_free_size = 0, mem_total_size = 0;
    checkScdaErrors(sdaaMemGetInfo(&mem_free_size, &mem_total_size));
    void *ptr;
    size_t size = mem_free_size - 128 * 1024 * 1024;
    checkScdaErrors(sdaaMalloc((void **)&ptr, size));
    checkScdaErrors(sdaaMemset(ptr, default_value_, size));

    phead_ = (pLNode)malloc(sizeof(LNode));
    ptail_ = (pLNode)malloc(sizeof(LNode));
    phead_->addr = ptr;
    phead_->size = 8 * 1024;  // as head
    phead_->prev = nullptr;
    phead_->next = ptail_;

    ptail_->addr = (char *)ptr + size;
    ptail_->size = 0;
    ptail_->prev = phead_;
    ptail_->next = nullptr;
}

void DeviceMemoryPool::free_allocated() {
    pLNode p = phead_->next;
    while (p->next != nullptr) {
        pLNode tmp = p->next;
        // printf("====[free_allocated  ]:%p, %d\n", p->addr, p->size);
        checkScdaErrors(sdaaMemset(p->addr, default_value_, p->size));
        free(p);
        p = tmp;
    }
}

void DeviceMemoryPool::destroy() {
    free_allocated();
    checkScdaErrors(sdaaFree(phead_->addr));
    free(ptail_);
    free(phead_);
}

void *DeviceMemoryPool::_alloc(size_t size) {
    pLNode p = phead_;
    while (p->next != nullptr) {
        size_t free_size =
            (size_t)(p->next->addr) - align((size_t)((char *)(p->addr) + p->size), 8 * 1024);
        if (free_size >= size) break;
        p = p->next;
    }
    if (p->next == nullptr) return nullptr;

    pLNode ret = (pLNode)malloc(sizeof(LNode));
    ret->size = size;
    ret->addr = (void *)(align((size_t)((char *)(p->addr) + p->size), 8 * 1024));
    ret->next = p->next;
    ret->prev = p;

    ret->next->prev = ret;
    ret->prev->next = ret;

    ALLOG(INFO) << "====[malloc] ptr=" << ret->addr << ", size=" << size;
    return ret->addr;
}

bool DeviceMemoryPool::_free(void *ptr) {
    pLNode p = phead_;
    while (p->next != nullptr) {
        if (p->addr == ptr) break;
        p = p->next;
    }
    if (p->next == nullptr) return false;

    checkScdaErrors(sdaaMemset(p->addr, default_value_, p->size));
    p->prev->next = p->next;
    p->next->prev = p->prev;

    ALLOG(INFO) << "====[free  ] ptr=" << p->addr << ", size=" << p->size;
    free(p);
    return true;
}

void DeviceMemoryPool::deviceMalloc(void **p, size_t size) {
    *p = _alloc(size);
    if (*p == nullptr) {
        ALLOG(ERROR) << "device malloc error";
    }
}

bool DeviceMemoryPool::deviceFree(void *p) { return _free(p); }

/*
return:
    1: memory not all free, problems with the test framework
    2: memory out, problems with the tested API
    3: both
*/
int DeviceMemoryPool::check(sdaaStream_t stream) {
    int res = 0;
    // if memory not all free
    if (phead_->next != ptail_) {
        free_allocated();
        res = 1;
    }

    int *d_result;
    sdaaMalloc((void **)&d_result, sizeof(int));
    checkHBM(stream, phead_->addr, (size_t)(ptail_->addr) - (size_t)(phead_->addr), default_value_,
             d_result);
    checkScdaErrors(sdaaStreamSynchronize(stream));
    int result = 0;
    checkScdaErrors(sdaaMemcpy(&result, d_result, sizeof(int), sdaaMemcpyDeviceToHost));
    checkScdaErrors(sdaaFree(d_result));
    if (result != 0) {
        res += 2;
    }

    return res;
}

}  // namespace sdaa
}  // namespace optest
