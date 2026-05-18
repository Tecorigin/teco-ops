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
 
#ifndef DEVICE_SDAA_MEMORY_H_  // NOLINT
#define DEVICE_SDAA_MEMORY_H_

#include <stddef.h>
#include <sdaa_runtime.h>

namespace optest {
namespace sdaa {

class DeviceMemory {
 public:
    static DeviceMemory *instance() {
        static thread_local DeviceMemory device_memory;
        return &device_memory;
    }
    void deviceMalloc(void **p, size_t size);
    bool deviceFree(void *p);
    int check(sdaaStream_t stream);
    void destroy() {}

 private:
    const int padding_size_ = 512 * 1024;
    const int md5_length_ = 32;
    const int addr_size_ = 512;
    int left_malloc_times = 0;
};

typedef struct LNode {
    void *addr;
    size_t size;
    struct LNode *prev;
    struct LNode *next;
} *pLNode;

class DeviceMemoryPool {
 public:
    static DeviceMemoryPool *instance() {
        static thread_local DeviceMemoryPool device_memory;
        return &device_memory;
    }

    void deviceMalloc(void **p, size_t size);
    bool deviceFree(void *p);
    int check(sdaaStream_t stream);
    void destroy();

 private:
    char default_value_ = 0xff;
    pLNode phead_ = nullptr;
    pLNode ptail_ = nullptr;
    DeviceMemoryPool();
    ~DeviceMemoryPool();
    void init();
    void free_allocated();
    void *_alloc(size_t size);
    bool _free(void *p);
};

}  // namespace sdaa
}  // namespace optest

#endif  // DEVICE_SDAA_MEMORY_H_   // NOLINT
