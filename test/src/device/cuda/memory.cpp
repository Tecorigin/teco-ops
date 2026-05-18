#include "device/cuda/memory.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <string>
#include "common/tecoallog.h"
#include "device/cuda/common.h"

namespace optest {
namespace cuda {

void DeviceMemory::deviceMalloc(void **p, size_t size) {
    checkCudaErrors(cudaMalloc(p, size));
    left_malloc_times++;
    ALLOG(INFO) << "====[malloc] ptr=" << *p << ", size=" << size;
}

bool DeviceMemory::deviceFree(void *p) {
    ALLOG(INFO) << "====[free  ] ptr=" << p;
    checkCudaErrors(cudaFree(p));
    left_malloc_times--;
    return true;
}

int DeviceMemory::check(cudaStream_t stream) {
    int res = 0;
    if (left_malloc_times != 0) res = 1;
    left_malloc_times = 0;
    return res;
}

}  // namespace cuda
}  // namespace optest
