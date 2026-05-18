#ifndef DEVICE_CUDA_MEMORY_H_  // NOLINT
#define DEVICE_CUDA_MEMORY_H_

#include <stddef.h>
#include <cuda_runtime.h>
#include <cuda.h>

namespace optest {
namespace cuda {

class DeviceMemory {
 public:
    static DeviceMemory *instance() {
        static thread_local DeviceMemory device_memory;
        return &device_memory;
    }
    void deviceMalloc(void **p, size_t size);
    bool deviceFree(void *p);
    int check(cudaStream_t stream);
    void destroy() {}

 private:
    int left_malloc_times = 0;
};

typedef DeviceMemory DeviceMemoryPool;

}  // namespace cuda
}  // namespace optest

#endif  // DEVICE_CUDA_MEMORY_H_   // NOLINT
