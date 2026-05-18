#ifndef DEVICE_CUDA_COMMON_H_  // NOLINT
#define DEVICE_CUDA_COMMON_H_

#include <cuda_runtime.h>
#include <cuda.h>

#define checkCudaErrors(func)                                                                   \
    {                                                                                           \
        cudaError_t status = func;                                                              \
        if (status != cudaSuccess) {                                                            \
            printf("file:%s, func:%s, line: %d, CUDA Error:%d\n", __FILE__, __func__, __LINE__, \
                   status);                                                                     \
        }                                                                                       \
    }

typedef cudaStream_t scdaStream_t;
typedef cudaEvent_t scdaEvent_t;

#define scdaSetDevice(id) cudaSetDevice(id)
#define scdaGetDeviceCount(count) cudaGetDeviceCount(count)
#define scdaStreamCreate(stream) cudaStreamCreate(stream)
#define scdaStreamSynchronize(stream) cudaStreamSynchronize(stream)
#define scdaStreamDestroy(stream) cudaStreamDestroy(stream)
#define scdaMemcpy(dst, src, size, direction) cudaMemcpy(dst, src, size, cuda##direction)
#define scdaMemset(src, value, size) cudaMemset(src, value, size)
#define checkScdaErrors(func) checkCudaErrors(func)
#define scdaEventCreate(event) cudaEventCreate(event)
#define scdaEventRecord(event, stream) cudaEventRecord(event, stream)
#define scdaEventSynchronize(event) cudaEventSynchronize(event)
#define scdaEventElapsedTime(ms, start, end) cudaEventElapsedTime(ms, start, end)
#define scdaEventDestroy(event) cudaEventDestroy(event)

#endif  // DEVICE_CUDA_COMMON_H_  // NOLINT
