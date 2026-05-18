#ifndef COMMON_TECO_KERNEL_CONTEXT_H_
#define COMMON_TECO_KERNEL_CONTEXT_H_

#ifdef USE_TECO
#include "sdaa_runtime.h"

#if defined(__SDAA__) && defined(__clang__)
struct uint3 {
    unsigned int x, y, z;
};

typedef struct uint3 uint3;

struct dim3 {
    unsigned int x, y, z;
#if defined(__cplusplus)
#if __cplusplus >= 201103L
    __host__ __device__
    constexpr dim3(unsigned int vx = 1,
                   unsigned int vy = 1,
                   unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
    __host__ __device__ constexpr dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
    __host__ __device__
    constexpr operator uint3(void) const { return uint3{x, y, z}; }
#else
    __host__ __device__
    dim3(unsigned int vx = 1,
         unsigned int vy = 1,
         unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
    __host__ __device__ dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
    __host__ __device__
    operator uint3(void) const {
        uint3 t;
        t.x = x;
        t.y = y;
        t.z = z;
        return t;
    }
#endif  // __cplusplus >= 201103L
#endif  // __cplusplus
};

__device__ inline float atomic_add(float *address, float val) {
    float old;
    bool is_ok = true;
    do {
        old = *address;
        float new_value = old + val;
        is_ok = __builtin_sw_slave_cas_32((int32_t *)address, *(int32_t *)&old,
                                          *(int32_t *)&new_value);
    } while (!is_ok);
    return old;
}

struct Half {
  half x;
  __host__ __device__  Half() = default;
  inline __host__ __device__ Half(float value);
  inline __host__ __device__ operator float() const;
};

inline __host__ __device__ Half::Half(float value)
    : x(half(value))
{
}

inline __host__ __device__ Half::operator float() const {
  return float(x);
}

#endif	// __SDAA__ && __clang__

namespace teco_kernel{
class TecoHandle {
public:
    TecoHandle(const TecoHandle&) = delete;
    TecoHandle& operator=(const TecoHandle&) = delete;

    static TecoHandle& getInstance() {
        static TecoHandle instance;
        return instance;
    }

    sdaaStream_t stream = NULL;

private:
    TecoHandle() = default;
    ~TecoHandle() = default;
};
}
#elif USE_CUDA
#include "cuda_runtime.h"
#include <cuda_fp16.h>

#if defined(__CUDACC__)
struct Half {
  __half x;
  Half() = default;
  inline __host__ __device__ Half(float value);
  inline __host__ __device__ operator float() const;
};

inline __host__ __device__ Half::Half(float value)
    : x(__float2half(value))
{
}

inline __host__ __device__ Half::operator float() const {
  return __half2float(x);
}
#endif // __CUDACC__

namespace cuda_kernel{
class CudaKernelHandle {
public:
    CudaKernelHandle(const CudaKernelHandle&) = delete;
    CudaKernelHandle& operator=(const CudaKernelHandle&) = delete;

    static CudaKernelHandle& getInstance() {
        static CudaKernelHandle instance;
        return instance;
    }

    cudaStream_t stream = NULL;

private:
    CudaKernelHandle() = default;
    ~CudaKernelHandle() = default;
};
}

#endif
#endif
