#include "device/cuda/version.h"
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "common/variable.h"

#if DNN_ENABLE
#include <cudnn.h>
#endif
#if BLAS_ENABLE
// #include <cublas.h>
#include <cublas_v2.h>
#endif

#include <iostream>
#include <string>
#include "device/cuda/common.h"

extern optest::GlobalVar global_var;

namespace optest {
namespace cuda {

std::string formatVersion(int version) { return std::to_string(version); }

void printVersions() {
    cuInit(0);
    int driver_version = 0, runtime_version = 0;
    char device_name[100];
    CUdevice device = global_var.kernel_id_;
    CUresult res = cuDeviceGetName(device_name, sizeof(device_name), device);
    checkCudaErrors(cudaDriverGetVersion(&driver_version));
    checkCudaErrors(cudaRuntimeGetVersion(&runtime_version));

    size_t dnn_version = 0;
    int blas_version = 0;
#if DNN_ENABLE
    dnn_version = cudnnGetVersion();
#endif

#if BLAS_ENABLE
    // cublasGetVersion(&blas_version);
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasGetVersion(handle, &blas_version);
    cublasDestroy(handle);
#endif

    std::cout << "----------------------------" << std::endl;
    std::cout << "device     : " << (std::string)device_name << std::endl;
    std::cout << "cudadriver : " << formatVersion(driver_version) << std::endl;
    std::cout << "cudart     : " << formatVersion(runtime_version) << std::endl;
    std::cout << "tecodnn    : " << formatVersion(dnn_version) << std::endl;
    std::cout << "tecoblas   : " << formatVersion(blas_version) << std::endl;
    std::cout << "----------------------------" << std::endl;
}

}  // namespace cuda
}  // namespace optest
