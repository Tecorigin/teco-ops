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

#include "interface/include/tecoops.h"

#include <sdaa_runtime.h>

#include <cstdlib>

#include "interface/common/check.h"
#include "interface/common/macro.h"
#include "interface/include/builtin_type.h"

#define SPM_SIZE (256 * 1024)  // 256K
tecoopsStatus_t tecoopsCreate(tecoopsHandle_t* handle) {
    if (handle == nullptr) {
        return TECOOPS_STATUS_BAD_PARAM;
    }
    tecoopsContext* ctx = new tecoopsContext();
    if (ctx == nullptr) {
        return TECOOPS_STATUS_ALLOC_FAILED;
    }
    ctx->spa_num = 1;
    ctx->spe_num = 32;
    ctx->spm_size = SPM_SIZE;
    ctx->stream = nullptr;
    *handle = ctx;
    return TECOOPS_STATUS_SUCCESS;
}

tecoopsStatus_t tecoopsDestroy(tecoopsHandle_t handle) {
    if (handle == nullptr) {
        return TECOOPS_STATUS_BAD_PARAM;
    }
    delete handle;
    return TECOOPS_STATUS_SUCCESS;
}

tecoopsStatus_t tecoopsSetStream(tecoopsHandle_t handle, sdaaStream_t stream) {
    if (handle == nullptr) {
        return TECOOPS_STATUS_BAD_PARAM;
    }
    handle->stream = stream;
    return TECOOPS_STATUS_SUCCESS;
}

tecoopsStatus_t tecoopsGetStream(tecoopsHandle_t handle, sdaaStream_t* stream) {
    if (handle == nullptr || stream == nullptr) {
        return TECOOPS_STATUS_BAD_PARAM;
    }
    *stream = handle->stream;
    return TECOOPS_STATUS_SUCCESS;
}

tecoopsStatus_t tecoopsCreateTensorDescriptor(tecoopsTensorDescriptor_t* tensorDesc) {
    *tensorDesc = new tecoopsTensorStruct();
    return TECOOPS_STATUS_SUCCESS;
}

tecoopsStatus_t tecoopsSetTensor4dDescriptor(tecoopsTensorDescriptor_t tensorDesc,
                                             tecoopsTensorFormat_t format,
                                             tecoopsDataType_t dataType, int n, int c, int h,
                                             int w) {
    tensorDesc->format = format;
    tensorDesc->dataType = dataType;
    tensorDesc->nbDims = 4;
    checkTecoopsStatus(tensorDesc->setTensorDimN(n));
    checkTecoopsStatus(tensorDesc->setTensorDimC(c));
    checkTecoopsStatus(tensorDesc->setTensorDimH(h));
    checkTecoopsStatus(tensorDesc->setTensorDimW(w));

    int nStride, cStride, hStride, wStride;
    if (format == TECOOPS_TENSOR_NCHW) {
        nStride = (c * h * w);
        cStride = (h * w);
        hStride = (w);
        wStride = (1);
    } else if (format == TECOOPS_TENSOR_NHWC) {
        nStride = (h * w * c);
        cStride = (1);
        hStride = (w * c);
        wStride = (c);
    } else if (format == TECOOPS_TENSOR_CHWN) {
        nStride = (1);
        cStride = (h * w * n);
        hStride = (w * n);
        wStride = (n);
    } else if (format == TECOOPS_TENSOR_NWHC) {
        nStride = (w * h * c);
        cStride = (1);
        hStride = (c);
        wStride = (h * c);
    } else {
        return TECOOPS_STATUS_BAD_PARAM;
    }
    checkTecoopsStatus(tensorDesc->setTensorStrideN(nStride));
    checkTecoopsStatus(tensorDesc->setTensorStrideC(cStride));
    checkTecoopsStatus(tensorDesc->setTensorStrideH(hStride));
    checkTecoopsStatus(tensorDesc->setTensorStrideW(wStride));
    return TECOOPS_STATUS_SUCCESS;
}

tecoopsStatus_t tecoopsGetTensor4dDescriptor(tecoopsTensorDescriptor_t tensorDesc,
                                             tecoopsDataType_t* dataType, int* n, int* c, int* h,
                                             int* w, int* nStride, int* cStride, int* hStride,
                                             int* wStride) {
    if (tensorDesc->nbDims != 4) {
        return TECOOPS_STATUS_BAD_PARAM;
    }

    if (dataType != nullptr) {
        *dataType = tensorDesc->dataType;
    }
    if (n != nullptr) {
        checkTecoopsStatus(tensorDesc->getTensorDimN(n));
    }
    if (c != nullptr) {
        checkTecoopsStatus(tensorDesc->getTensorDimC(c));
    }
    if (h != nullptr) {
        checkTecoopsStatus(tensorDesc->getTensorDimH(h));
    }
    if (w != nullptr) {
        checkTecoopsStatus(tensorDesc->getTensorDimW(w));
    }

    if (nStride != nullptr) {
        checkTecoopsStatus(tensorDesc->getTensorStrideN(nStride));
    }
    if (cStride != nullptr) {
        checkTecoopsStatus(tensorDesc->getTensorStrideC(cStride));
    }
    if (hStride != nullptr) {
        checkTecoopsStatus(tensorDesc->getTensorStrideH(hStride));
    }
    if (wStride != nullptr) {
        checkTecoopsStatus(tensorDesc->getTensorStrideW(wStride));
    }
    return TECOOPS_STATUS_SUCCESS;
}

tecoopsStatus_t tecoopsSetTensorNdDescriptor(tecoopsTensorDescriptor_t tensorDesc,
                                             tecoopsDataType_t dataType, int nbDims,
                                             const int dimA[], const int strideA[]) {
    if (nbDims <= 0) {
        return TECOOPS_STATUS_BAD_PARAM;
    }

    tensorDesc->dataType = dataType;
    tensorDesc->format = TECOOPS_TENSOR_NCHW;
    if (nbDims == 4) {
        tensorDesc->n = dimA[0];
        tensorDesc->c = dimA[1];
        tensorDesc->h = dimA[2];
        tensorDesc->w = dimA[3];
    }
    tensorDesc->nbDims = nbDims;

    if (nbDims > TECOOPS_DIM_MAX) {
        if (TECO_PREDICT_FALSE(tensorDesc->expand_dims != nullptr)) {
            free(tensorDesc->expand_dims);
        }
        if (TECO_PREDICT_FALSE(tensorDesc->expand_strides != nullptr)) {
            free(tensorDesc->expand_strides);
        }
        tensorDesc->expand_dims = (int*)malloc(sizeof(int) * nbDims);
        tensorDesc->expand_strides = (int*)malloc(sizeof(int) * nbDims);
        tensorDesc->dimA = tensorDesc->expand_dims;
        tensorDesc->strideA = tensorDesc->expand_strides;
    }
    memcpy(tensorDesc->dimA, dimA, sizeof(int) * nbDims);

    if (strideA != nullptr) {
        memcpy(tensorDesc->strideA, strideA, sizeof(int) * nbDims);
    } else {
        (tensorDesc->strideA)[nbDims - 1] = 1;
        for (int i = nbDims - 2; i >= 0; i--) {
            (tensorDesc->strideA)[i] = (tensorDesc->strideA)[i + 1] * dimA[i + 1];
        }
    }
    return TECOOPS_STATUS_SUCCESS;
}

tecoopsStatus_t tecoopsGetTensorNdDescriptor(const tecoopsTensorDescriptor_t tensorDesc,
                                             int nbDimsRequested, tecoopsDataType_t* dataType,
                                             int* nbDims, int dimA[], int strideA[]) {
    if (nbDimsRequested <= 0) {
        return TECOOPS_STATUS_BAD_PARAM;
    }

    if (dataType != nullptr) {
        *dataType = tensorDesc->dataType;
    }

    int return_nbDims = tensorDesc->nbDims;
    if (nbDims != nullptr) {
        *nbDims = tensorDesc->nbDims;
        return_nbDims = (return_nbDims < nbDimsRequested) ? return_nbDims : nbDimsRequested;
    }
    if (dimA != nullptr) {
        memcpy(dimA, tensorDesc->dimA, sizeof(int) * (return_nbDims));
    }
    if (strideA != nullptr) {
        memcpy(strideA, tensorDesc->strideA, sizeof(int) * (return_nbDims));
    }
    return TECOOPS_STATUS_SUCCESS;
}

tecoopsStatus_t tecoopsDestroyTensorDescriptor(tecoopsTensorDescriptor_t tensorDesc) {
    if (tensorDesc != nullptr) {
        delete tensorDesc;
    }
    return TECOOPS_STATUS_SUCCESS;
}

const char* tecoopsGetErrorString(tecoopsStatus_t status) {
    switch (status) {
        case TECOOPS_STATUS_SUCCESS:
            return "TECOOPS_STATUS_SUCCESS";
        case TECOOPS_STATUS_NOT_INITIALIZED:
            return "TECOOPS_STATUS_NOT_INITIALIZED";
        case TECOOPS_STATUS_ALLOC_FAILED:
            return "TECOOPS_STATUS_ALLOC_FAILED";
        case TECOOPS_STATUS_BAD_PARAM:
            return "TECOOPS_STATUS_BAD_PARAM";
        case TECOOPS_STATUS_INTERNAL_ERROR:
            return "TECOOPS_STATUS_INTERNAL_ERROR";
        case TECOOPS_STATUS_INVALID_VALUE:
            return "TECOOPS_STATUS_INVALID_VALUE";
        case TECOOPS_STATUS_NOT_SUPPORTED:
            return "TECOOPS_STATUS_NOT_SUPPORTED";
        default:
            return "UNKNOWN_ERROR";
    }
}
