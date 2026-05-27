#include "interface/common/tensor.h"

#include "interface/common/check.h"
#include "interface/common/convert.h"
#include "ual/com/log.h"

using tecoops::Convert;

// tecoopsTensorStruct
tecoopsStatus_t tecoopsTensorStruct::check() {
    switch (format) {
        case TECOOPS_TENSOR_NCHW:
        case TECOOPS_TENSOR_NHWC:
        case TECOOPS_TENSOR_NWHC:
        case TECOOPS_TENSOR_CHWN: {
            if (nbDims != 4) {
                ERROR("tecoops tensor struct format and nbDims mismatch\n");
                return TECOOPS_STATUS_BAD_PARAM;
            }
            return TECOOPS_STATUS_SUCCESS;
        }
        default:
            ERROR("tecoops tensor struct illegal format: %d\n", format);
            return TECOOPS_STATUS_BAD_PARAM;
    }
}

tecoopsStatus_t tecoopsTensorStruct::getTensorDimN(int* dim_size) {
    checkTecoopsStatus(check());

    int index;
    switch (format) {
        case TECOOPS_TENSOR_NCHW:
        case TECOOPS_TENSOR_NHWC:
        case TECOOPS_TENSOR_NWHC:
            index = 0;
            break;
        case TECOOPS_TENSOR_CHWN:
            index = 3;
            break;
        default:
            ERROR("tecoops getTensorDimN illegal format: %d\n", format);
            return TECOOPS_STATUS_BAD_PARAM;
    }
    *dim_size = dimA[index];
    return TECOOPS_STATUS_SUCCESS;
}

tecoopsStatus_t tecoopsTensorStruct::getTensorDimH(int* dim_size) {
    checkTecoopsStatus(check());

    int index;
    switch (format) {
        case TECOOPS_TENSOR_NHWC:
        case TECOOPS_TENSOR_CHWN:
            index = 1;
            break;
        case TECOOPS_TENSOR_NCHW:
        case TECOOPS_TENSOR_NWHC:
            index = 2;
            break;
        default:
            ERROR("tecoops getTensorDimH illegal format: %d\n", format);
            return TECOOPS_STATUS_BAD_PARAM;
    }
    *dim_size = dimA[index];
    return TECOOPS_STATUS_SUCCESS;
}

tecoopsStatus_t tecoopsTensorStruct::getTensorDimW(int* dim_size) {
    checkTecoopsStatus(check());

    int index;
    switch (format) {
        case TECOOPS_TENSOR_NWHC:
            index = 1;
            break;
        case TECOOPS_TENSOR_NHWC:
        case TECOOPS_TENSOR_CHWN:
            index = 2;
            break;
        case TECOOPS_TENSOR_NCHW:
            index = 3;
            break;
        default:
            ERROR("tecoops getTensorDimW illegal format: %d\n", format);
            return TECOOPS_STATUS_BAD_PARAM;
    }
    *dim_size = dimA[index];
    return TECOOPS_STATUS_SUCCESS;
}

tecoopsStatus_t tecoopsTensorStruct::getTensorDimC(int* dim_size) {
    checkTecoopsStatus(check());

    int index;
    switch (format) {
        case TECOOPS_TENSOR_CHWN:
            index = 0;
            break;
        case TECOOPS_TENSOR_NCHW:
            index = 1;
            break;
        case TECOOPS_TENSOR_NHWC:
        case TECOOPS_TENSOR_NWHC:
            index = 3;
            break;
        default:
            ERROR("tecoops getTensorDimC illegal format: %d\n", format);
            return TECOOPS_STATUS_BAD_PARAM;
    }
    *dim_size = dimA[index];
    return TECOOPS_STATUS_SUCCESS;
}

tecoopsStatus_t tecoopsTensorStruct::getTensorStrideN(int* stride_size) {
    checkTecoopsStatus(check());

    int index;
    switch (format) {
        case TECOOPS_TENSOR_NCHW:
        case TECOOPS_TENSOR_NHWC:
        case TECOOPS_TENSOR_NWHC:
            index = 0;
            break;
        case TECOOPS_TENSOR_CHWN:
            index = 3;
            break;
        default:
            ERROR("tecoops getTensorStrideN illegal format: %d\n", format);
            return TECOOPS_STATUS_BAD_PARAM;
    }
    *stride_size = strideA[index];
    return TECOOPS_STATUS_SUCCESS;
}

tecoopsStatus_t tecoopsTensorStruct::getTensorStrideH(int* stride_size) {
    checkTecoopsStatus(check());

    int index;
    switch (format) {
        case TECOOPS_TENSOR_NHWC:
        case TECOOPS_TENSOR_CHWN:
            index = 1;
            break;
        case TECOOPS_TENSOR_NCHW:
        case TECOOPS_TENSOR_NWHC:
            index = 2;
            break;
        default:
            ERROR("tecoops getTensorStrideH illegal format: %d\n", format);
            return TECOOPS_STATUS_BAD_PARAM;
    }
    *stride_size = strideA[index];
    return TECOOPS_STATUS_SUCCESS;
}

tecoopsStatus_t tecoopsTensorStruct::getTensorStrideW(int* stride_size) {
    checkTecoopsStatus(check());

    int index;
    switch (format) {
        case TECOOPS_TENSOR_NWHC:
            index = 1;
            break;
        case TECOOPS_TENSOR_NHWC:
        case TECOOPS_TENSOR_CHWN:
            index = 2;
            break;
        case TECOOPS_TENSOR_NCHW:
            index = 3;
            break;
        default:
            ERROR("tecoops getTensorStrideW illegal format: %d\n", format);
            return TECOOPS_STATUS_BAD_PARAM;
    }
    *stride_size = strideA[index];
    return TECOOPS_STATUS_SUCCESS;
}

tecoopsStatus_t tecoopsTensorStruct::getTensorStrideC(int* stride_size) {
    checkTecoopsStatus(check());

    int index;
    switch (format) {
        case TECOOPS_TENSOR_CHWN:
            index = 0;
            break;
        case TECOOPS_TENSOR_NCHW:
            index = 1;
            break;
        case TECOOPS_TENSOR_NHWC:
        case TECOOPS_TENSOR_NWHC:
            index = 3;
            break;
        default:
            ERROR("tecoops getTensorStrideC illegal format: %d\n", format);
            return TECOOPS_STATUS_BAD_PARAM;
    }
    *stride_size = strideA[index];
    return TECOOPS_STATUS_SUCCESS;
}

tecoopsStatus_t tecoopsTensorStruct::setTensorDimN(int dim_size) {
    checkTecoopsStatus(check());

    int index;
    switch (format) {
        case TECOOPS_TENSOR_NCHW:
        case TECOOPS_TENSOR_NHWC:
        case TECOOPS_TENSOR_NWHC:
            index = 0;
            break;
        case TECOOPS_TENSOR_CHWN:
            index = 3;
            break;
        default:
            ERROR("tecoops setTensorDimN illegal format: %d\n", format);
            return TECOOPS_STATUS_BAD_PARAM;
    }
    dimA[index] = dim_size;
    n = dim_size;
    return TECOOPS_STATUS_SUCCESS;
}

tecoopsStatus_t tecoopsTensorStruct::setTensorDimH(int dim_size) {
    checkTecoopsStatus(check());

    int index;
    switch (format) {
        case TECOOPS_TENSOR_NHWC:
        case TECOOPS_TENSOR_CHWN:
            index = 1;
            break;
        case TECOOPS_TENSOR_NCHW:
        case TECOOPS_TENSOR_NWHC:
            index = 2;
            break;
        default:
            ERROR("tecoops setTensorDimH illegal format: %d\n", format);
            return TECOOPS_STATUS_BAD_PARAM;
    }
    dimA[index] = dim_size;
    h = dim_size;
    return TECOOPS_STATUS_SUCCESS;
}

tecoopsStatus_t tecoopsTensorStruct::setTensorDimW(int dim_size) {
    checkTecoopsStatus(check());

    int index;
    switch (format) {
        case TECOOPS_TENSOR_NWHC:
            index = 1;
            break;
        case TECOOPS_TENSOR_NHWC:
        case TECOOPS_TENSOR_CHWN:
            index = 2;
            break;
        case TECOOPS_TENSOR_NCHW:
            index = 3;
            break;
        default:
            ERROR("tecoops setTensorDimW illegal format: %d\n", format);
            return TECOOPS_STATUS_BAD_PARAM;
    }
    dimA[index] = dim_size;
    w = dim_size;
    return TECOOPS_STATUS_SUCCESS;
}

tecoopsStatus_t tecoopsTensorStruct::setTensorDimC(int dim_size) {
    checkTecoopsStatus(check());

    int index;
    switch (format) {
        case TECOOPS_TENSOR_CHWN:
            index = 0;
            break;
        case TECOOPS_TENSOR_NCHW:
            index = 1;
            break;
        case TECOOPS_TENSOR_NHWC:
        case TECOOPS_TENSOR_NWHC:
            index = 3;
            break;
        default:
            ERROR("tecoops setTensorDimC illegal format: %d\n", format);
            return TECOOPS_STATUS_BAD_PARAM;
    }
    dimA[index] = dim_size;
    c = dim_size;
    return TECOOPS_STATUS_SUCCESS;
}

tecoopsStatus_t tecoopsTensorStruct::setTensorStrideN(int stride_size) {
    checkTecoopsStatus(check());

    int index;
    switch (format) {
        case TECOOPS_TENSOR_NCHW:
        case TECOOPS_TENSOR_NHWC:
        case TECOOPS_TENSOR_NWHC:
            index = 0;
            break;
        case TECOOPS_TENSOR_CHWN:
            index = 3;
            break;
        default:
            ERROR("tecoops setTensorStrideN illegal format: %d\n", format);
            return TECOOPS_STATUS_BAD_PARAM;
    }
    strideA[index] = stride_size;
    nStride = stride_size;
    return TECOOPS_STATUS_SUCCESS;
}

tecoopsStatus_t tecoopsTensorStruct::setTensorStrideH(int stride_size) {
    checkTecoopsStatus(check());

    int index;
    switch (format) {
        case TECOOPS_TENSOR_NHWC:
        case TECOOPS_TENSOR_CHWN:
            index = 1;
            break;
        case TECOOPS_TENSOR_NCHW:
        case TECOOPS_TENSOR_NWHC:
            index = 2;
            break;
        default:
            ERROR("tecoops setTensorStrideH illegal format: %d\n", format);
            return TECOOPS_STATUS_BAD_PARAM;
    }
    strideA[index] = stride_size;
    hStride = stride_size;
    return TECOOPS_STATUS_SUCCESS;
}

tecoopsStatus_t tecoopsTensorStruct::setTensorStrideW(int stride_size) {
    checkTecoopsStatus(check());

    int index;
    switch (format) {
        case TECOOPS_TENSOR_NWHC:
            index = 1;
            break;
        case TECOOPS_TENSOR_NHWC:
        case TECOOPS_TENSOR_CHWN:
            index = 2;
            break;
        case TECOOPS_TENSOR_NCHW:
            index = 3;
            break;
        default:
            ERROR("tecoops setTensorStrideW illegal format: %d\n", format);
            return TECOOPS_STATUS_BAD_PARAM;
    }
    strideA[index] = stride_size;
    wStride = stride_size;
    return TECOOPS_STATUS_SUCCESS;
}

tecoopsStatus_t tecoopsTensorStruct::setTensorStrideC(int stride_size) {
    checkTecoopsStatus(check());

    int index;
    switch (format) {
        case TECOOPS_TENSOR_CHWN:
            index = 0;
            break;
        case TECOOPS_TENSOR_NCHW:
            index = 1;
            break;
        case TECOOPS_TENSOR_NHWC:
        case TECOOPS_TENSOR_NWHC:
            index = 3;
            break;
        default:
            ERROR("tecoops setTensorStrideC illegal format: %d\n", format);
            return TECOOPS_STATUS_BAD_PARAM;
    }
    strideA[index] = stride_size;
    cStride = stride_size;
    return TECOOPS_STATUS_SUCCESS;
}

tecoopsStatus_t tecoopsTensorStruct::getTensorSizeInBytes(size_t* tensor_size) {
    *tensor_size = 1;
    for (int i = nbDims - 1; i >= 0; i--) {
        *tensor_size += (dimA[i] - 1) * strideA[i];
    }
    *tensor_size *= Convert::toDescDataTypeSize(dataType);
    return TECOOPS_STATUS_SUCCESS;
}
