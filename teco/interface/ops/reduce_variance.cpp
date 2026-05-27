#include "ual/ops/reduce_variance/reduce_variance.hpp"

#include "interface/common/check.h"
#include "interface/common/convert.h"
#include "interface/common/macro.h"
#include "interface/include/builtin_type.h"
#include "interface/include/tecoops.h"
#include "ual/args/reduce_variance_args.h"

using tecoops::Convert;
using tecoops::ual::args::ReduceVarianceArgs;
using tecoops::ual::args::ReduceVariancePatchArgs;
using tecoops::ual::ops::ReduceVarianceOp;

// check the compliance of the args of reduce variance operator
static tecoopsStatus_t reduceVarianceCheckArgs(tecoopsHandle_t handle, int axis, int correction,
                                               const tecoopsTensorDescriptor_t xDesc, const void* x,
                                               const tecoopsTensorDescriptor_t yDesc, void* y) {
    if (handle == nullptr) {
        ERROR("NULL Error: tecoopsHandle_t is nullptr \n");
        return TECOOPS_STATUS_NOT_INITIALIZED;
    }

    if (xDesc == nullptr || x == nullptr || yDesc == nullptr || yDesc == nullptr) {
        ERROR("NULL Error: some params are nullptr \n");
        return TECOOPS_STATUS_BAD_PARAM;
    }

    if (xDesc->dataType != yDesc->dataType) {
        ERROR("x data_type must be equal to y data_type \n");
        return TECOOPS_STATUS_BAD_PARAM;
    }

    if (!((xDesc->dataType == TECOOPS_DATA_FLOAT || xDesc->dataType == TECOOPS_DATA_HALF))) {
        ERROR("x y must be TECOOPS_DATA_FLOAT or TECOOPS_DATA_HALF\n");
        return TECOOPS_STATUS_BAD_PARAM;
    }

    if (axis < 0 || axis > xDesc->nbDims) {
        ERROR("axis is illogic \n");
        return TECOOPS_STATUS_BAD_PARAM;
    }

    if ((unsigned long long)x % 4 != 0 || (unsigned long long)y % 4 != 0) {  // NOLINT
        ERROR("x y address is not 4B alignment\n");
        return TECOOPS_STATUS_BAD_PARAM;
    }

    if (correction != 0 && correction != 1) {
        ERROR("value of correction must be 0 or 1\n");
        return TECOOPS_STATUS_INVALID_VALUE;
    }

    if ((xDesc->dimA[axis] - correction) <= 0) {
        ERROR("correction should be strictly less than the reduction factor\n");
        return TECOOPS_STATUS_INVALID_VALUE;
    }

    if (yDesc->dimA[axis] != 1) {
        ERROR("channel of yDesc's axis is not 1\n");
        return TECOOPS_STATUS_BAD_PARAM;
    }

    for (int i = 0; i < xDesc->nbDims; ++i) {
        if ((i != axis) && (xDesc->dimA[i] != yDesc->dimA[i])) {
            ERROR("channel of xDesc and yDesc is not equal\n");
            return TECOOPS_STATUS_BAD_PARAM;
        }
        if (xDesc->dimA[axis] <= 0) {
            ERROR("channel of xDesc exists zero or negative\n");
            return TECOOPS_STATUS_BAD_PARAM;
        }
    }
    return TECOOPS_STATUS_SUCCESS;
}

tecoopsStatus_t tecoopsReduceVariance(const tecoopsHandle_t handle, int axis, int correction,
                                      const tecoopsTensorDescriptor_t xDesc, const void* x,
                                      const tecoopsTensorDescriptor_t yDesc, void* y) {
    auto input_error = reduceVarianceCheckArgs(handle, axis, correction, xDesc, x, yDesc, y);
    checkTecoopsStatus(input_error);
    // init
    ReduceVarianceArgs arg;
    arg.spe_num = handle->spe_num;
    arg.dims = xDesc->nbDims;

    // compute data_num
    arg.data_num = 1;
    for (int i = 0; i < arg.dims; ++i) {
        arg.data_num *= xDesc->dimA[i];
    }

    arg.matrix_num = 1;
    arg.matrix_col = 1;
    arg.matrix_row = 1;
    arg.correction = correction;

    if (arg.dims - 1 == axis) {
        arg.matrix_col = xDesc->dimA[axis];
        arg.matrix_row = arg.data_num / arg.matrix_col;
    } else if (0 < axis && axis < arg.dims - 1) {
        for (int i = 0; i < axis; i++) {
            arg.matrix_num *= xDesc->dimA[i];
        }
        arg.matrix_row = xDesc->dimA[axis];
        arg.matrix_col = arg.data_num / (arg.matrix_num * arg.matrix_row);
    } else {
        arg.matrix_row = xDesc->dimA[axis];
        arg.matrix_col = arg.data_num / arg.matrix_row;
    }

    arg.x = x;
    arg.y = y;

    ReduceVariancePatchArgs patch_arg;
    patch_arg.rvargs = &arg;
    patch_arg.data_type = Convert::toUALDataType(xDesc->dataType);
    patch_arg.axis = axis;
    patch_arg.dimA = xDesc->dimA;

    if (xDesc->dimA[axis] == 1) {
        int dataSize = xDesc->dataType == TECOOPS_DATA_FLOAT ? sizeof(float) : sizeof(half);
        return tecoopsMemset(handle, arg.y, 0, static_cast<size_t>(arg.data_num * dataSize));
    }

    RUN_OP(ReduceVarianceOp, arg, patch_arg, handle);
    return TECOOPS_STATUS_SUCCESS;
}
