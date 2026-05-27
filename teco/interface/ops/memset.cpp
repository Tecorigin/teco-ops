
#include "ual/ops/memset/memset.hpp"

#include "interface/common/convert.h"
#include "interface/common/macro.h"
#include "interface/include/builtin_type.h"
#include "interface/include/tecoops.h"
#include "ual/args/memset_args.h"

using tecoops::Convert;
using tecoops::ual::args::MemsetArgs;
using tecoops::ual::ops::MemsetOp;

static tecoopsStatus_t checkMemsetInput(tecoopsHandle_t handle, void* x, const int value,
                                        size_t size) {
    if (handle == NULL) {
        return TECOOPS_STATUS_BAD_PARAM;
    }
    if (x == NULL) {
        return TECOOPS_STATUS_BAD_PARAM;
    }
    return TECOOPS_STATUS_SUCCESS;
}

tecoopsStatus_t tecoopsMemset(tecoopsHandle_t handle, void* x, const int value, size_t size) {
    // check input
    tecoopsStatus_t input_error = checkMemsetInput(handle, x, value, size);
    if (input_error != TECOOPS_STATUS_SUCCESS)
        return input_error;

    MemsetArgs mem_arg;

    mem_arg.spe_num = handle->spe_num;
    mem_arg.data_size = size;
    mem_arg.value = value;
    mem_arg.x = x;

    RUN_OP(MemsetOp, mem_arg, mem_arg, handle);
    return TECOOPS_STATUS_SUCCESS;
}
