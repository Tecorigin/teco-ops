#include "ual/ops/flatten_rays/flatten_rays.hpp"

#include "interface/common/convert.h"
#include "interface/common/macro.h"
#include "interface/include/builtin_type.h"
#include "interface/include/tecoops.h"
#include "ual/args/flatten_rays_args.h"

using tecoops::Convert;
using tecoops::ual::args::FlattenRaysArgs;
using tecoops::ual::args::FlattenRaysPatchArgs;
using tecoops::ual::ops::FlattenRaysOp;

tecoopsStatus_t tecoopsFlattenRays(tecoopsHandle_t handle, const int* rays, uint32_t N, uint32_t M,
                                   int* res, tecoopsAlgo_t algo) {
    if (handle == nullptr) {
        return TECOOPS_STATUS_NOT_INITIALIZED;
    }

    FlattenRaysArgs arg;
    arg.spe_num = handle->spe_num;
    arg.rays = rays;
    arg.N = N;
    arg.M = M;
    arg.res = res;

    FlattenRaysPatchArgs patch_arg;
    patch_arg.atargs = &arg;
    patch_arg.data_type = tecoops::ual::common::UAL_DTYPE_INT32;
    patch_arg.algo = Convert::toUALAlgoType(algo);

    RUN_OP(FlattenRaysOp, arg, patch_arg, handle);

    return TECOOPS_STATUS_SUCCESS;
}