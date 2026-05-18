#include <cstdlib>
#include <sdaa_runtime.h>
#include "interface/include/tecoops.h"
#include "interface/common/handle.h"

tecoopsStatus_t tecoopsCreate(tecoopsHandle_t *handle) {
    if (handle == nullptr) {
        return TECOOPS_STATUS_BAD_PARAM;
    }
    tecoopsContext *ctx = new tecoopsContext();
    if (ctx == nullptr) {
        return TECOOPS_STATUS_ALLOC_FAILED;
    }
    ctx->spa_num = 1;
    ctx->spe_num = 32;
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

tecoopsStatus_t tecoopsGetStream(tecoopsHandle_t handle, sdaaStream_t *stream) {
    if (handle == nullptr || stream == nullptr) {
        return TECOOPS_STATUS_BAD_PARAM;
    }
    *stream = handle->stream;
    return TECOOPS_STATUS_SUCCESS;
}

const char *tecoopsGetErrorString(tecoopsStatus_t status) {
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