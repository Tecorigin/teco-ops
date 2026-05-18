#include "device/cuda/profiler.h"
#include <sched.h>
#include <cuda.h>
#include <cupti.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <utility>
#include <vector>

using TimeStamp = std::chrono::high_resolution_clock::time_point;
#define max(a, b) (((a) > (b)) ? (a) : (b))

#define CUPTI_CALL(call)                                                                   \
    do {                                                                                   \
        CUptiResult _status = call;                                                        \
        if (_status != CUPTI_SUCCESS) {                                                    \
            const char *errstr;                                                            \
            cuptiGetResultString(_status, &errstr);                                        \
            fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", __FILE__, \
                    __LINE__, #call, errstr);                                              \
            exit(EXIT_FAILURE);                                                            \
        }                                                                                  \
    } while (0)

namespace optest {
namespace cuda {

std::vector<std::pair<std::string, double>> DeviceProfiler::result_;

void DeviceProfiler::start() {
    result_.clear();
    cuptiActivityFlushAll(0);
    start_ = std::chrono::high_resolution_clock::now();
}

void DeviceProfiler::end() {
    end_ = std::chrono::high_resolution_clock::now();
    cuptiActivityFlushAll(0);
}

ProfilePerfInfo DeviceProfiler::duration() {
    info_.interface_time =
        std::chrono::duration_cast<std::chrono::microseconds>(end_ - start_).count();  // us
    deal();  // get launch and kernel time
    return info_;
}

DeviceProfiler::DeviceProfiler() { init_tracing(); }

void DeviceProfiler::init_tracing() {
    cuptiActivityRegisterCallbacks(bufferRequest, bufferCompleted);
    // cuInit(0);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME);  // not used
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
}

void DeviceProfiler::bufferRequest(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
#define BUF_SIZE (1 * 1024)
    uint8_t *bfr = (uint8_t *)malloc(BUF_SIZE);

    *size = BUF_SIZE;
    *buffer = bfr;
    *maxNumRecords = 0;
}

void DeviceProfiler::bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size,
                                     size_t validSize) {
    CUptiResult status;
    CUpti_Activity *record = NULL;

    if (validSize > 0) {
        do {
            status = cuptiActivityGetNextRecord(buffer, validSize, &record);
            if (status == CUPTI_SUCCESS) {
                get(record);
            } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
                break;
            } else {
                CUPTI_CALL(status);
            }
        } while (1);

        // report any records dropped from the queue
        size_t dropped;
        CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
        if (dropped != 0) {
            printf("Dropped %u activity records\n", (unsigned int)dropped);
        }
    }

    free(buffer);
}

void DeviceProfiler::get(const CUpti_Activity *record) {
    switch (record->kind) {
        case CUPTI_ACTIVITY_KIND_RUNTIME: {
            CUpti_ActivityAPI *api = (CUpti_ActivityAPI *)record;
            if (api->cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000) {
                auto name = "cudaLaunchKernel";
                double time = (unsigned long long)(api->end) - (unsigned long long)(api->start);
                result_.push_back(std::pair<std::string, double>(name, time));
            }
            break;

            /*
            211: CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000
            10:  CUPTI_RUNTIME_TRACE_CBID_cudaGetLastError_v3020
            131: CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_v3020
            */
        }
        case CUPTI_ACTIVITY_KIND_KERNEL:
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
            CUpti_ActivityKernel7 *kernel = (CUpti_ActivityKernel7 *)record;
            auto name = "slave_" + (std::string)kernel->name;
            double time = (unsigned long long)(kernel->end) - (unsigned long long)(kernel->start);
            result_.push_back(std::pair<std::string, double>(name, time));
            break;
        }
        default: break;
    }
}

void DeviceProfiler::deal() {
    info_.kernel_time = 0.0;
    info_.launch_time = 0.0;
    info_.launch_count = 0;
    info_.kernel_details.clear();

    for (auto iter = result_.begin(); iter != result_.end(); ++iter) {
        std::string name = iter->first;
        double time = iter->second;
        if (name == "cudaLaunchKernel") {
            info_.launch_time += time;
            // info_.launch_count++;
        } else if (name.find("slave") < name.length()) {
            info_.kernel_time += time;
            info_.launch_count++;
            info_.kernel_details.push_back(
                std::pair<std::string, double>(name, time / 1000));  // us
        }
    }

    info_.kernel_time /= 1000;  // us
    info_.launch_time /= 1000;  // us
}

}  // namespace cuda
}  // namespace optest
