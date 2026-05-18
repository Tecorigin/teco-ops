// MIT License
// 
// Copyright (c) 2024, Tecorigin Co., Ltd.
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
 
#include <sched.h>
#include <sdaa_runtime.h>

#include <sdpti.h>
#ifdef SDPTI_VERSION_MAJOR  // old sdpti
#include <sdaa_prof_str.h>
#endif

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

#include "common/variable.h"
#include "device/sdaa/profiler.h"

extern optest::GlobalVar global_var;

using TimeStamp = std::chrono::high_resolution_clock::time_point;
#define max(a, b) (((a) > (b)) ? (a) : (b))

#define CHECK_RET(ret)                                                  \
    do {                                                                \
        SDptiResult value = ret;                                        \
        if (value != SDPTI_SUCCESS) {                                   \
            printf("Call %s failed, ret is %d\n", __FUNCTION__, value); \
        }                                                               \
    } while (0);

namespace optest {
namespace sdaa {

#ifndef SDPTI_VERSION_MAJOR
const char *GetApiName(SDpti_CallbackDomain domain, SDpti_CallbackId cbid) {
    SDptiResult ret = SDPTI_SUCCESS;
    const char *name;
    ret = sdptiGetCallbackName(domain, cbid, &name);
    if (ret != SDPTI_SUCCESS) {
        return "unknown";
    }
    return name;
}
#endif

std::vector<std::pair<std::string, double>> DeviceProfiler::result_;
std::vector<std::string> DeviceProfiler::cache_miss_result_;

#ifndef SDPTI_VERSION_MAJOR
void DeviceProfiler::start() {
    init_tracing();
    result_.clear();
    cache_miss_result_.clear();
    start_ = std::chrono::high_resolution_clock::now();
}

void DeviceProfiler::end() {
    end_ = std::chrono::high_resolution_clock::now();
    sdptiActivityFlushAll(0);
    sdptiFinalize();
}
#else
void DeviceProfiler::start() {
    sdptiActivityFlushAll(0);
    result_.clear();
    start_ = std::chrono::high_resolution_clock::now();
}

void DeviceProfiler::end() {
    end_ = std::chrono::high_resolution_clock::now();
    sdptiActivityFlushAll(0);
}
#endif

ProfilePerfInfo DeviceProfiler::duration() {
    info_.interface_time =
        std::chrono::duration_cast<std::chrono::microseconds>(end_ - start_).count();  // us
    deal();  // get launch and kernel time
    return info_;
}

#ifndef SDPTI_VERSION_MAJOR
DeviceProfiler::DeviceProfiler() { /*init_tracing();*/
}
#else
DeviceProfiler::DeviceProfiler() { init_tracing(); }
#endif

DeviceProfiler::~DeviceProfiler() {}

void DeviceProfiler::init_tracing() {
    sdptiInit();

#ifndef SDPTI_VERSION_MAJOR
    uint64_t buffer_size = 10 * 1024 * 1024, value_size{0};
    CHECK_RET(sdptiActivitySetAttribute(SDPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &value_size,
                                        &buffer_size));
    uint64_t id = global_var.dev_id_;
    uint64_t device_count = 1;
    CHECK_RET(sdptiActivitySetAttribute(SDPTI_ACTIVITY_ATTR_DEVICE_ENABLED_ID, &device_count, &id));

    CHECK_RET(sdptiActivityEnable(SDPTI_ACTIVITY_KIND_ICACHE_MISS));
#endif
    CHECK_RET(sdptiActivityEnable(SDPTI_ACTIVITY_KIND_RUNTIME));
    CHECK_RET(sdptiActivityEnable(SDPTI_ACTIVITY_KIND_KERNEL));
    sdptiActivityRegisterCallbacks(bufferRequest, bufferCompleted);
}

void DeviceProfiler::bufferRequest(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
    // sizeof(int64) * 20 * 3(launch+kernel+sync) * 10(max kernel count) * 1000(repeats) < 10M
#define BUF_SIZE (10 * 1024 * 1024)

    uint8_t *bfr = (uint8_t *)malloc(BUF_SIZE);

    *size = BUF_SIZE;
    *buffer = bfr;
    *maxNumRecords = 0;
}

void DeviceProfiler::bufferCompleted(uint8_t *buffer, size_t size, size_t validSize) {
    SDptiResult status;
    SDpti_Activity *record = NULL;

    if (validSize > 0) {
        do {
            status = sdptiActivityGetNextRecord(buffer, validSize, &record);
            if (status == SDPTI_SUCCESS) {
                get(record);
            } else if (status == SDPTI_ERROR_MAX_LIMIT_REACHED) {
                break;
            }
        } while (1);
    }
#ifndef SDPTI_VERSION_MAJOR
    free(buffer);
#endif
}

void DeviceProfiler::get(const SDpti_Activity *record) {
    switch (record->kind) {
        case SDPTI_ACTIVITY_KIND_RUNTIME: {
            SDpti_ActivityAPI *api = (SDpti_ActivityAPI *)record;
#ifndef SDPTI_VERSION_MAJOR
            auto name = GetApiName(SDPTI_CB_DOMAIN_RUNTIME_API, api->cbid);
#else
            auto name = sdaa_api_name(api->cbid);
#endif
            auto time = (unsigned long long)(api->end) - (unsigned long long)(api->start);
            result_.push_back(std::pair<std::string, double>(name, time));

            break;
        }
        case SDPTI_ACTIVITY_KIND_KERNEL: {
            SDpti_ActivityKernel *kernel = (SDpti_ActivityKernel *)record;
            auto name = kernel->name;
#ifndef SDPTI_VERSION_MAJOR
            double time = kernel->end - kernel->start;
            // printf("correlation %llu, kernel %s, time %f\n", kernel->runtimeCorrelationId, name,
            //        time);
#else
            double time = kernel->hwExcutionTime;
#endif
            result_.push_back(std::pair<std::string, double>(name, time));
            break;
        }
#ifndef SDPTI_VERSION_MAJOR
        case SDPTI_ACTIVITY_KIND_ICACHE_MISS: {
            SDpti_ActivityICacheMiss *icache_miss = (SDpti_ActivityICacheMiss *)record;
            char buffer[1024];
            snprintf(buffer, sizeof(buffer),
                     "correlation %lu, access %lu, miss %lu, avg miss ns %.2f, "
                     "max miss ns %lu, core_id = %u",
                     icache_miss->runtimeCorrelationId, icache_miss->icacheAccess,
                     icache_miss->icacheMiss, icache_miss->icacheMissAveTime,
                     icache_miss->icacheMissMaxTime, icache_miss->icacheMissMaxTimeCoreId);
            cache_miss_result_.push_back(std::string(buffer));
            break;
        }
#endif
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
        if (name == "sdaaLaunchKernel") {
            info_.launch_time += time;
            info_.launch_count++;
        } else if (name == "sdaaStreamSynchronize" || name == "sdaaDeviceSynchronize") {
            // ignore
        } else {
            info_.kernel_time += time;
            info_.kernel_details.push_back(
                std::pair<std::string, double>(name, time / 1000));  // us
        }
    }

    info_.kernel_time /= 1000;  // us
    info_.launch_time /= 1000;  // us

    info_.cache_miss_details = cache_miss_result_;
}

}  // namespace sdaa
}  // namespace optest
