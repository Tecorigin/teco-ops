#ifndef DEVICE_CUDA_PROFILER_H_  // NOLINT
#define DEVICE_CUDA_PROFILER_H_

#include <cupti.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <utility>
#include <string>
#include <vector>
#include "common/tecoallog.h"

using TimeStamp = std::chrono::high_resolution_clock::time_point;

namespace optest {
namespace cuda {

class ProfilePerfInfo {
 public:
    double interface_time;
    double kernel_time;
    double launch_time;
    int launch_count;
    std::vector<std::pair<std::string, double>> kernel_details;
    std::vector<std::string> cache_miss_details;

    ProfilePerfInfo &operator+(const ProfilePerfInfo &perf_info) {
        this->interface_time += perf_info.interface_time;
        this->kernel_time += perf_info.kernel_time;
        this->launch_time += perf_info.launch_time;
        this->launch_count += perf_info.launch_count;
        if (perf_info.kernel_details.size() != kernel_details.size()) {
            ALLOG(ERROR) << "slave func not the same in stablity test";
        } else {
            for (int i = 0; i < kernel_details.size(); i++) {
                this->kernel_details[i].second += perf_info.kernel_details[i].second;
            }
        }
        return *this;
    }

    ProfilePerfInfo operator/(int count) {
        this->interface_time /= count;
        this->kernel_time /= count;
        this->launch_time /= count;
        this->launch_count /= count;
        return *this;
    }
};

class DeviceProfiler {
 public:
    static DeviceProfiler *instance() {
        static thread_local DeviceProfiler profiler;
        return &profiler;
    }

    void start();
    void end();
    ProfilePerfInfo duration();

 private:
    TimeStamp start_;
    TimeStamp end_;
    ProfilePerfInfo info_;
    static std::vector<std::pair<std::string, double>> result_;

 private:
    DeviceProfiler();

    void init_tracing();
    static void bufferRequest(uint8_t **buffer, size_t *size, size_t *maxNumRecords);
    static void bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size,
                                size_t validSize);
    static void get(const CUpti_Activity *record);
    void deal();
};

}  // namespace cuda
}  // namespace optest

#endif  // DEVICE_CUDA_PROFILER_H_  // NOLINT
