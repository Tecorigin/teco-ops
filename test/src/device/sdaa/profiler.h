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
 
#ifndef DEVICE_SDAA_PROFILER_H_  // NOLINT
#define DEVICE_SDAA_PROFILER_H_

#include <stdio.h>
#include <stdlib.h>
#include <sdpti.h>
#include <chrono>
#include <iostream>
#include <utility>
#include <string>
#include <vector>
#include "common/tecoallog.h"

using TimeStamp = std::chrono::high_resolution_clock::time_point;

namespace optest {
namespace sdaa {

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
            ALLOG(ERROR) << "kernel func not the same in stablity test";
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
    static std::vector<std::string> cache_miss_result_;

 private:
    DeviceProfiler();
    ~DeviceProfiler();

    void init_tracing();
    static void bufferRequest(uint8_t **buffer, size_t *size, size_t *maxNumRecords);
    static void bufferCompleted(uint8_t *buffer, size_t size, size_t validSize);
    static void get(const SDpti_Activity *record);
    void deal();
};

}  // namespace sdaa
}  // namespace optest

#endif  // DEVICE_SDAA_PROFILER_H_  // NOLINT
