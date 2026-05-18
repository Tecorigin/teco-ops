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
 
#include <string>
#include "common/variable.h"
#include "device/device.h"
#include "suite/env.h"

extern optest::GlobalVar global_var;
void TestEnvironment::SetUp() {
    // 1. set up env
    ALLOG(INFO) << "SetUp environment.";

    // 2. get device total num
    int dev_num = 0;
    checkScdaErrors(scdaGetDeviceCount(&dev_num));

    // 3. random device id [0, dev_num)
    // global_var.dev_id_ = getCardId(dev_num);
    if (global_var.dev_id_ < 0 || global_var.dev_id_ >= dev_num) {
        ALLOG(ERROR) << "device id " << global_var.dev_id_ << " must >= 0 and < " << dev_num;
        throw std::invalid_argument(std::string(__FILE__) + " +" + std::to_string(__LINE__));
    }
    if (global_var.kernel_id_ < 0 || global_var.kernel_id_ >= dev_num) {
        ALLOG(ERROR) << "kernel id " << global_var.kernel_id_ << " must >= 0 and < " << dev_num;
        throw std::invalid_argument(std::string(__FILE__) + " +" + std::to_string(__LINE__));
    }

    checkScdaErrors(scdaSetDevice(global_var.dev_id_));
    // set current device
    ALLOG(INFO) << "Set current device as device: " << global_var.dev_id_;
}

void TestEnvironment::TearDown() {
    ALLOG(INFO) << "TearDown environment.";

    auto summary = global_var.summary_;
    std::cout << "[ SUMMARY  ] "
              << "Total " << summary.case_count << " cases of " << summary.suite_count
              << " op(s).\n";
    if (summary.failed_list.empty()) {
        std::cout << "ALL PASSED.\n";
    } else {
        auto case_list = summary.failed_list;
        std::cout << case_list.size() << " CASES FAILED:\n";
        for (auto it = case_list.begin(); it != case_list.end(); ++it) {
            std::cout << "Failed: " << (*it) << "\n";
        }
    }
}
