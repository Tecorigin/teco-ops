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
 
#ifndef COMMON_VARIABLE_H_  // NOLINT
#define COMMON_VARIABLE_H_
#include <list>
#include <string>
#include <cctype>
#include <algorithm>
#include "common/tecoallog.h"

namespace optest {

struct TestSummary {
    size_t case_count = 0;
    size_t suite_count = 0;
    std::list<std::string> failed_list;
};

class GlobalVar {
 public:
    std::string cases_dir_ = "";
    std::string cases_list_ = "";
    std::string case_path_ = "";
    TestSummary summary_;

    int dev_id_ = -1;  // the picked device id, make sure memory on the picked device.
    int kernel_id_ =
        -1;  // the picked kernel id, make sure run on the picked device. if -1, use dev_id.
    int rand_n_ = -1;  // pick n * random case, -1 for uninitialized
    int repeat_ = 0;   // perf-repeat repeat * kernel enqueue sdaartQueue_t, and get
                       // ave hw_time
    int warm_repeat_ = 0;
    bool shuffle_ = false;  // shuffle cases.
    bool stable_ = false;   // stability test
    int stable_repeat_ = 0;

    std::string getParam(const std::string &str, std::string key) {
        key = key + "=";
        auto npos = str.find(key);
        if (npos == std::string::npos) {
            return "";
        } else {
            return str.substr(npos + key.length());
        }
    }

    void init(int argc, char **argv) {
        auto to_int = [](std::string s, std::string opt) -> int {
            if (std::count_if(s.begin(), s.end(),
                              [](unsigned char c) { return std::isdigit(c); }) == s.size() &&
                !s.empty()) {
                return std::atoi(s.c_str());
            } else {
                return -1;
            }
        };
        for (int i = 0; i < argc; i++) {
            std::string arg = argv[i];
            cases_dir_ = cases_dir_.empty() ? getParam(arg, "--cases_dir") : cases_dir_;
            cases_list_ = cases_list_.empty() ? getParam(arg, "--cases_list") : cases_list_;
            case_path_ = case_path_.empty() ? getParam(arg, "--case_path") : case_path_;
            dev_id_ = (dev_id_ == -1) ? to_int(getParam(arg, "--gid"), "--gid") : dev_id_;
            kernel_id_ = (kernel_id_ == -1) ? to_int(getParam(arg, "--kid"), "--kid") : kernel_id_;
            rand_n_ = (rand_n_ == -1) ? to_int(getParam(arg, "--rand_n"), "--rand_n") : rand_n_;
            repeat_ = getParam(arg, "--perf_repeat").empty() ?
                          repeat_ :
                          to_int(getParam(arg, "--perf_repeat"), "--perf_repeat");
            // repeat_ = repeat_ > 1000 ? 1000 : repeat_;
            warm_repeat_ = getParam(arg, "--warm_repeat").empty() ?
                               warm_repeat_ :
                               to_int(getParam(arg, "--warm_repeat"), "--warm_repeat");

            shuffle_ =
                (shuffle_ == false) ? (arg.find("--gtest_shuffle") != std::string::npos) : shuffle_;

            stable_ =
                (stable_ == false) ? (arg.find("--test_stable") != std::string::npos) : stable_;

            stable_repeat_ = getParam(arg, "--stable_repeat").empty() ?
                                 stable_repeat_ :
                                 to_int(getParam(arg, "--stable_repeat"), "--stable_repeat");
        }
        if (dev_id_ == -1) {
            ALLOG(ERROR) << "must set device_id" << std::endl;
            std::cout << "--gid=[id]" << std::endl;
            std::cout << "  set device id" << std::endl;
            // throw std::invalid_argument(std::string(__FILE__) + " +" + std::to_string(__LINE__));
        }
        kernel_id_ = (kernel_id_ == -1) ? dev_id_ : kernel_id_;
        if (dev_id_ / 4 != kernel_id_ / 4) {
            ALLOG(ERROR) << "gid and kid must in same physical device.";
            throw std::invalid_argument(std::string(__FILE__) + " +" + std::to_string(__LINE__));
        }
        // print();
    }

    void print() {
        std::cout << "cases_dir is " << cases_dir_ << std::endl;
        std::cout << "cases_list is " << cases_list_ << std::endl;
        std::cout << "cases_path is " << case_path_ << std::endl;
        std::cout << "rand_n is " << rand_n_ << std::endl;
        std::cout << "repeat is " << repeat_ << std::endl;
        std::cout << "shuffle is " << shuffle_ << std::endl;
    }
};

}  // namespace optest

#endif  // COMMON_VARIABLE_H_  // NOLINT
