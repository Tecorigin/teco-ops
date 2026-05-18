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
 
#ifndef COMMON_CONTEXT_H_  // NOLINT
#define COMMON_CONTEXT_H_

#include <memory>
#include <string>

namespace optest {

class Context {
 public:
    static const Context *instance() {
        static Context context;
        return &context;
    }

    Context() {
        print_fault_flag_ = getEnv("DNN_PRINT_FAULT_FLAG");

        print_fault_num_ = getEnv("DNN_PRINT_FAULT_NUM");
        print_fault_num_ = print_fault_num_ == 0 ? 30 : print_fault_num_;

        compare_with_gpu_ = getEnv("DNN_NOT_COMPARE_WITH_GPU");
        compare_with_gpu_ = !compare_with_gpu_;

        reserve_data_ = getEnv("DNN_RESERVE_DATA");
        show_all_errors_ = getEnv("DNN_SHOW_ALL_ERRORS");
        show_test_log_ = getEnv("DNN_TEST_LOG");
        check_all_mem_flag_ = getEnv("DNN_CHECK_ALL_MEM");
        only_test_performance_ = getEnv("DNN_ONLY_TEST_PERFORMANCE");
        save_device_output_ = getEnv("DNN_SAVE_DEVICE_OUTPUT");
        device_output_dir_ = getStrEnv("DNN_SAVE_DEVICE_DIR", "dumpteco");
        not_init_input_ = getEnv("DNN_NOT_INIT_INPUT");
        set_nan_with_beta_ = getEnv("DNN_SET_NAN_WITH_BETA");
        save_unstable_result_ = getEnv("DNN_SAVE_UNSTABLE_RESULT");
        unstable_result_dir_ = getStrEnv("DNN_SAVE_UNSTABLE_DIR", "unstable_result");
    }
    ~Context() {}

    bool printFaultFlag() const { return print_fault_flag_; }
    size_t printFaultNum() const { return print_fault_num_; }
    bool compareWithGPU() const { return compare_with_gpu_; }
    bool reserveDataFlag() const { return reserve_data_; }
    bool showAllErrors() const { return show_all_errors_; }
    bool showTestLog() const { return show_test_log_; }
    bool checkAllMemory() const { return check_all_mem_flag_; }
    bool onlyTestPerformance() const { return only_test_performance_; }
    bool saveDeviceOutput() const { return save_device_output_; }
    std::string getDeviceOutputDir() const { return device_output_dir_; }
    bool notInitInput() const { return not_init_input_; }
    bool setNanWithBeta() const { return set_nan_with_beta_; }
    bool saveUnstableResult() const { return save_unstable_result_; }
    std::string getUnstableResultDir() const { return unstable_result_dir_; }

 protected:
    int getEnv(const char *name) {
        const char *env = std::getenv(name);
        if (env == nullptr) {
            return 0;
        }
        return std::atoi(env);
    }

    std::string getStrEnv(const char *name, const char *default_str) {
        const char *env = std::getenv(name);
        if (env == nullptr) {
            return std::string(default_str);
        }
        return std::string(env);
    }

 private:
    bool print_fault_flag_ = false;
    size_t print_fault_num_ = 30;
    bool compare_with_gpu_ = true;
    bool reserve_data_ = false;
    bool show_all_errors_ = false;
    bool show_test_log_ = false;
    bool check_all_mem_flag_ = false;
    bool only_test_performance_ = false;
    bool save_device_output_ = false;
    std::string device_output_dir_;
    bool not_init_input_ = false;
    bool set_nan_with_beta_ = false;
    bool save_unstable_result_ = false;
    std::string unstable_result_dir_;
};

}  // namespace optest

#endif  // COMMON_CONTEXT_H_  // NOLINT
