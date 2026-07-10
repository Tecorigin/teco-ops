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
 
#ifndef CASE_EVALUATOR_H_  // NOLINT
#define CASE_EVALUATOR_H_

#include <algorithm>
#include <iostream>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <string>
#include <utility>
#include <memory>
#include <map>
#include "common/tools.h"
#include "common/context.h"
#include "common/half.hpp"
#include "case/pb_tools.h"
#include "case/criterion.h"
#include "case/parser.h"
#ifdef __AVX__
#include <immintrin.h>
#include <limits>
#endif

#include "nlohmann/json.hpp"  // NOLINT

using json = nlohmann::json;

namespace optest {
// determine whether the test case passes.
// this class receives:
// 1.output data --> error
// 2.latency of device
// 3.io size/io bandwidth  --> io efficiency
// 4.ops/peak force  --> compute efficiency
// 5.interface time --> print
// 6.workspace --> print
constexpr double EPSILON = 1e-9;
constexpr double EPSILON_FLOAT = 1e-6;
constexpr double EPSILON_HALF = 1e-4;
constexpr int THREAD_NUM = 32;

#define COMPARE_PARALELL

struct Error {
    Error() {}
    explicit Error(double me) : max_error(me) {}

    // this is for diff3
    double error_ratio = 0;
    size_t error_num = 0;
    std::vector<double> error_thresholds{1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2};
    std::vector<double> error_ratios{0, 0, 0, 0, 0, 0};
    // end

    double max_error;
    int64_t index = -1;
    double baseline_value = 0;
    double compare_value = 0;
    std::vector<double> error_vect;
    bool has_nan = false;
    bool has_inf = false;
};

// pack 1 tensor's name/ criterion(func/threshold)/ diff together.
struct ErrorWrap {
    ErrorWrap() {}
    ErrorWrap(std::string name, size_t num, Criterion c, Error e_teco) :
        name(name), num(num), criterion(c), error_teco(e_teco) {}
    ErrorWrap(std::string name, size_t num, Criterion c, Error e_teco, Error e_gpu) :
        name(name), num(num), criterion(c), error_teco(e_teco), error_gpu(e_gpu) {}
    std::string name = "";  // tensor's name
    size_t num = 0;         // tensor's data num
    Criterion criterion;    // criterion
    Error error_teco;       // the error of this criterion
    Error error_gpu;        // the error of this criterion of gpu

    void print() {
        // std::cout << "[" << name << "]:" << std::endl;
        if (error_teco.has_nan || error_gpu.has_nan) {
            std::cout << "func:                 " << "/" << std::endl;
            std::cout << "enbale:               " << "/" << std::endl;
            std::cout << "teco_max_error:       " << "/" << std::endl;
            std::cout << "gpu_max_error:        " << "/" << std::endl;
        } else {
            std::cout << "func:                 " << showFormula(criterion.formula) << std::endl;
            std::cout << "enbale:               " << criterion.enable << std::endl;
            std::cout << "teco_max_error:       " << std::scientific << std::setprecision(5)
                      << error_teco.max_error << std::endl;
            std::cout << "gpu_max_error:        " << std::scientific << std::setprecision(5)
                      << error_gpu.max_error << std::endl;
        }

        std::cout << "golden_threshold:     " << std::fixed << std::setprecision(2)
                  << criterion.golden_threshold << std::endl;
        if (criterion.formula == DIFF3 || criterion.formula == DIFF3_MAX) {
            std::cout << std::setiosflags(std::ios::left) << std::setw(5) << "" << std::setw(10)
                      << "index" << std::setw(18) << "device_value" << std::setw(18)
                      << "baseline_value" << std::endl;
            std::cout << std::setiosflags(std::ios::left) << std::scientific << std::setw(5)
                      << "teco" << std::setw(10) << error_teco.index << std::setprecision(8)
                      << std::setw(18) << error_teco.compare_value << std::setw(18)
                      << error_teco.baseline_value << std::endl;
            std::cout << std::setiosflags(std::ios::left) << std::scientific << std::setw(5)
                      << "gpu" << std::setw(10) << error_gpu.index << std::setprecision(8)
                      << std::setw(18) << error_gpu.compare_value << std::setw(18)
                      << error_gpu.baseline_value << std::endl;
        }
    }

    // this is only for diff3
    void print_diff3() {
        if (error_teco.has_nan) {
            std::cout << "func:                 " << "/" << std::endl;
            std::cout << "enbale:               " << "/" << std::endl;
            std::cout << "max_error_threshold:  " << "/" << std::endl;
            std::cout << "max_error:            " << "/" << std::endl;
        } else {
            std::cout << "func:                 " << showFormula(criterion.formula) << std::endl;
            std::cout << "enbale:               " << criterion.enable << std::endl;
            std::cout << "max_error_threshold:  " << std::scientific << std::setprecision(2)
                      << criterion.max_error << std::endl;
            std::cout << "max_error:            " << std::scientific << std::setprecision(5)
                      << error_teco.max_error << std::endl;
        }

        if ((criterion.formula == DIFF3 || criterion.formula == DIFF3_MAX) && !error_teco.has_nan) {
            std::cout << "error_threshold:      " << std::scientific << std::setprecision(2)
                      << criterion.error_threshold << std::endl;
            std::cout << "ratio_threshold:      " << std::scientific << std::setprecision(2)
                      << criterion.ratio_threshold << std::endl;
            std::cout << "error_ratio:          " << std::scientific << std::setprecision(2)
                      << error_teco.error_ratio << std::endl;
            std::cout << "error_num:            " << error_teco.error_num << std::endl;
            std::cout << "total_num:            " << num << std::endl;

            std::cout << std::scientific << std::setiosflags(std::ios::left) << std::setw(20)
                      << "error_thresholds:";
            for (int i = 0; i < error_teco.error_thresholds.size(); i++) {
                std::cout << std::scientific << std::setw(10) << std::setprecision(0)
                          << error_teco.error_thresholds[i] << " ";
            }
            std::cout << std::endl;
            std::cout << std::scientific << std::setiosflags(std::ios::left) << std::setw(20)
                      << "ratio_thresholds:";
            for (int i = 0; i < error_teco.error_thresholds.size(); i++) {
                std::cout << std::scientific << std::setw(10) << std::setprecision(2)
                          << error_teco.error_ratios[i] << " ";
            }
        }
        std::cout << std::endl;
    }
};

typedef enum {
    TECOTEST_STATUS_SUCCESS = 0,
    TECOTEST_STATUS_EXECUTE_ERROR = 1,
    TECOTEST_STATUS_RESULT_ERROR = 2,
    TECOTEST_STATUS_RESULT_NAN_ERROR = 3,
    TECOTEST_STATUS_RESULT_INF_ERROR = 4,
    TECOTEST_STATUS_STABILITY_ERROR = 5,
    TECOTEST_STATUS_MEMORY_OUT_ERROR1 = 6,
    TECOTEST_STATUS_MEMORY_OUT_ERROR2 = 7,
    TECOTEST_STATUS_KERNEL_NAME_ERROR = 8,
    TECOTEST_STATUS_FILE_OPEN_FAULT = 9,
    TECOTEST_STATUS_PARSE_PT_FAULT = 10,
} tecotestStatus_t;

std::string showTestStatus(tecotestStatus_t status);

template <typename T>
bool resetNanOrInfAsZero(T *teco_result, T *gpu_result, T *baseline_result, size_t count,
                         const std::string &name, Error *error_teco, Error *error_gpu) {
    auto check = [&](size_t start, size_t end, bool &is_failed, size_t &i, bool &has_nan,
                     bool &has_inf) {
        has_nan = false;
        has_inf = false;
        is_failed = false;
        i = 0;
        for (i = start; i < end; ++i) {
            if (unlikely(is_nan(teco_result[i]) && std::isnan(gpu_result[i]))) {
                teco_result[i] = (T)0.0;
                gpu_result[i] = (T)0.0;
                baseline_result[i] = (T)0.0;
                has_nan = true;
            } else if (unlikely(is_inf(teco_result[i]) && is_inf(gpu_result[i]) &&
                                teco_result[i] == gpu_result[i])) {
                // if a is inf, b is -inf, don't deal here.
                // when check hasNanOrInf will set diff as DBL_MAX (instead infinity).
                teco_result[i] = (T)0.0;
                gpu_result[i] = (T)0.0;
                baseline_result[i] = (T)0.0;
                has_inf = true;
            } else if (unlikely(is_nan(teco_result[i]))) {
                ALLOG(ERROR) << "Only found result computed on teco " + name + "[" << i
                             << "] is NaN, failed!!!";
                error_teco->has_nan = true;
                error_gpu->has_nan = false;
                is_failed = true;
                return;
            } else if (unlikely(std::isnan(gpu_result[i]))) {
                ALLOG(ERROR) << "Only found result computed on gpu " + name + "[" << i
                             << "] is NaN, failed!!!";
                error_teco->has_nan = false;
                error_gpu->has_nan = true;
                is_failed = true;
                return;
            } else if (unlikely(is_inf(teco_result[i]))) {
                ALLOG(ERROR) << "Only found result computed on teco " + name + "[" << i
                             << "] is Inf, failed!!!";
                error_teco->has_inf = true;
                error_gpu->has_inf = false;
                is_failed = true;
                return;
            } else if (unlikely(is_inf(gpu_result[i]))) {
                if (std::is_same<T, half_float::half>::value &&
                    (teco_result[i] == 65504 || teco_result[i] == -65504)) {
                    teco_result[i] = (T)0.0;
                    gpu_result[i] = (T)0.0;
                    baseline_result[i] = (T)0.0;
                    has_inf = true;
                } else {
                    ALLOG(ERROR) << "Only found result computed on gpu " + name + "[" << i
                                 << "] is Inf, failed!!!";
                    error_teco->has_inf = false;
                    error_gpu->has_inf = true;
                    is_failed = true;
                    return;
                }
            } else if (unlikely(is_inf(baseline_result[i])) ||
                       unlikely(std::isnan(baseline_result[i]))) {
                double abs_error = std::fabs(teco_result[i] - gpu_result[i]);
                double rela_error = abs_error / std::fabs(gpu_result[i]);
                double abs_eps = 1e-6;
                double rela_eps = 1e-6;
                if (std::is_same<T, half_float::half>::value) {
                    abs_eps = 1e-4;
                    rela_eps = 1e-3;
                }
                if (abs_error < abs_eps || rela_error < rela_eps) {
                    teco_result[i] = (T)0.0;
                    gpu_result[i] = (T)0.0;
                    baseline_result[i] = (T)0.0;
                    ALLOG(WARNING) << "Only Found result computed on cpu " + name + "[" << i
                                   << "] is nan/inf, set cpu/gpu/teco as 0, and go on.";
                } else {
                    baseline_result[i] = gpu_result[i];
                    ALLOG(WARNING) << "Only Found result computed on cpu " + name + "[" << i
                                   << "] is nan/inf, set it as gpu value, and go on.";
                }
            }
        }
        // set half value to zero when less than 2^(-14) @ teco is not subnormal number
        if (std::is_same<T, half_float::half>::value) {
            double min_half = pow(2, -14);
            for (int j = start; j < end; ++j) {
                uint16_t value = *((uint16_t *)(&(teco_result[j])));
                if (((value & 0x7C00) == 0) && ((value & 0x03FF) != 0)) {
                } else {
                    if (std::fabs(baseline_result[j]) < min_half) {
                        baseline_result[j] = 0.0;
                    }
                    if (std::fabs(gpu_result[j]) < min_half) {
                        gpu_result[j] = 0.0;
                    }
                }
            }
        }
    };

    bool has_nan = false;
    bool has_inf = false;
    bool is_failed = false;
    size_t index = 0;
#ifndef COMPARE_PARALELL
    check(0, count, is_failed, index, has_nan, has_inf);
#else
    bool has_nan_arr[THREAD_NUM] = {false};
    bool has_inf_arr[THREAD_NUM] = {false};
    bool is_failed_arr[THREAD_NUM] = {false};
    size_t index_arr[THREAD_NUM] = {0};
    std::thread *thread[THREAD_NUM];
    size_t per_thread_num = (count + THREAD_NUM - 1) / THREAD_NUM;
    for (int i = 0; i < THREAD_NUM; i++) {
        size_t start = i * per_thread_num;
        size_t end = start + per_thread_num > count ? count : start + per_thread_num;
        thread[i] =
            new std::thread(check, start, end, std::ref(is_failed_arr[i]), std::ref(index_arr[i]),
                            std::ref(has_nan_arr[i]), std::ref(has_inf_arr[i]));
    }
    for (int i = 0; i < THREAD_NUM; i++) {
        thread[i]->join();
    }
    for (int i = 0; i < THREAD_NUM; i++) {
        delete thread[i];
        thread[i] = nullptr;
    }

    for (int i = THREAD_NUM - 1; i >= 0; i--) {
        if (is_failed_arr[i]) {
            is_failed = true;
            index = index_arr[i];
        }
        has_nan |= has_nan_arr[i];
        has_inf |= has_inf_arr[i];
    }
#endif

    if (is_failed) {
        error_teco->max_error = teco_result[index];
        error_teco->compare_value = teco_result[index];
        error_teco->baseline_value = baseline_result[index];
        error_teco->index = index;
        error_gpu->max_error = gpu_result[index];
        error_gpu->compare_value = gpu_result[index];
        error_gpu->baseline_value = baseline_result[index];
        error_gpu->index = index;
        return false;
    }
    if (has_nan) {
        ALLOG(WARNING) << "Found result of teco and gpu [" + name +
                              "] are both NaN, set them as "
                              "0, and go on.";
    }
    if (has_inf) {
        ALLOG(WARNING) << "Found result of teco and gpu [" + name +
                              "] are both Inf, set them as "
                              "0, and go on.";
    }
    return true;
}

template <typename T>
bool resetNanOrInfAsZero(T *device, T *baseline, size_t count, Error *error_teco) {
    auto check = [&](size_t start, size_t end, bool &is_failed, bool &has_nan, bool &has_inf) {
        has_nan = false;
        has_inf = false;
        is_failed = false;
        for (size_t i = start; i < end; ++i) {
            if (unlikely(std::isnan(baseline[i]) && is_nan(device[i]))) {
                baseline[i] = (T)0.0;
                device[i] = (T)0.0;
                has_nan = true;
            } else if (unlikely(is_inf(baseline[i]) && is_inf(device[i]) &&
                                baseline[i] == device[i])) {
                // if a is inf, b is -inf, don't deal here.
                // when check hasNanOrInf will set diff as DBL_MAX (instead infinity).
                baseline[i] = (T)0.0;
                device[i] = (T)0.0;
                has_inf = true;
            } else if (unlikely(std::isnan(baseline[i]))) {
                ALLOG(ERROR) << "Only found result computed on baseline is NaN, failed!!!";
                is_failed = true;
                return;
            } else if (unlikely(is_nan(device[i]))) {
                ALLOG(ERROR) << "Only found result computed on TECO is NaN, failed!!!";
                is_failed = true;
                error_teco->has_nan = true;
                return;
            } else if (unlikely(is_inf(baseline[i]))) {
                ALLOG(ERROR) << "Only found result computed on baseline is Inf, failed!!!";
                is_failed = true;
                return;
            } else if (unlikely(is_inf(device[i]))) {
                ALLOG(ERROR) << "Only found result computed on TECO is Inf, failed!!!";
                error_teco->has_inf = true;
                is_failed = true;
                return;
            }
        }

        // set half value to zero when less than 2^(-14)
        if (std::is_same<T, half_float::half>::value) {
            double min_half = pow(2, -14);
            for (int j = start; j < end; ++j) {
                if (std::fabs(baseline[j]) < min_half) {
                    baseline[j] = 0.0;
                }
            }
        }
    };

    bool has_nan = false;
    bool has_inf = false;
    bool is_failed = false;
#ifndef COMPARE_PARALELL
    check(0, count, is_failed, has_nan, has_inf);
#else
    bool has_nan_arr[THREAD_NUM] = {false};
    bool has_inf_arr[THREAD_NUM] = {false};
    bool is_failed_arr[THREAD_NUM] = {false};
    std::thread *thread[THREAD_NUM];
    size_t per_thread_num = (count + THREAD_NUM - 1) / THREAD_NUM;
    for (int i = 0; i < THREAD_NUM; i++) {
        size_t start = i * per_thread_num;
        size_t end = start + per_thread_num > count ? count : start + per_thread_num;
        thread[i] = new std::thread(check, start, end, std::ref(is_failed_arr[i]),
                                    std::ref(has_nan_arr[i]), std::ref(has_inf_arr[i]));
    }
    for (int i = 0; i < THREAD_NUM; i++) {
        thread[i]->join();
    }
    for (int i = 0; i < THREAD_NUM; i++) {
        delete thread[i];
        thread[i] = nullptr;
    }

    for (int i = THREAD_NUM - 1; i >= 0; i--) {
        is_failed |= is_failed_arr[i];
        has_nan |= has_nan_arr[i];
        has_inf |= has_inf_arr[i];
    }
#endif
    if (is_failed) {
        return false;
    }
    if (has_nan) {
        ALLOG(WARNING) << "Found result of baseline and device are both NaN, set them as "
                          "0, and go on.";
    }
    if (has_inf) {
        ALLOG(WARNING) << "Found result of baseline and device are both Inf, set them as "
                          "0, and go on.";
    }
    return true;
}

class Evaluator {
 public:
    Evaluator() {}
    virtual ~Evaluator() {}
    void copy(const Evaluator *e);

    template <typename T>
    Error computeError_template(T *baseline, T *device_result, size_t count,
                                const Criterion &criterion, const std::string &name) {
        Error error;
        auto func = criterion.formula;
        if (!Context::instance()->compareWithGPU()) {
            // if both device and baseline is nan/inf, set them zero
            if (!resetNanOrInfAsZero(device_result, baseline, count, &error)) {
                error.has_nan = true;
                return error;
            }
        }

        switch (func) {
            case DIFF1: {
                error = computeDiff1(baseline, device_result, count, criterion);
                break;
            }
            case DIFF2: {
                error = computeDiff2(baseline, device_result, count, criterion);
                break;
            }
            case DIFF3: {
                error = computeDiff3(baseline, device_result, count, criterion, name, true);
                break;
            }
            case DIFF3_MAX: {
                error = computeDiff3(baseline, device_result, count, criterion, name, false);
                break;
            }
            case DIFF3_MEAN: {
                error = computeDiff3_mean(baseline, device_result, count, criterion);
                break;
            }
            case DIFF4: {
                error = computeDiff4(baseline, device_result, count, criterion);
                break;
            }
            case MAPE: {
                error = computeMape(baseline, device_result, count, criterion);
                break;
            }
            case MAE: {
                error = computeMae(baseline, device_result, count, criterion);
                break;
            }
            case MSE_RELA: {
                error = computeMse_rela(baseline, device_result, count, criterion);
                break;
            }
            default:
                ALLOG(ERROR) << "Evaluator: found unsupported criterion when compute "
                                "result error.";
        }
        return error;
    }

    Error computeError(void *a, void *b, size_t count, const Criterion &criterion,
                       const std::string &name, const testpt::DataType dtype);

    bool resetNanOrInfToZero(void *teco_result, void *gpu_result, void *baseline_result,
                             size_t count, const std::string &name, const testpt::DataType dtype,
                             Error *error_teco, Error *error_gpu);
    bool resetNanOrInfToZero(void *teco_result, void *baseline_result, size_t count,
                             const testpt::DataType dtype, Error *error_teco);
    double computeEfficiency(double num, double latency, double den);

    tecotestStatus_t isPassed();
    std::vector<std::vector<std::string>> what();
    tecotestStatus_t isPassed_cpu();
    std::vector<std::vector<std::string>> what_cpu();
    tecotestStatus_t isPassed_bf16();
    std::vector<std::vector<std::string>> what_bf16();
    // get name + criterion + error
    const std::vector<ErrorWrap> &errors() { return error_vec_; }

 private:
    template <typename T>
    Error computeDiff1(T *baseline, T *device_result, size_t count, Criterion criterion) {
        double numerator_sum = 0.0;
        double denominator_sum = 0.0;
#ifndef COMPARE_PARALELL
        for (size_t i = 0; i < count; i++) {
            numerator_sum += std::fabs((double)(baseline[i]) - (double)(device_result[i]));
            denominator_sum += std::fabs((double)(baseline[i]));
        }
#else
        auto diff1 = [&](size_t start, size_t end, double &numerator, double &denominator) {
            numerator = 0.0;
            denominator = 0.0;
            for (size_t i = start; i < end; ++i) {
                numerator += std::fabs((double)(baseline[i]) - (double)(device_result[i]));
                denominator += std::fabs((double)(baseline[i]));
            }
        };

        double numerator_arr[THREAD_NUM] = {0.0};
        double denominator_arr[THREAD_NUM] = {0.0};
        std::thread *thread[THREAD_NUM];
        size_t per_thread_num = (count + THREAD_NUM - 1) / THREAD_NUM;
        for (int i = 0; i < THREAD_NUM; i++) {
            size_t start = i * per_thread_num;
            size_t end = start + per_thread_num > count ? count : start + per_thread_num;
            thread[i] = new std::thread(diff1, start, end, std::ref(numerator_arr[i]),
                                        std::ref(denominator_arr[i]));
        }
        for (int i = 0; i < THREAD_NUM; i++) {
            thread[i]->join();
        }
        for (int i = 0; i < THREAD_NUM; i++) {
            delete thread[i];
            thread[i] = nullptr;
        }

        for (int i = 0; i < THREAD_NUM; i++) {
            numerator_sum += numerator_arr[i];
            denominator_sum += denominator_arr[i];
        }
#endif

        return Error(numerator_sum / (denominator_sum + EPSILON));
    }

    template <typename T>
    Error computeDiff2(T *baseline, T *device_result, size_t count, Criterion criterion) {
        double numerator_sum = 0.0;
        double denominator_sum = 0.0;
#ifndef COMPARE_PARALELL
        for (size_t i = 0; i < count; i++) {
            double delta = std::fabs((double)(baseline[i]) - (double)(device_result[i]));
            numerator_sum += pow(delta, 2);
            denominator_sum += pow(std::fabs((double)(baseline[i])), 2);
        }
#else
        auto diff2 = [&](size_t start, size_t end, double &numerator, double &denominator) {
            numerator = 0.0;
            denominator = 0.0;
            for (size_t i = start; i < end; ++i) {
                double delta = std::fabs((double)(baseline[i]) - (double)(device_result[i]));
                numerator += pow(delta, 2);
                denominator += pow(std::fabs((double)(baseline[i])), 2);
            }
        };

        double numerator_arr[THREAD_NUM] = {0.0};
        double denominator_arr[THREAD_NUM] = {0.0};
        std::thread *thread[THREAD_NUM];
        size_t per_thread_num = (count + THREAD_NUM - 1) / THREAD_NUM;
        for (int i = 0; i < THREAD_NUM; i++) {
            size_t start = i * per_thread_num;
            size_t end = start + per_thread_num > count ? count : start + per_thread_num;
            thread[i] = new std::thread(diff2, start, end, std::ref(numerator_arr[i]),
                                        std::ref(denominator_arr[i]));
        }
        for (int i = 0; i < THREAD_NUM; i++) {
            thread[i]->join();
        }
        for (int i = 0; i < THREAD_NUM; i++) {
            delete thread[i];
            thread[i] = nullptr;
        }

        for (int i = 0; i < THREAD_NUM; i++) {
            numerator_sum += numerator_arr[i];
            denominator_sum += denominator_arr[i];
        }
#endif

        return Error(sqrt(numerator_sum / (denominator_sum + EPSILON)));
    }

    template <typename T>
    Error computeDiff3(T *baseline, T *device_result, size_t count, Criterion criterion,
                       std::string name, bool all_error_flag) {
        Error error;
        double max_ratio = -1.0;
        size_t max_index = 0;
        double eps = EPSILON;
        if (std::is_same<T, float>::value) {
            eps = EPSILON_FLOAT;
        } else if (std::is_same<T, half_float::half>::value) {
            eps = EPSILON_HALF;
        }
        if (criterion.calc_eps > 0) {
            eps = criterion.calc_eps;
        }

#ifndef COMPARE_PARALELL
        for (size_t i = 0; i < count; ++i) {
            double numerator = std::fabs((double)(device_result[i]) - (double)(baseline[i]));
            double ratio = 0;
            if (std::fabs(baseline[i]) < eps) {
                ratio = numerator;
            } else {
                ratio = numerator / (std::fabs((double)(baseline[i])) + EPSILON);
            }
            if (ratio > max_ratio) {
                max_ratio = ratio;
                max_index = i;
            }
            if (all_error_flag || Context::instance()->printFaultFlag()) {
                error.error_vect.push_back(ratio);
            }

            if (all_error_flag) {
                for (int j = 0; j < error.error_thresholds.size(); j++) {
                    if (ratio > error.error_thresholds[j]) error.error_ratios[j]++;
                }
                if (ratio > criterion.error_threshold) {
                    error.error_num++;
                }
            }
        }
#else
        auto diff3 = [&](size_t start, size_t end, double &m_ratio, size_t &m_index,
                         size_t &error_num, std::vector<double> &ratio_vect,
                         std::vector<double> &error_ratios) {
            m_ratio = -1.0;
            double ratio = 0.0;
            for (size_t i = start; i < end; ++i) {
                double numerator = std::fabs((double)(device_result[i]) - (double)(baseline[i]));
                if (std::fabs(baseline[i]) < eps) {
                    ratio = numerator;
                } else {
                    ratio = numerator / (std::fabs((double)(baseline[i])) + EPSILON);
                }
                if (ratio > m_ratio) {
                    m_ratio = ratio;
                    m_index = i;
                }
                if (all_error_flag || Context::instance()->printFaultFlag()) {
                    ratio_vect.push_back(ratio);
                }

                if (all_error_flag) {
                    for (int j = 0; j < error.error_thresholds.size(); j++) {
                        if (ratio > error.error_thresholds[j]) error_ratios[j]++;
                    }
                    if (ratio > criterion.error_threshold) {
                        error_num++;
                    }
                }
            }
        };

        double max_ratio_arr[THREAD_NUM] = {0.0};
        size_t max_index_arr[THREAD_NUM] = {0};
        size_t error_num_arr[THREAD_NUM] = {0};
        std::vector<std::vector<double>> ratio_vect_arr(THREAD_NUM, std::vector<double>());
        std::vector<std::vector<double>> error_ratios_arr(THREAD_NUM,
                                                          std::vector<double>(error.error_ratios));
        std::thread *thread[THREAD_NUM];
        size_t per_thread_num = (count + THREAD_NUM - 1) / THREAD_NUM;
        for (int i = 0; i < THREAD_NUM; i++) {
            size_t start = i * per_thread_num;
            size_t end = start + per_thread_num > count ? count : start + per_thread_num;
            thread[i] = new std::thread(diff3, start, end, std::ref(max_ratio_arr[i]),
                                        std::ref(max_index_arr[i]), std::ref(error_num_arr[i]),
                                        std::ref(ratio_vect_arr[i]), std::ref(error_ratios_arr[i]));
        }
        for (int i = 0; i < THREAD_NUM; i++) {
            thread[i]->join();
        }
        for (int i = 0; i < THREAD_NUM; i++) {
            delete thread[i];
            thread[i] = nullptr;
        }

        for (int i = 0; i < THREAD_NUM; i++) {
            if (max_ratio_arr[i] > max_ratio) {
                max_ratio = max_ratio_arr[i];
                max_index = max_index_arr[i];
            }
            if (all_error_flag || Context::instance()->printFaultFlag()) {
                error.error_vect.insert(error.error_vect.end(), ratio_vect_arr[i].begin(),
                                        ratio_vect_arr[i].end());
            }
            if (all_error_flag) {
                for (int j = 0; j < error.error_ratios.size(); j++) {
                    error.error_ratios[j] += error_ratios_arr[i][j];
                }
                error.error_num += error_num_arr[i];
            }
        }
#endif
        error.max_error = max_ratio;
        error.index = max_index;
        error.baseline_value = baseline[max_index];
        error.compare_value = device_result[max_index];

        //
        if (Context::instance()->printFaultFlag()) {
            std::cout << "[" + name + "]:\n";
            size_t count_tmp = 0;
            for (size_t i = 0; i < error.error_vect.size(); ++i) {
                double ratio = error.error_vect[i];
                if (ratio > criterion.error_threshold) {
                    count_tmp++;
                    if (count_tmp <= Context::instance()->printFaultNum()) {
                        std::cout << std::setw(10) << i;
                        if (std::is_same<T, uint8_t>::value) {
                            std::cout << ",  compare:" << std::setw(12) << std::setprecision(5)
                                      << (int32_t)(device_result[i]);
                            std::cout << ",  expected:" << std::setw(12) << std::setprecision(5)
                                      << (int32_t)(baseline[i]);
                        } else {
                            std::cout << ",  compare:" << std::setw(12) << std::setprecision(5)
                                      << device_result[i];
                            std::cout << ",  expected:" << std::setw(12) << std::setprecision(5)
                                      << baseline[i];
                        }
                        std::cout << ",  error:" << std::scientific << std::setprecision(2) << ratio
                                  << std::endl;
                    } else {
                        break;
                    }
                }
            }
        }

        if (all_error_flag) {
            for (int j = 0; j < error.error_thresholds.size(); j++) {
                error.error_ratios[j] /= count;
            }
            error.error_ratio = (double)error.error_num / count;
        }

        return error;
    }

    template <typename T>
    Error computeDiff4(T *baseline, T *device_result, size_t count, Criterion criterion) {
        // todo(maliang)
        int max_count = 0;
        int num_count = 0;
        for (size_t i = 0; i < count; ++i) {
            max_count += device_result[i] < baseline[i];
            num_count += device_result[i] != baseline[i];
        }

        return Error((num_count < 100) ? 0 : max_count / (num_count + EPSILON));
    }

    template <typename T>
    Error computeMape(T *baseline, T *device_result, size_t count, Criterion criterion) {
        double sum_ratio = 0.0;
#ifndef COMPARE_PARALELL
        for (size_t i = 0; i < count; ++i) {
            double numerator = std::fabs((double)(device_result[i]) - (double)(baseline[i]));
            sum_ratio += numerator / (std::fabs((double)(baseline[i])) + EPSILON);
        }
#else
        auto mape = [&](size_t start, size_t end, double &ratio) {
            ratio = 0.0;
            for (size_t i = start; i < end; ++i) {
                double numerator = std::fabs((double)(device_result[i]) - (double)(baseline[i]));
                ratio += numerator / (std::fabs((double)(baseline[i])) + EPSILON);
            }
        };

        double ratio_arr[THREAD_NUM] = {0.0};
        std::thread *thread[THREAD_NUM];
        size_t per_thread_num = (count + THREAD_NUM - 1) / THREAD_NUM;
        for (int i = 0; i < THREAD_NUM; i++) {
            size_t start = i * per_thread_num;
            size_t end = start + per_thread_num > count ? count : start + per_thread_num;
            thread[i] = new std::thread(mape, start, end, std::ref(ratio_arr[i]));
        }
        for (int i = 0; i < THREAD_NUM; i++) {
            thread[i]->join();
        }
        for (int i = 0; i < THREAD_NUM; i++) {
            delete thread[i];
            thread[i] = nullptr;
        }

        for (int i = 0; i < THREAD_NUM; i++) {
            sum_ratio += ratio_arr[i];
        }
#endif

        return Error(sum_ratio / count);
    }

    template <typename T>
    Error computeMae(T *baseline, T *device_result, size_t count, Criterion criterion) {
        double sum_ratio = 0.0;
#ifndef COMPARE_PARALELL
        for (size_t i = 0; i < count; ++i) {
            double numerator = std::fabs((double)(device_result[i]) - (double)(baseline[i]));
            sum_ratio += numerator;
        }
#else
        auto mae = [&](size_t start, size_t end, double &ratio) {
            ratio = 0.0;
            for (size_t i = start; i < end; ++i) {
                double numerator = std::fabs((double)(device_result[i]) - (double)(baseline[i]));
                ratio += numerator;
            }
        };

        double ratio_arr[THREAD_NUM] = {0.0};
        std::thread *thread[THREAD_NUM];
        size_t per_thread_num = (count + THREAD_NUM - 1) / THREAD_NUM;
        for (int i = 0; i < THREAD_NUM; i++) {
            size_t start = i * per_thread_num;
            size_t end = start + per_thread_num > count ? count : start + per_thread_num;
            thread[i] = new std::thread(mae, start, end, std::ref(ratio_arr[i]));
        }
        for (int i = 0; i < THREAD_NUM; i++) {
            thread[i]->join();
        }
        for (int i = 0; i < THREAD_NUM; i++) {
            delete thread[i];
            thread[i] = nullptr;
        }

        for (int i = 0; i < THREAD_NUM; i++) {
            sum_ratio += ratio_arr[i];
        }
#endif

        return Error(sum_ratio / count);
    }

    template <typename T>
    Error computeMse_rela(T *baseline, T *device_result, size_t count, Criterion criterion) {
        double sum_ratio = 0.0;

#ifndef COMPARE_PARALELL
        for (size_t i = 0; i < count; ++i) {
            double numerator = std::fabs((double)(device_result[i]) - (double)(baseline[i]));
            numerator /=
                std::fabs(std::max((double)(device_result[i]), (double)(baseline[i]))) + 1e-5;
            sum_ratio += numerator;
        }
#else
        auto mae = [&](size_t start, size_t end, double &ratio) {
            ratio = 0.0;
            for (size_t i = start; i < end; ++i) {
                double numerator = std::fabs((double)(device_result[i]) - (double)(baseline[i]));
                numerator /=
                    std::fabs(std::max((double)(device_result[i]), (double)(baseline[i]))) + 1e-5;
                ratio += numerator;
            }
        };

        double ratio_arr[THREAD_NUM] = {0.0};
        std::thread *thread[THREAD_NUM];
        size_t per_thread_num = (count + THREAD_NUM - 1) / THREAD_NUM;
        for (int i = 0; i < THREAD_NUM; i++) {
            size_t start = i * per_thread_num;
            size_t end = start + per_thread_num > count ? count : start + per_thread_num;
            thread[i] = new std::thread(mae, start, end, std::ref(ratio_arr[i]));
        }
        for (int i = 0; i < THREAD_NUM; i++) {
            thread[i]->join();
        }
        for (int i = 0; i < THREAD_NUM; i++) {
            delete thread[i];
            thread[i] = nullptr;
        }

        for (int i = 0; i < THREAD_NUM; i++) {
            sum_ratio += ratio_arr[i];
        }
#endif

        return Error(sum_ratio / count);
    }

    template <typename T>
    Error computeDiff3_mean(T *baseline, T *device_result, size_t count, Criterion criterion) {
        double eps = EPSILON;
        if (std::is_same<T, float>::value) {
            eps = EPSILON_FLOAT;
        } else if (std::is_same<T, half_float::half>::value) {
            eps = EPSILON_HALF;
        }
        if (criterion.calc_eps > 0) {
            eps = criterion.calc_eps;
        }

        double sum_ratio = 0.0;
#ifndef COMPARE_PARALELL
        for (size_t i = 0; i < count; ++i) {
            double numerator = std::fabs((double)(device_result[i]) - (double)(baseline[i]));
            if (std::fabs(baseline[i]) < eps) {
                sum_ratio += numerator;
            } else {
                sum_ratio += numerator / (std::fabs((double)(baseline[i])) + EPSILON);
            }
        }
#else
        auto diff3_mean = [&](size_t start, size_t end, double &ratio) {
            for (size_t i = start; i < end; ++i) {
                double numerator = std::fabs((double)(device_result[i]) - (double)(baseline[i]));
                if (std::fabs(baseline[i]) < eps) {
                    ratio += numerator;
                } else {
                    ratio += numerator / (std::fabs((double)(baseline[i])) + EPSILON);
                }
            }
        };

        double ratio_arr[THREAD_NUM] = {0.0};
        std::thread *thread[THREAD_NUM];
        size_t per_thread_num = (count + THREAD_NUM - 1) / THREAD_NUM;
        for (int i = 0; i < THREAD_NUM; i++) {
            size_t start = i * per_thread_num;
            size_t end = start + per_thread_num > count ? count : start + per_thread_num;
            thread[i] = new std::thread(diff3_mean, start, end, std::ref(ratio_arr[i]));
        }
        for (int i = 0; i < THREAD_NUM; i++) {
            thread[i]->join();
        }
        for (int i = 0; i < THREAD_NUM; i++) {
            delete thread[i];
            thread[i] = nullptr;
        }

        for (int i = 0; i < THREAD_NUM; i++) {
            sum_ratio += ratio_arr[i];
        }
#endif
        return Error(sum_ratio / count);
        ;
    }

 public:
    std::vector<ErrorWrap> error_vec_;  // vetor output's error

    double workspace_size_ = -1;  // for -1
};

struct PerfInfo {
    // api time
    double interface_time = -1000;  // us

    // hardware time of device
    double hardware_time = -1000;        // us
    std::vector<double> hardware_times;  // us

    // launch_time
    double launch_time = -1;  // us
    int launch_count = -1;    // one api launch counts

    std::vector<std::pair<std::string, double>> kernel_details;  // us
    std::vector<std::string> cache_miss_details;

    // memcpy host to device time (device only
    double h2d_time = -1;
    double d2h_time = -1;

    // theory ops/io/peak force/bandwidth for efficiency
    size_t theory_ops = 0;      // op
    size_t theory_io = 0;       // bytes
    double compute_force = -1;  // op/s
    double io_bandwidth = -1;   // GB/s
};

struct ComparedPerfInfo {
    std::string device;
    double interface_time = -1000;                            // us
    std::vector<std::pair<std::string, double>> api_details;  // us
};

struct EvaluateResult {
    // id
    std::string op_name = "";
    std::string case_path = "";
    std::string case_name = "";
    // perf info
    PerfInfo device;
    // compared device info
    ComparedPerfInfo compared_device;
    // tensor info
    std::vector<MetaTensor> tensors;

    // errors
    std::vector<ErrorWrap> errors;
    // result
    bool is_passed = false;
    std::vector<std::vector<std::string>> what;
    std::unordered_map<std::string, unsigned int> result_hash;

    tecotestStatus_t status = TECOTEST_STATUS_SUCCESS;
};

}  // namespace optest

#endif  // CASE_EVALUATOR_H_  // NOLINT
