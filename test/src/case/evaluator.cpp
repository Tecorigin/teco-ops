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

#include "case/evaluator.h"
#include <limits>
#include <string>
#include <vector>
namespace optest {

// found inf or nan, return true.
#ifdef __AVX__
bool hasNanOrInf(float *data, size_t count) {
    const __m256 exp_bit = _mm256_set1_ps(std::numeric_limits<float>::infinity());

    size_t stride = 256 / (sizeof(float) * 8);  // 1 __m256 saved 8 *
                                                // (sizeof(float) * 8 bit)
    size_t repeat = count / stride * stride;

    __m256 m_data;
    for (size_t i = 0; i < repeat; i += stride) {
        m_data = _mm256_load_ps(data + i);
        m_data = _mm256_and_ps(exp_bit, m_data);
        m_data = _mm256_cmp_ps(m_data, exp_bit, _CMP_EQ_OQ);
        if (_mm256_movemask_ps(m_data) != 0) {
            return true;
        }
    }

    for (size_t i = repeat; i < count - repeat; ++i) {
        if (std::isnan(data[i]) || std::isinf(data[i])) {
            return true;
        }
    }
    return false;
}
#else
bool hasNanOrInf(float *data, size_t count) {
    for (int i = 0; i < count; ++i) {
        if (std::isinf(data[i]) || std::isnan(data[i])) {
            return true;
        }
    }
    return false;
}
#endif

void skipNanOrInfAsZero(float *a, float *b, size_t count) {
    bool has_nan = false;
    bool has_inf = false;
    for (size_t i = 0; i < count; ++i) {
        int tmp = *(int *)&a[i];
        if (unlikely(std::isnan(a[i]))) {
            a[i] = 0.0f;
            b[i] = 0.0f;
            has_nan = true;
        } else if (unlikely(std::isinf(a[i]))) {
            a[i] = 0.0f;
            b[i] = 0.0f;
            has_inf = true;
        }
    }
    if (has_nan) {
        ALLOG(WARNING) << "Found result of baseline is NaN,"
                       << " set baseline and device as 0, and go on.";
    }
    if (has_inf) {
        ALLOG(WARNING) << "Found result of baseline is Inf,"
                       << " set baseline and device as 0, and go on.";
    }
}

bool Evaluator::resetNanOrInfToZero(void *teco_result, void *gpu_result, void *baseline_result,
                                    size_t count, const std::string &name,
                                    const testpt::DataType dtype, Error *error_teco,
                                    Error *error_gpu) {
    switch (dtype) {
        case testpt::DTYPE_HALF:
            return resetNanOrInfAsZero(
                (half_float::half *)teco_result, (half_float::half *)gpu_result,
                (half_float::half *)baseline_result, count, name, error_teco, error_gpu);
        case testpt::DTYPE_FLOAT:
            return resetNanOrInfAsZero((float *)teco_result, (float *)gpu_result,
                                       (float *)baseline_result, count, name, error_teco,
                                       error_gpu);
        case testpt::DTYPE_INT8:
            return resetNanOrInfAsZero((int8_t *)teco_result, (int8_t *)gpu_result,
                                       (int8_t *)baseline_result, count, name, error_teco,
                                       error_gpu);
        case testpt::DTYPE_INT16:
            return resetNanOrInfAsZero((int16_t *)teco_result, (int16_t *)gpu_result,
                                       (int16_t *)baseline_result, count, name, error_teco,
                                       error_gpu);
        case testpt::DTYPE_INT32:
            return resetNanOrInfAsZero((int32_t *)teco_result, (int32_t *)gpu_result,
                                       (int32_t *)baseline_result, count, name, error_teco,
                                       error_gpu);
        case testpt::DTYPE_INT64:
            return resetNanOrInfAsZero((int64_t *)teco_result, (int64_t *)gpu_result,
                                       (int64_t *)baseline_result, count, name, error_teco,
                                       error_gpu);
        case testpt::DTYPE_UINT8:
            return resetNanOrInfAsZero((uint8_t *)teco_result, (uint8_t *)gpu_result,
                                       (uint8_t *)baseline_result, count, name, error_teco,
                                       error_gpu);
        case testpt::DTYPE_UINT16:
            return resetNanOrInfAsZero((uint16_t *)teco_result, (uint16_t *)gpu_result,
                                       (uint16_t *)baseline_result, count, name, error_teco,
                                       error_gpu);
        case testpt::DTYPE_UINT32:
            return resetNanOrInfAsZero((uint32_t *)teco_result, (uint32_t *)gpu_result,
                                       (uint32_t *)baseline_result, count, name, error_teco,
                                       error_gpu);
        case testpt::DTYPE_UINT64:
            return resetNanOrInfAsZero((uint64_t *)teco_result, (uint64_t *)gpu_result,
                                       (uint64_t *)baseline_result, count, name, error_teco,
                                       error_gpu);
        case testpt::DTYPE_DOUBLE:
            return resetNanOrInfAsZero((double *)teco_result, (double *)gpu_result,
                                       (double *)baseline_result, count, name, error_teco,
                                       error_gpu);
        case testpt::DTYPE_BOOL:
            return resetNanOrInfAsZero((bool *)teco_result, (bool *)gpu_result,
                                       (bool *)baseline_result, count, name, error_teco, error_gpu);
        default:
            ALLOG(ERROR) << "Don't support this dtype.";
            throw UnSupportType(std::string(__FILE__) + " +" + std::to_string(__LINE__));
    }
}

bool Evaluator::resetNanOrInfToZero(void *teco_result, void *baseline_result, size_t count,
                                    const testpt::DataType dtype, Error *error_teco) {
    switch (dtype) {
        case testpt::DTYPE_HALF:
            return resetNanOrInfAsZero((half_float::half *)teco_result,
                                       (half_float::half *)baseline_result, count, error_teco);
        case testpt::DTYPE_FLOAT:
            return resetNanOrInfAsZero((float *)teco_result, (float *)baseline_result, count,
                                       error_teco);
        case testpt::DTYPE_INT8:
            return resetNanOrInfAsZero((int8_t *)teco_result, (int8_t *)baseline_result, count,
                                       error_teco);
        case testpt::DTYPE_INT16:
            return resetNanOrInfAsZero((int16_t *)teco_result, (int16_t *)baseline_result, count,
                                       error_teco);
        case testpt::DTYPE_INT32:
            return resetNanOrInfAsZero((int32_t *)teco_result, (int32_t *)baseline_result, count,
                                       error_teco);
        case testpt::DTYPE_INT64:
            return resetNanOrInfAsZero((int64_t *)teco_result, (int64_t *)baseline_result, count,
                                       error_teco);
        case testpt::DTYPE_UINT8:
            return resetNanOrInfAsZero((uint8_t *)teco_result, (uint8_t *)baseline_result, count,
                                       error_teco);
        case testpt::DTYPE_UINT16:
            return resetNanOrInfAsZero((uint16_t *)teco_result, (uint16_t *)baseline_result, count,
                                       error_teco);
        case testpt::DTYPE_UINT32:
            return resetNanOrInfAsZero((uint32_t *)teco_result, (uint32_t *)baseline_result, count,
                                       error_teco);
        case testpt::DTYPE_UINT64:
            return resetNanOrInfAsZero((uint64_t *)teco_result, (uint64_t *)baseline_result, count,
                                       error_teco);
        case testpt::DTYPE_DOUBLE:
            return resetNanOrInfAsZero((double *)teco_result, (double *)baseline_result, count,
                                       error_teco);
        case testpt::DTYPE_BOOL:
            return resetNanOrInfAsZero((bool *)teco_result, (bool *)baseline_result, count,
                                       error_teco);
        case testpt::DTYPE_BFLOAT16:
            return resetNanOrInfAsZero((uint16_t *)teco_result, (uint16_t *)baseline_result, count,
                                       error_teco);
        default:
            ALLOG(ERROR) << "Don't support this dtype.";
            throw std::invalid_argument(std::string(__FILE__) + " +" + std::to_string(__LINE__));
    }
}

Error Evaluator::computeError(void *baseline, void *device_result, size_t count,
                              const Criterion &criterion, const std::string &name,
                              const testpt::DataType dtype) {
    switch (dtype) {
        case testpt::DTYPE_HALF:
            return computeError_template((half_float::half *)baseline,
                                         (half_float::half *)device_result, count, criterion, name);
        case testpt::DTYPE_FLOAT:
            return computeError_template((float *)baseline, (float *)device_result, count,
                                         criterion, name);
        case testpt::DTYPE_BFLOAT16:
            return computeError_template((uint16_t *)baseline, (uint16_t *)device_result, count,
                                         criterion, name);
        case testpt::DTYPE_INT8:
            return computeError_template((int8_t *)baseline, (int8_t *)device_result, count,
                                         criterion, name);
        case testpt::DTYPE_INT16:
            return computeError_template((int16_t *)baseline, (int16_t *)device_result, count,
                                         criterion, name);
        case testpt::DTYPE_INT32:
            return computeError_template((int32_t *)baseline, (int32_t *)device_result, count,
                                         criterion, name);
        case testpt::DTYPE_INT64:
            return computeError_template((int64_t *)baseline, (int64_t *)device_result, count,
                                         criterion, name);
        case testpt::DTYPE_UINT8:
            return computeError_template((uint8_t *)baseline, (uint8_t *)device_result, count,
                                         criterion, name);
        case testpt::DTYPE_UINT16:
            return computeError_template((uint16_t *)baseline, (uint16_t *)device_result, count,
                                         criterion, name);
        case testpt::DTYPE_UINT32:
            return computeError_template((uint32_t *)baseline, (uint32_t *)device_result, count,
                                         criterion, name);
        case testpt::DTYPE_UINT64:
            return computeError_template((uint64_t *)baseline, (uint64_t *)device_result, count,
                                         criterion, name);
        case testpt::DTYPE_DOUBLE:
            return computeError_template((double *)baseline, (double *)device_result, count,
                                         criterion, name);
        case testpt::DTYPE_BOOL:
            return computeError_template((bool *)baseline, (bool *)device_result, count, criterion,
                                         name);
        default:
            ALLOG(ERROR) << "Don't support this dtype.";
            throw UnSupportType(std::string(__FILE__) + " +" + std::to_string(__LINE__));
    }
}

tecotestStatus_t Evaluator::isPassed() {
    if (error_vec_.empty()) {
        ALLOG(WARNING) << "The result error is empty. Output shape is 0 in prototxt,"
                          " or only test performance now.";
    }

    for (int i = 0; i < error_vec_.size(); ++i) {
        if (error_vec_[i].criterion.enable == false) {
            continue;
        }

        auto func = error_vec_[i].criterion.formula;
        Criterion criterion = error_vec_[i].criterion;
        Error error_teco = error_vec_[i].error_teco;
        Error error_gpu = error_vec_[i].error_gpu;

        if (error_teco.has_nan || error_gpu.has_nan) {
            return TECOTEST_STATUS_RESULT_NAN_ERROR;
        } else if (error_teco.has_inf || error_gpu.has_inf) {
            return TECOTEST_STATUS_RESULT_INF_ERROR;
        } else {
            if (Formula::DIFF3 == func) {
                for (int i = 0; i < error_teco.error_vect.size(); i++) {
                    if (error_teco.error_vect[i] >
                            error_gpu.error_vect[i] * criterion.golden_threshold &&
                        error_teco.error_vect[i] > criterion.golden_eps) {
                        return TECOTEST_STATUS_RESULT_ERROR;
                    }
                }
            } else if (Formula::DIFF4 == func) {
                // todo(maliang)
                if (error_teco.max_error > criterion.error_threshold ||
                    ((error_teco.max_error < 1 - criterion.error_threshold) &&
                     (error_teco.max_error != 0)) ||
                    error_teco.max_error < 0) {
                    return TECOTEST_STATUS_RESULT_ERROR;
                }
            } else if (error_teco.max_error > error_gpu.max_error * criterion.golden_threshold &&
                       error_teco.max_error > criterion.golden_eps) {
                return TECOTEST_STATUS_RESULT_ERROR;
            }
        }
    }

    return TECOTEST_STATUS_SUCCESS;
}

// only when failed, call this func. to get error reason
std::vector<std::vector<std::string>> Evaluator::what() {
    std::vector<std::vector<std::string>> result;

    for (int i = 0; i < error_vec_.size(); ++i) {
        std::vector<std::string> res;
        if (error_vec_[i].criterion.enable == false) {
            result.emplace_back(res);
            continue;
        }

        auto name = error_vec_[i].name;
        auto func = error_vec_[i].criterion.formula;
        Criterion criterion = error_vec_[i].criterion;
        Error error_teco = error_vec_[i].error_teco;
        Error error_gpu = error_vec_[i].error_gpu;

        std::string error_str = "";
        if (error_teco.has_nan || error_gpu.has_nan) {
            if (error_teco.has_nan) {
                error_str += "The teco error of [" + name + "] is NOT digit.";
            } else {
                error_str += "The gpu error of [" + name + "] is NOT digit.";
            }
            res.emplace_back(error_str);
        } else {
            if (Formula::DIFF3 == func) {
                size_t fault_num = 0;
                std::string error_str_tmp = "";
                for (int i = 0; i < error_teco.error_vect.size(); i++) {
                    if (error_teco.error_vect[i] >
                            error_gpu.error_vect[i] * criterion.golden_threshold &&
                        error_teco.error_vect[i] > criterion.golden_eps) {
                        fault_num++;
                        std::ostringstream oss_error_teco, oss_error_gpu;
                        oss_error_teco.setf(std::ios::scientific);
                        oss_error_gpu.setf(std::ios::scientific);
                        oss_error_teco << error_teco.error_vect[i];
                        oss_error_gpu << error_gpu.error_vect[i];
                        if (Context::instance()->printFaultFlag() &&
                            fault_num <= Context::instance()->printFaultNum()) {
                            error_str_tmp += "\n" + std::to_string(i) +
                                             ", teco:" + oss_error_teco.str() +
                                             ", gpu:" + oss_error_gpu.str();
                        }
                    }
                }
                if (fault_num > 0) {
                    error_str += "The ratio of [" + name + "] teco_error/gpu_error > " +
                                 std::to_string(criterion.golden_threshold) + " on " +
                                 showFormula(func) + " is " + std::to_string(fault_num) + "/" +
                                 std::to_string(error_teco.error_vect.size());
                    error_str += error_str_tmp;

                    res.emplace_back(error_str);
                }
            } else if (error_teco.max_error > error_gpu.max_error * criterion.golden_threshold &&
                       error_teco.max_error > criterion.golden_eps) {
                std::ostringstream oss_error_teco, oss_error_gpu, oss_error_threshold;
                oss_error_teco.setf(std::ios::scientific);
                oss_error_gpu.setf(std::ios::scientific);
                oss_error_teco << error_teco.max_error;
                oss_error_gpu << error_gpu.max_error;
                oss_error_threshold << criterion.golden_threshold;
                error_str += "[" + name + "] teco_error/gpu_error = " + oss_error_teco.str() + "/" +
                             oss_error_gpu.str() + " = " +
                             std::to_string(error_teco.max_error / error_gpu.max_error) + " > " +
                             oss_error_threshold.str() + " on " + showFormula(func);
                res.emplace_back(error_str);
            }
        }
        result.emplace_back(res);
    }
    return result;
}

tecotestStatus_t Evaluator::isPassed_cpu() {
    if (error_vec_.empty()) {
        ALLOG(WARNING) << "The result error is empty. Output shape is 0 in prototxt,"
                          " or only test performance now.";
    }

    for (int i = 0; i < error_vec_.size(); ++i) {
        if (error_vec_[i].criterion.enable == false) {
            continue;
        }

        auto func = error_vec_[i].criterion.formula;
        Criterion criterion = error_vec_[i].criterion;
        Error error = error_vec_[i].error_teco;

        if (error.has_nan) {
            return TECOTEST_STATUS_RESULT_NAN_ERROR;
        } else if (Formula::DIFF3 == func) {
            if (error.max_error > criterion.max_error) {
                return TECOTEST_STATUS_RESULT_ERROR;
            }
            if (error.error_ratio > criterion.ratio_threshold) {
                return TECOTEST_STATUS_RESULT_ERROR;
            }
        } else if (Formula::DIFF4 == func) {
            if (error.max_error > criterion.error_threshold ||
                ((error.max_error < 1 - criterion.error_threshold) && (error.max_error != 0)) ||
                error.max_error < 0) {
                return TECOTEST_STATUS_RESULT_ERROR;
            }
        } else if (error.max_error > criterion.max_error) {
            return TECOTEST_STATUS_RESULT_ERROR;
        }
    }

    return TECOTEST_STATUS_SUCCESS;
    ;
}

// only when failed, call this func. to get error reason
std::vector<std::vector<std::string>> Evaluator::what_cpu() {
    std::vector<std::vector<std::string>> result;

    for (int i = 0; i < error_vec_.size(); ++i) {
        std::vector<std::string> res;
        if (error_vec_[i].criterion.enable == false) {
            result.emplace_back(res);
            continue;
        }

        auto name = error_vec_[i].name;
        auto func = error_vec_[i].criterion.formula;
        Criterion criterion = error_vec_[i].criterion;
        Error error = error_vec_[i].error_teco;

        std::string error_str = "";
        if (error.has_nan) {
            error_str += "The error nan/inf of [" + name + "] is NOT digit.";
            res.emplace_back(error_str);
        } else if (Formula::DIFF3 == func) {
            if (error.max_error > criterion.max_error) {
                std::ostringstream oss_error, oss_threshold;
                oss_error.setf(std::ios::scientific);
                oss_threshold.setf(std::ios::scientific);
                oss_error << error.max_error;
                oss_threshold << criterion.max_error;
                error_str += "The max error " + oss_error.str() + " of [" + name + "] is over " +
                             showFormula(func) + " max threshold " + oss_threshold.str();
                res.emplace_back(error_str);
            } else if (error.error_ratio > criterion.ratio_threshold) {
                std::ostringstream oss_error, oss_ratio_threshold, oss_error_threshold;
                oss_error.setf(std::ios::scientific);
                oss_ratio_threshold.setf(std::ios::scientific);
                oss_error_threshold.setf(std::ios::scientific);
                oss_error << error.error_ratio;
                oss_ratio_threshold << criterion.ratio_threshold;
                oss_error_threshold << criterion.error_threshold;
                error_str += "The error_ratio " + oss_error.str() + " of [" + name + "] is over " +
                             showFormula(func) + " ratio_threshold " + oss_ratio_threshold.str() +
                             " on error_threshold " + oss_error_threshold.str();
                res.emplace_back(error_str);
            }
        } else if (Formula::DIFF4 == func) {
            if (error.max_error > criterion.error_threshold ||
                ((error.max_error < 1 - criterion.error_threshold) && (error.max_error != 0)) ||
                error.max_error < 0) {
                std::ostringstream oss;
                oss.setf(std::ios::scientific);
                oss << error.max_error;
                error_str += "The error " + oss.str() + " of [" + name + "] is over " +
                             showFormula(func) + " threshold " + " [" +
                             std::to_string(1 - criterion.error_threshold) + " , " +
                             std::to_string(criterion.error_threshold) + "]";
                res.emplace_back(error_str);
            }
        } else {
            if (error.max_error > criterion.max_error) {
                std::ostringstream oss_error, oss_threshold;
                oss_error.setf(std::ios::scientific);
                oss_threshold.setf(std::ios::scientific);
                oss_error << error.max_error;
                oss_threshold << criterion.error_threshold;
                error_str += "The error " + oss_error.str() + " of [" + name + "] is over " +
                             showFormula(func) + " threshold " + oss_threshold.str();
                res.emplace_back(error_str);
            }
        }
        result.push_back(res);
    }

    return result;
}

tecotestStatus_t Evaluator::isPassed_bf16() {
    if (error_vec_.empty()) {
        ALLOG(WARNING) << "The result error is empty. Output shape is 0 in prototxt,"
                          " or only test performance now.";
    }

    for (int i = 0; i < error_vec_.size(); ++i) {
        std::vector<std::string> res;
        if (error_vec_[i].criterion.enable == false) {
            continue;
        }

        auto func = error_vec_[i].criterion.formula;
        Criterion criterion = error_vec_[i].criterion;
        Error error = error_vec_[i].error_teco;

        if (error.has_nan) {
            return TECOTEST_STATUS_RESULT_NAN_ERROR;
        } else if (error.has_inf) {
            return TECOTEST_STATUS_RESULT_INF_ERROR;
        } else if (error.max_error > criterion.max_error) {
            return TECOTEST_STATUS_RESULT_ERROR;
        }
    }

    return TECOTEST_STATUS_SUCCESS;
    ;
}

std::vector<std::vector<std::string>> Evaluator::what_bf16() {
    std::vector<std::vector<std::string>> result;

    for (int i = 0; i < error_vec_.size(); ++i) {
        std::vector<std::string> res;
        if (error_vec_[i].criterion.enable == false) {
            result.emplace_back(res);
            continue;
        }

        auto name = error_vec_[i].name;
        auto func = error_vec_[i].criterion.formula;
        Criterion criterion = error_vec_[i].criterion;
        Error error = error_vec_[i].error_teco;

        std::string error_str = "";
        if (error.has_nan || error.has_inf) {
            error_str += "The error nan/inf of [" + name + "] is NOT digit.";
            res.emplace_back(error_str);
        } else if (error.max_error > 0.025) {
            std::ostringstream oss_error, oss_threshold;
            oss_error.setf(std::ios::scientific);
            oss_threshold.setf(std::ios::scientific);
            oss_error << error.max_error;
            oss_threshold << criterion.error_threshold;
            error_str += "The error " + oss_error.str() + " of [" + name + "] is over " +
                         showFormula(func) + " threshold " + oss_threshold.str();
            res.emplace_back(error_str);
        }
        result.push_back(res);
    }
    return result;
}

double Evaluator::computeEfficiency(double num, double latency, double den) {
    if (num < 0 || latency <= 0 || den <= 0) {
        // if didn't set these values
        return -1;
    }
    return num / (latency * den);
}

void Evaluator::copy(const Evaluator *e) { error_vec_ = e->error_vec_; }

std::string showTestStatus(tecotestStatus_t status) {
    switch (status) {
        case TECOTEST_STATUS_SUCCESS: return "TEST_STATUS_SUCCESS";
        case TECOTEST_STATUS_EXECUTE_ERROR: return "TEST_STATUS_EXECUTE_ERROR";
        case TECOTEST_STATUS_RESULT_ERROR: return "TEST_STATUS_RESULT_ERROR";
        case TECOTEST_STATUS_RESULT_NAN_ERROR: return "TEST_STATUS_RESULT_NAN_ERROR";
        case TECOTEST_STATUS_RESULT_INF_ERROR: return "TEST_STATUS_RESULT_INF_ERROR";
        case TECOTEST_STATUS_STABILITY_ERROR: return "TEST_STATUS_STABILITY_ERROR";
        case TECOTEST_STATUS_MEMORY_OUT_ERROR1: return "TEST_STATUS_MEMORY_OUT_ERROR1";
        case TECOTEST_STATUS_MEMORY_OUT_ERROR2: return "TEST_STATUS_MEMORY_OUT_ERROR2";
        case TECOTEST_STATUS_KERNEL_NAME_ERROR: return "TECOTEST_STATUS_KERNEL_NAME_ERROR";
        case TECOTEST_STATUS_FILE_OPEN_FAULT: return "TECOTEST_STATUS_FILE_OPEN_FAULT";
        case TECOTEST_STATUS_PARSE_PT_FAULT: return "TECOTEST_STATUS_PARSE_PT_FAULT";
        default: return "TEST_STATUS_SUCCESS";
    }
}

}  // namespace optest
