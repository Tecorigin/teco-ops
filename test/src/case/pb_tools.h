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

#ifndef CASE_PB_TOOLS_H_  // NOLINT
#define CASE_PB_TOOLS_H_

#include <random>
#include <thread>
#include <vector>
#include <string>
#include <unordered_map>
#include "test_proto/optest.pb.h"
#include "common/half.hpp"
#include "common/tecoallog.h"
#include "common/bfloat16.h"
#include "common/tools.h"

namespace optest {

size_t getSizeOfDataType(testpt::DataType dtype);

// include stride
// if no stride this count == shape count
size_t shapeStrideCount(const testpt::Shape *shape);
// not include stride
size_t shapeElementCount(const testpt::Shape *shape);

int16_t cvtFloatToHalf(float x);
int64_t cvtFloatToInt64(float x);
float cvtHalfToFloat(int16_t);

void arrayCastFloatToHalf(int16_t *dst, float *src, int num);
void arrayCastFloatToInt64(int64_t *dst, float *src, int num);
void arrayCastHalfToFloat(float *dst, int16_t *src, int num);
void arrayCastHalfToInt8or16HalfUp(void *dst, int16_t *src, int pos, int num, int int8or16);
uint64_t GenNumberOfFixedWidth(uint64_t a, int witdh);

size_t proc_usage_peak();
std::unordered_map<std::string, std::vector<std::string>> readFileByLine(const std::string &file);
// half mult
int float_mult(int in_a, int in_b, int float_16or32, int round_mode, int ieee754);

// half add
int float_add(int in_a, int in_b, int float_16or32, int round_mode, int add_or_sub, int ieee754);

void generateRandomDataSerialBf16(uint16_t *data, size_t start, size_t end, int seed,
                                  const testpt::RandomData *random_param);
void generateRandomDataBf16(uint16_t *data, size_t count, const testpt::RandomData *random_param);
// force fix to float
template <typename FixedType>
void forceFixToFloat(FixedType *src, float *dst, const size_t num) {
    for (size_t i = 0; i < num; ++i) {
        dst[i] = float(int(src[i]));
    }
}

typedef enum {
    ROUND_MODE_TO_ZERO,
    ROUND_MODE_OFF_ZERO,
    ROUND_MODE_UP,
    ROUND_MODE_DOWN,
    ROUND_MODE_NEAREST_OFF_ZERO,
    ROUND_MODE_NEAREST_EVEN,
    ROUND_MODE_MATH,
    ROUND_MODE_NO,
} round_mode_t;

// half -> int8/int16
template <typename FixedType>
void cvtHalfToFixed(const float *src, FixedType *dst, const size_t num, const int position = 0,
                    const float scale = 1.0, const int offset = 0) {
    const float max = pow(2, sizeof(FixedType) * 8 - 1) + (-1);
    const float min = pow(2, sizeof(FixedType) * 8 - 1) * (-1);
    for (size_t i = 0; i < num; ++i) {
        int16_t res = float_mult(cvtFloatToHalf(src[i]), cvtFloatToHalf(scale), 0,
                                 ROUND_MODE_NEAREST_EVEN, 1);
        // use 10 because half exponend width only 5 bit
        int pos_tmp = position >= 0 ? 10 : -10;
        float tmp = powf(2, -pos_tmp);
        int16_t tmp_half = cvtFloatToHalf(tmp);
        int16_t offset_half = cvtFloatToHalf(offset);
        for (int cycle = 0; cycle < position / pos_tmp; ++cycle) {
            res = float_mult(res, tmp_half, 0, ROUND_MODE_NEAREST_EVEN, 1);
        }
        if (position % pos_tmp) {
            tmp = pow(2, -(position % pos_tmp));
            int16_t tmp_half = cvtFloatToHalf(tmp);
            res = float_mult(res, tmp_half, 0, ROUND_MODE_NEAREST_EVEN, 1);
        }
        res = float_add(res, offset_half, 0, ROUND_MODE_NEAREST_EVEN, 0, 1);
        float res1 = cvtHalfToFloat(res);
        if (res1 > max) {
            res1 = max;
        } else if (res1 < min) {
            res1 = min;
        }
        dst[i] = static_cast<FixedType>(round(res1));
    }
}

void setValue(void *data, size_t count, testpt::DataType dtype, testpt::RandomData *random_param);

template <typename T>
void generateRandomDataSerial(T *data, size_t start, size_t end, int seed,
                              const testpt::RandomData *random_param) {
    size_t count = end - start;
    // int seed = random_param->seed();
    // generate random data
    std::default_random_engine re(seed);  // re for random engine
    if (random_param->distribution() == testpt::UNIFORM) {
        double lower = random_param->lower_bound();
        double upper = random_param->upper_bound();

        if (std::is_same<T, uint8_t>::value || std::is_same<T, uint16_t>::value ||
            std::is_same<T, uint32_t>::value || std::is_same<T, uint64_t>::value) {
            if (upper < 0) {
                ALLOG(WARNING) << "uint8/uint16/uint32/uint64: upper < 0, the data will be all 0."
                               << " lower=" << lower << ", upper=" << upper;
                upper = 0.0;
                lower = 0.0;
            } else if (lower < 0) {
                ALLOG(WARNING) << "uint8/uint16/uint32/uint64: lower < 0, use 0 instead."
                               << " lower=" << lower << ", upper=" << upper;
                lower = 0.0;
            }
        }

        if (lower >= upper) {
            ALLOG(WARNING) << "lower >= upper, the data will be all upper."
                           << " lower=" << lower << ", upper=" << upper;
            if (std::is_same<T, half_float::half>::value) {
                for (size_t i = start; i < end; ++i) {
                    if (std::fabs(upper) < 7e-5) {
                        data[i] = 0.0;
                    } else {
                        data[i] = upper;
                    }
                }
            } else {
                for (size_t i = start; i < end; ++i) {
                    data[i] = upper;
                }
            }
        } else {
            // uniform_real_distribution is [lower, upper)
            std::uniform_real_distribution<double> dis(lower, upper);
            if (std::is_same<T, half_float::half>::value) {
                for (size_t i = start; i < end; ++i) {
                    double tmp = dis(re);
                    if (std::fabs(tmp) < 7e-5) {
                        data[i] = 0.0;
                    } else {
                        data[i] = tmp;
                    }
                }
            } else {
                for (size_t i = start; i < end; ++i) {
                    data[i] = dis(re);
                }
            }
        }
    } else if (random_param->distribution() == testpt::GAUSSIAN) {
        double mu = random_param->mu();
        double sigma = random_param->sigma();
        std::normal_distribution<double> dis(mu, sigma);
        if (std::is_same<T, uint8_t>::value || std::is_same<T, uint16_t>::value ||
            std::is_same<T, uint32_t>::value || std::is_same<T, uint64_t>::value) {
            for (size_t i = start; i < end; ++i) {
                data[i] = std::fabs(dis(re));
            }
        } else if (std::is_same<T, half_float::half>::value) {
            for (size_t i = start; i < end; ++i) {
                double tmp = dis(re);
                if (std::fabs(tmp) < 7e-5) {
                    data[i] = 0.0;
                } else {
                    data[i] = tmp;
                }
            }
        } else {
            for (size_t i = start; i < end; ++i) {
                data[i] = dis(re);
            }
        }

    } else if (random_param->distribution() == testpt::UNIQUE) {
        double lower = random_param->lower_bound();
        double upper = random_param->upper_bound();

        if (std::is_same<T, uint8_t>::value || std::is_same<T, uint16_t>::value ||
            std::is_same<T, uint32_t>::value || std::is_same<T, uint64_t>::value) {
            if (upper < 0) {
                ALLOG(WARNING) << "uint8/uint16/uint32/uint64: upper < 0, the data will be all 0."
                               << " lower=" << lower << ", upper=" << upper;
                upper = 0.0;
                lower = 0.0;
            } else if (lower < 0) {
                ALLOG(WARNING) << "uint8/uint16/uint32/uint64: lower < 0, use 0 instead."
                               << " lower=" << lower << ", upper=" << upper;
                lower = 0.0;
            }

            if (upper - lower < count) {
                ALLOG(WARNING)
                    << "uint8/uint16/uint32/uint64: upper-lower<count, some data must be same."
                    << " lower=" << lower << ", upper=" << upper << ", count=" << count;
            }
        }

        if (lower >= upper) {
            ALLOG(WARNING) << "lower >= upper, the data will be all upper."
                           << " lower=" << lower << " upper=" << upper;
            if (std::is_same<T, half_float::half>::value) {
                for (size_t i = start; i < end; ++i) {
                    if (std::fabs(upper) < 7e-5) {
                        data[i] = 0.0;
                    } else {
                        data[i] = upper;
                    }
                }
            } else {
                for (size_t i = start; i < end; ++i) {
                    data[i] = upper;
                }
            }
        } else {
            double step = (upper - lower) / count;
            double tmp = lower;
            if (std::is_same<T, half_float::half>::value) {
                for (size_t i = start; i < end; ++i) {
                    if (std::fabs(tmp) < 7e-5) {
                        data[i] = 0.0;
                    } else {
                        data[i] = tmp;
                    }
                    tmp += step;
                }
            } else {
                for (size_t i = start; i < end; ++i) {
                    data[i] = tmp;
                    tmp += step;
                }
            }
            // srand(seed);
            // std::random_shuffle(data + start, data + end);
        }
    }

    // positive
    if (random_param->has_positive() && random_param->positive()) {
        for (size_t i = start; i < end; ++i) {
            data[i] = std::fabs(data[i]);
        }
    }

    // negative
    if (random_param->has_negative() && random_param->negative()) {
        for (size_t i = start; i < end; ++i) {
            data[i] = -std::fabs(data[i]);
        }
    }
}

constexpr int INIT_THREAD_NUM = 32;
template <typename T>
void generateRandomData(T *data, size_t count, const testpt::RandomData *random_param) {
    int seed = random_param->seed();
    if (count < 1024 * INIT_THREAD_NUM) {
        generateRandomDataSerial<T>(data, 0, count, seed, random_param);
    } else {
        std::thread *thread[INIT_THREAD_NUM];
        size_t per_thread_num = (count + INIT_THREAD_NUM - 1) / INIT_THREAD_NUM;
        for (int i = 0; i < INIT_THREAD_NUM; i++) {
            size_t start = i * per_thread_num;
            size_t end = start + per_thread_num > count ? count : start + per_thread_num;
            thread[i] = new std::thread(generateRandomDataSerial<T>, data, start, end, seed + i,
                                        random_param);
        }
        for (int i = 0; i < INIT_THREAD_NUM; i++) {
            thread[i]->join();
        }
        for (int i = 0; i < INIT_THREAD_NUM; i++) {
            delete thread[i];
            thread[i] = nullptr;
        }
    }

    // note: zero nan inf may be in same position, and finally is inf
    // no matters, for random

    // zero_dropout
    if (random_param->has_zero_dropout() && random_param->zero_dropout() > 0) {
        float dropout = random_param->zero_dropout();
        size_t end_tmp = count * dropout;
        srand(seed);
        for (size_t i = 0; i < count; ++i) {
            size_t value = rand() % count;
            if (value < end_tmp) {
                data[i] = 0.f;
            }
        }
    }

    // nan_dropout
    if (random_param->has_nan_dropout() && random_param->nan_dropout() > 0) {
        float dropout = random_param->nan_dropout();
        size_t end_tmp = count * dropout;
        srand(seed + 10);
        for (size_t i = 0; i < count; ++i) {
            size_t value = rand() % count;
            if (value < end_tmp) {
                data[i] = 0.0 / 0.0;
            }
        }
    }

    // inf_dropout
    if (random_param->has_inf_dropout() && random_param->inf_dropout() > 0) {
        float dropout = random_param->inf_dropout();
        size_t end_tmp = count * dropout;
        srand(seed + 20);
        for (size_t i = 0; i < count; ++i) {
            size_t value = rand() % count;
            if (value < end_tmp) {
                data[i] = 1.0 / 0.0;
            }
        }
    }
}

}  // namespace optest

#endif  // CASE_PB_TOOLS_H_  // NOLINT
