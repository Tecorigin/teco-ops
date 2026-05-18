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
 
#ifndef COMMON_TOOLS_H_  // NOLINT
#define COMMON_TOOLS_H_

#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include <string>
#include <cstring>
#include <cmath>
#include <exception>
#include "common/tecoallog.h"
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#define GTEST_CHECK(condition, ...)                                                           \
    if (unlikely(!(condition))) {                                                             \
        ALLOG(ERROR) << "Check failed: " #condition ". " #__VA_ARGS__;                        \
        throw std::invalid_argument(std::string(__FILE__) + " +" + std::to_string(__LINE__)); \
    }

#define GTEST_CHECK_CUSTOM(condition, e, ...)                             \
    if (unlikely(!(condition))) {                                         \
        ALLOG(ERROR) << "Check failed: " #condition ". " #__VA_ARGS__;    \
        throw e(std::string(__FILE__) + " +" + std::to_string(__LINE__)); \
    }

#define GTEST_WARNING(condition, ...)                                    \
    if (unlikely(!(condition))) {                                        \
        ALLOG(WARNING) << "Check failed: " #condition ". " #__VA_ARGS__; \
    }

namespace optest {
// for debug
void saveDataToFile(const std::string &file, void *data, size_t size);
void readDataFromFile(const std::string &file, void *data, size_t size);
void removeDir(std::string path);
void createDir(std::string file);
bool isProtoChanged(const std::string &proto_path, const std::string &key_path);
std::string subReplaceFirst(std::string str, std::string old_str, std::string new_str);

bool getEnv(const std::string &env, bool default_ret);
class UnSupportType : public std::exception {
 public:
    // 构造函数设置异常信息
    explicit UnSupportType(std::string p) { this->m_p = p; }

    // 重写 what 函数
    virtual const char *what() { return m_p.c_str(); }

    // 异常信息
    std::string m_p;
};

class FileOpenFault : public std::exception {
 public:
    // 构造函数设置异常信息
    explicit FileOpenFault(std::string p) { this->m_p = p; }

    // 重写 what 函数
    virtual const char *what() { return m_p.c_str(); }

    // 异常信息
    std::string m_p;
};

class ParsePtFault : public std::exception {
 public:
    // 构造函数设置异常信息
    explicit ParsePtFault(std::string p) { this->m_p = p; }

    // 重写 what 函数
    virtual const char *what() { return m_p.c_str(); }

    // 异常信息
    std::string m_p;
};

template <typename T>
bool is_nan(T arg) {
    return std::isnan(arg);
}

template <typename T>
bool is_inf(T arg) {
    return std::isinf(arg);
}
}  // namespace optest

#endif  // COMMON_TOOLS_H_  // NOLINT
