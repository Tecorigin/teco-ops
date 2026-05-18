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
 
#ifndef SUITE_SUITE_H_    // NOLINT
#define SUITE_SUITE_H_

#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <list>
#include <tuple>
#include <memory>
#include <functional>
#include <queue>
#include <set>
#include <thread>
#include <mutex>
#include "gtest/gtest.h"
#include "case/case_collector.h"
#include "case/pb_tools.h"
#include "case/evaluator.h"
#include "suite/env.h"
#include "suite/executor.h"
#include "suite/op_register.h"
using namespace ::testing;

// string is op name
// size_t is case_id
// for multi thread: get case_list by op_name, and case_id is useless.
// for single thread: get case_list by op_name, and get case_path by case_list +
// case_id
class TestOpFixture : public TestWithParam<std::tuple<std::string, size_t>> {
 public:
    TestOpFixture() {}
    virtual ~TestOpFixture() {}

    // setup for 1 test suite
    static void SetUpTestCase();
    static void TearDownTestCase();

    // setup for 1 test case
    void SetUp() {}
    void TearDown();
    void Run();

    static std::string op_name_;
    static std::string al_type_;
    static std::vector<std::string> case_path_vec_;
    static std::shared_ptr<optest::ExecuteContext> ectx_;
    static std::shared_ptr<optest::ExecuteConfig> ecfg_;

 private:
    std::list<optest::EvaluateResult> res_;

    void print(optest::EvaluateResult eva, bool average = false);
    void report(optest::EvaluateResult eva);
    void recordXml(optest::EvaluateResult eva);
};

#endif  // SUITE_SUITE_H_  // NOLINT
