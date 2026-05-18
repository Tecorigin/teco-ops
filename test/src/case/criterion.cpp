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
 
#include <sstream>
#include <fstream>
#include <iostream>
#include "case/criterion.h"
#include "common/tools.h"

namespace optest {

std::string showFormula(Formula f) {
    switch (f) {
        case DIFF1: return "DIFF1";
        case DIFF2: return "DIFF2";
        case DIFF3: return "DIFF3";
        case DIFF3_MAX: return "DIFF3_MAX";
        case DIFF3_MEAN: return "DIFF3_MEAN";
        case DIFF4: return "DIFF4";
        case MAPE: return "MAPE";
        case MAE: return "MAE";
        case MSE_RELA: return "MSE_RELA";
        default:
            ALLOG(ERROR) << "Got an unsupported criterion formula.";
            throw std::invalid_argument(std::string(__FILE__) + " +" + std::to_string(__LINE__));
    }
}

}  // namespace optest
