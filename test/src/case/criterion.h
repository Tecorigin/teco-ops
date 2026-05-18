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
 
#ifndef CASE_CRITERION_H_  // NOLINT
#define CASE_CRITERION_H_

#include <sys/stat.h>
#include <fstream>
#include <memory>
#include <string>
#include <set>
#include "nlohmann/json.hpp"  // NOLINT

using json = nlohmann::json;

namespace optest {

enum Formula { DIFF1, DIFF2, DIFF3, DIFF3_MAX, DIFF3_MEAN, DIFF4, MAPE, MAE, MSE_RELA };

// constexpr double GOLDEN_THRESHOLD_HALF = 1e-3;
constexpr double GOLDEN_THRESHOLD_FLOAT = 1e-6;

std::string showFormula(optest::Formula f);

struct Criterion {
    Criterion() {}
    Criterion(Formula f, double gt, double ge, bool e = true) :
        formula(f), golden_threshold(gt), golden_eps(ge), enable(e) {}

    double max_error = 1e-2;
    double error_threshold = 1e-4;
    double ratio_threshold = 1e-4;
    // end

    double golden_threshold = 10;
    double golden_eps = GOLDEN_THRESHOLD_FLOAT;
    Formula formula = DIFF3;

    double calc_eps = 0.0;

    bool enable = true;  // if false, only compute it, but won't mark case failed.

    bool operator<(const struct Criterion &c) const {
        if (formula == c.formula) {
            return false;  // for deduplication
        } else {
            return formula < c.formula;
        }
    }

    void print() const {
        std::cout << "    Formula:           " << showFormula(formula) << std::endl;
        std::cout << "    enable:            " << enable << std::endl;
        std::cout << "    golden_threshold:  " << golden_threshold << std::endl;
        std::cout << "    golden_eps:        " << golden_eps << std::endl;
        std::cout << "    error_threshold:   " << error_threshold << std::endl;
        std::cout << "    ratio_threshold:   " << ratio_threshold << std::endl;
    }
};

// todo(maliang):use json instead later
class OpCriterions {
 public:
    static const OpCriterions *instance() {
        std::string filename = "";
        const char *env = std::getenv("TECOAL_TECOTEST_PROJECT");
        if (env == nullptr) {
            filename = "../config/criterions.json";
        } else {
            filename = std::string(env) + "/config/criterions.json";
        }

        static OpCriterions instance(filename);
        return &instance;
    }
    explicit OpCriterions(std::string filename) { getFromJson(filename); }
    ~OpCriterions() {}

    std::set<Criterion> get(const std::string op_name, const std::string mode = "default") const {
        // todo(maliang):
        std::set<Criterion> criterions;
        if (!info_.is_null()) {
            Criterion c;
            c = Criterion(DIFF3_MAX, (double)(info_["default"][0]["diff3_max"]["golden_threshold"]),
                          (double)(info_["default"][0]["diff3_max"]["golden_eps"]), true);
            criterions.insert(c);
            c = Criterion(DIFF3_MEAN,
                          (double)(info_["default"][0]["diff3_mean"]["golden_threshold"]),
                          (double)(info_["default"][0]["diff3_mean"]["golden_eps"]), true);
            criterions.insert(c);

            c = Criterion(DIFF1, (double)(info_["default"][0]["diff1"]["golden_threshold"]),
                          (double)(info_["default"][0]["diff1"]["golden_eps"]), false);
            criterions.insert(c);
            c = Criterion(DIFF2, (double)(info_["default"][0]["diff2"]["golden_threshold"]),
                          (double)(info_["default"][0]["diff2"]["golden_eps"]), false);
            criterions.insert(c);
            c = Criterion(DIFF3, (double)(info_["default"][0]["diff3"]["golden_threshold"]),
                          (double)(info_["default"][0]["diff3"]["golden_eps"]), false);
            criterions.insert(c);
            c = Criterion(MAPE, (double)(info_["default"][0]["mape"]["golden_threshold"]),
                          (double)(info_["default"][0]["mape"]["golden_eps"]), false);
            criterions.insert(c);
            c = Criterion(MAE, (double)(info_["default"][0]["mae"]["golden_threshold"]),
                          (double)(info_["default"][0]["mae"]["golden_eps"]), false);
            criterions.insert(c);
            c = Criterion(MSE_RELA, (double)(info_["default"][0]["mse_rela"]["golden_threshold"]),
                          (double)(info_["default"][0]["mse_rela"]["golden_eps"]), false);
            criterions.insert(c);
        } else {
            Criterion c;
            c = Criterion(DIFF3_MAX, 10, 1e-6, true);
            criterions.insert(c);
            c = Criterion(DIFF3_MEAN, 10, 1e-6, true);
            criterions.insert(c);
        }

        return criterions;
    }

 private:
    void getFromJson(std::string filename) {
        if (filename != "") {
            struct stat buf;
            if (stat(filename.c_str(), &buf) != -1) {
                std::ifstream in(filename);
                in >> info_;
                in.close();
            }
        }
    }

    json info_;
};

}  // namespace optest

#endif  // CASE_CRITERION_H_  // NOLINT
