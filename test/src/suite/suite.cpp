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
 
#include "suite/suite.h"

extern optest::GlobalVar global_var;
std::string TestOpFixture::op_name_ = "";  // NOLINT
std::string TestOpFixture::al_type_ = "";  // NOLINT
std::vector<std::string> TestOpFixture::case_path_vec_ = {};
std::shared_ptr<optest::ExecuteConfig> TestOpFixture::ecfg_ =
    std::make_shared<optest::ExecuteConfig>();
std::shared_ptr<optest::ExecuteContext> TestOpFixture::ectx_ = nullptr;  // depends on thread num.

// setup for 1 op
void TestOpFixture::SetUpTestCase() {
    // get op name and case list.
    auto test_case = UnitTest::GetInstance()->current_test_case();
    auto case_name = std::string(test_case->name());

    std::string op_tmp = case_name.substr(0, case_name.find_first_of("/"));
    int sep = op_tmp.find_first_of("_");
    al_type_ = op_tmp.substr(0, sep);
    op_name_ = op_tmp.substr(sep + 1);
    case_path_vec_ = Collector(al_type_, op_name_).list();

    // record info.
    global_var.summary_.suite_count += 1;
    global_var.summary_.case_count += case_path_vec_.size();
    ectx_ = std::make_shared<optest::ExecuteContext>();
    ectx_->init();
}

// teardown for 1 op
void TestOpFixture::TearDownTestCase() {
    if (ectx_ != nullptr) {  // only for thread 1 actually.
        ectx_.reset();
    }

    op_name_.clear();
    case_path_vec_.clear();
}

void TestOpFixture::TearDown() {
    // print result on screen or to file.
    for (auto it = res_.begin(); it != res_.end(); ++it) {  // for case_path
        report(*it);
    }
    res_.clear();
}

// wrap a executor and it status flag
// task buffer is a vector and each element is an ExecutorWrap.
// when setup, set in_used as true, then setup(), then set exe.
// when it's ready, set been_chosen as true, then teardown, then reset().
struct ExecutorWrap {
    ExecutorWrap() = default;
    explicit ExecutorWrap(std::shared_ptr<optest::Executor> e) : exe(e) { in_used = true; }

    std::shared_ptr<optest::Executor> exe = nullptr;
    // flag for teardown polling and pick up.
    // if 1 thread choose this exe, other thread shouldn't choose it.
    bool been_chosen = false;
    // flag for setup
    // if 1 thread choose this exe, other thread shouldn't choose it.
    bool in_used = false;

    void used() { in_used = true; }
    void set(std::shared_ptr<optest::Executor> e) { exe = e; }
    void reset() {
        been_chosen = false;
        in_used = false;
        exe = nullptr;
    }
    bool is_free() { return !in_used; }
    // ready means ready to teardown:
    // exe is not null and not been chose by other thread and is ready.
    bool ready() {
        if (exe == nullptr) {
            return false;
        } else if (been_chosen == true) {
            return false;
        } else {
            return exe->ready();  // kernel is done, is ready.
        }
    }
};

// wrap a executor context and it status flag
// executor context encapsulates handle queue ... and anything can share.
// ecw_vec is a vector and each element is an ExecuteContextWrap.
// when setup, set in_used as true, then setup(), then set exe.
// when it's ready, set been_chosen as true, then teardown, then reset().
struct ExecuteContextWrap {
    std::shared_ptr<optest::ExecuteContext> ectx = nullptr;
    void init() {
        if (ectx == nullptr) {
            ectx = std::make_shared<optest::ExecuteContext>();
            ectx->init();
        }
    }
    void destroy() {
        if (ectx != nullptr) {
            ectx->destroy();
            ectx.reset();
        }
    }
    void reset() { ectx->reset(); }
};

struct Context {
    explicit Context(size_t buffer_num) {
        ecw_vec.resize(buffer_num);
        exe_vec.resize(buffer_num);
        for (auto it = ecw_vec.begin(); it != ecw_vec.end(); ++it) {
            (*it) = std::make_shared<ExecuteContextWrap>();
        }
        for (auto it = exe_vec.begin(); it != exe_vec.end(); ++it) {
            (*it) = std::make_shared<ExecutorWrap>();
        }
    }

    void destroy() {
        exe_vec.clear();
        for (auto it = ecw_vec.begin(); it != ecw_vec.end(); ++it) {
            // free each context (handle queue .. in it)
            (*it)->destroy();
        }
        results.clear();
    }

    // 1 to 1 corresponding
    // exe_vec saved executor
    // and ecw_vec saved context for executor.
    // their size is same, and exe_vec[0] will use context in ecw_vec[0]
    std::vector<std::shared_ptr<ExecuteContextWrap>> ecw_vec;
    // this is so-called task queue, but kernel's latency is not same,
    // so it is not fifo. so use vector.
    std::vector<std::shared_ptr<ExecutorWrap>> exe_vec;

    std::list<optest::EvaluateResult> results;
    // set current device for all thread.
    std::set<std::thread::id, std::greater<std::thread::id>> been_initialized;
};

void TestOpFixture::Run() {
    size_t case_idx = std::get<1>(GetParam());
    auto case_path = case_path_vec_[case_idx];
    std::string case_name;
    std::string::size_type start = case_path.find_last_of("/");
    if (start != std::string::npos) {
        start += 1;
        case_name = case_path.substr(start);
    } else {
        case_name = case_path;
    }
    ALLOG(VLOG) << "case_path: " << case_path;
    auto exceptionHandler = [&case_path, &case_name](
                                const std::exception &e,
                                const optest::tecotestStatus_t &tecotestStatus) {
        ectx_->reset();
        optest::EvaluateResult res;
        res.op_name = op_name_;
        res.case_path = case_path;
        res.case_name = case_name;
        std::vector<std::string> what_tmp;
        what_tmp.emplace_back("Unknown error: maybe exception raised, other info is lost.");
        res.what.emplace_back(what_tmp);
        res.status = tecotestStatus;
        ADD_FAILURE() << "DEVICEOP GTEST: catched " << e.what() << " in single thread mode. (of "
                      << case_path << ")";
        return res;
    };
    try {
        auto exe = getOpExecutor(op_name_);
        exe->op_name_ = op_name_;
        exe->al_type_ = al_type_;
        exe->case_name_ = case_name;
        exe->result()->op_name = op_name_;
        exe->result()->case_path = case_path;
        exe->result()->case_name = case_name;
        exe->init(ectx_);
        exe->setup(case_path, ecfg_);
#ifdef USE_CUDA
        exe->runBaseline();
#elif defined USE_TECO
        exe->getBaseline();
        exe->launch();
        exe->sync();
#endif
        auto res = exe->teardown();
        res_.emplace_back(res);
    } catch (optest::ParsePtFault &e) {
        res_.emplace_back(exceptionHandler(e, optest::TECOTEST_STATUS_PARSE_PT_FAULT));
    } catch (optest::FileOpenFault &e) {
        res_.emplace_back(exceptionHandler(e, optest::TECOTEST_STATUS_FILE_OPEN_FAULT));
    } catch (std::exception &e) {
        res_.emplace_back(exceptionHandler(e, optest::TECOTEST_STATUS_EXECUTE_ERROR));
    }
}

// * print result
// * is pass?
// * calc average
void TestOpFixture::report(optest::EvaluateResult eva) {
    if (global_var.repeat_ > 1) {
        print(eva, true);  // print average log.
    } else {
        print(eva, false);  // normal log.
    }
    if (!eva.is_passed) {
        global_var.summary_.failed_list.emplace_back(eva.case_path);
    }
    recordXml(eva);
    bool passed = eva.is_passed;
    EXPECT_TRUE(passed);
}

void TestOpFixture::print(optest::EvaluateResult eva, bool average) {
    ALLOG(VLOG) << "Result:";
    std::cout << "\n";

    std::cout << HIGHLIGHT << BLUE << "[TensorInfo]:" << RESET << std::endl;
    if (!eva.tensors.empty()) {
        for (auto it = eva.tensors.begin(); it != eva.tensors.end(); ++it) {
            it->print();
        }
    }
    std::cout << "\n";

    std::cout << HIGHLIGHT << BLUE << "[Kernel Detail]:" << RESET << std::endl;
    // for (auto iter = eva.device.kernel_details.begin(); iter != eva.device.kernel_details.end();
    //      ++iter) {
    //     std::cout << std::setprecision(5) << iter->first << ": " << iter->second / 1000
    //               << " (ms)\n";
    // }
    if (eva.status == optest::TECOTEST_STATUS_KERNEL_NAME_ERROR) {
        ALLOG(ERROR) << "maybe some kernel are missing because of not startswith teco_slave";
    }
    for (int i = 0; i < eva.device.kernel_details.size(); i++) {
        std::cout << std::setprecision(5) << eva.device.kernel_details[i].first << ": "
                  << eva.device.kernel_details[i].second / 1000 << " (ms)\n";
        if (eva.device.kernel_details.size() == eva.device.cache_miss_details.size())
            std::cout << eva.device.cache_miss_details[i] << std::endl;
    }
    std::cout << "\n";

    std::cout << HIGHLIGHT << BLUE << "[Performance]:" << RESET << std::endl;
    if (true == average) {
        std::cout << "[Average Interface Time ]: " << std::setprecision(5)
                  << eva.device.interface_time / 1000 << " (ms)\n";
        std::cout << "[Average Hardware Time  ]: " << std::setprecision(5)
                  << eva.device.hardware_time / 1000 << " (ms)\n";
        std::cout << "[Average Launch Time    ]: " << std::setprecision(5)
                  << eva.device.launch_time / 1000 << "/" << eva.device.launch_count << " (ms)\n";
        std::cout << "[TheoryIOs              ]: " << std::setprecision(5) << eva.device.theory_io
                  << " (Bytes)\n";
        std::cout << "[Average IoBandWidth    ]: " << std::setprecision(5)
                  << eva.device.io_bandwidth << " (GB/s)\n";
        std::cout << "[TheoryOps              ]: " << std::setprecision(5) << eva.device.theory_ops
                  << " (Ops)\n";
        std::cout << "[Average ComputeForce   ]: " << std::setprecision(5)
                  << eva.device.compute_force << " (TFLOPS)\n";

    } else {
        std::cout << "[Interface Time    ]: " << std::setprecision(5)
                  << eva.device.interface_time / 1000 << " (ms)\n";
        std::cout << "[Hardware Time     ]: " << std::setprecision(5)
                  << eva.device.hardware_time / 1000 << " (ms)\n";
        std::cout << "[Launch Time       ]: " << std::setprecision(5)
                  << eva.device.launch_time / 1000 << "/" << eva.device.launch_count << " (ms)\n";
        std::cout << "[TheoryIOs         ]: " << std::setprecision(5) << eva.device.theory_io
                  << " (Bytes)\n";
        std::cout << "[IoBandWidth       ]: " << std::setprecision(5) << eva.device.io_bandwidth
                  << " (GB/s)\n";
        std::cout << "[TheoryOps         ]: " << std::setprecision(5) << eva.device.theory_ops
                  << " (Ops)\n";
        std::cout << "[ComputeForce      ]: " << std::setprecision(5) << eva.device.compute_force
                  << " (TFLOPS)\n";
    }
    std::cout << "\n";

    std::cout << HIGHLIGHT << BLUE << "[Compared Device]:" << RESET << std::endl;
    std::cout << "[Compared Device                   ]: " << eva.compared_device.device << "\n";
    std::cout << "[Compared Device Interface Time    ]: " << std::setprecision(5)
              << eva.compared_device.interface_time / 1000 << " (ms)\n";
    std::cout << "[Compared Device Interface Details ]:\n";
    for (auto iter = eva.compared_device.api_details.begin();
         iter != eva.compared_device.api_details.end(); ++iter) {
        std::cout << std::setprecision(5) << iter->first << ": " << iter->second / 1000
                  << " (ms)\n";
    }
    std::cout << "\n";

    auto print_error = [](std::vector<optest::ErrorWrap> errors,
                          std::vector<std::vector<std::string>> whats) {
        std::cout << HIGHLIGHT << BLUE << "[Diffs]:" << RESET << std::endl;
        // for (auto error : errors) {
        //   error.print();
        // }
        std::string last_op_name = "";
        for (int i = 0; i < errors.size(); i++) {
            if (errors[i].name != last_op_name) {
                if (last_op_name != "") std::cout << std::endl;
                std::cout << "[" << errors[i].name << "]:" << std::endl;
            }
            last_op_name = errors[i].name;
            std::cout << "---------------------------" << std::endl;
            auto formula = showFormula(errors[i].criterion.formula);
            errors[i].print_diff3();

            for (auto line : whats[i]) {
                std::cout << line << "\n";
            }
        }
    };

    if (eva.is_passed) {  // if passed, just print errors.
        print_error(eva.errors, eva.what);
        std::cout << "[^      OK ] " << eva.case_path << "\n";
    } else {  // if failed.
        if (!eva.errors.empty()) {
            print_error(eva.errors, eva.what);
        }
        // std::cout << "[Results]:\n";
        // for (auto line : eva.what) {
        //   std::cout << line << "\n";
        // }
        std::cout << "[^  FAILED ] [" << showTestStatus(eva.status) << "] " << eva.case_path
                  << "\n";
    }
}

void TestOpFixture::recordXml(optest::EvaluateResult er) {
    char ymd_time[24];
    time_t timep;
    time(&timep);
    strftime(ymd_time, sizeof(ymd_time), "%Y_%m_%d_%H_%M_%S", localtime(&timep));

    this->RecordProperty("op_name", op_name_);
    this->RecordProperty("case_name", er.case_name);

    /*
    std::string tensor_info = "";
    if (!er.tensors.empty()) {
        tensor_info += "{";
        for (auto it = er.tensors.begin(); it != er.tensors.end() - 1; ++it) {
            tensor_info += it->to_string() + ",";
        }
        tensor_info += er.tensors.back().to_string() + "}";
    }
    this->RecordProperty("tensor_info", tensor_info);
    */

    if (global_var.repeat_ != 0) {
        this->RecordProperty("repeat_count", global_var.repeat_);
    }

    // save date(Year_month_day_hour_minute_second).
    std::ostringstream date_oss;
    date_oss << ymd_time;
    this->RecordProperty("date", date_oss.str());

    // save case_path
    this->RecordProperty("case_path", er.case_path);

    std::ostringstream mhw_oss;
    mhw_oss << std::setprecision(10) << er.device.hardware_time / 1000;
    this->RecordProperty("hardware_time(ms)", mhw_oss.str());

    std::ostringstream interface_oss;
    interface_oss << std::setprecision(10) << er.device.interface_time / 1000;
    this->RecordProperty("interface_time(ms)", interface_oss.str());

    std::ostringstream launch_oss;
    launch_oss << std::setprecision(10) << er.device.launch_time / 1000 << "/"
               << er.device.launch_count;
    this->RecordProperty("launch_time(ms)", launch_oss.str());

    std::ostringstream theory_ios_oss;
    theory_ios_oss << er.device.theory_io;
    this->RecordProperty("theory_ios(Bytes)", theory_ios_oss.str());

    std::ostringstream io_bandwidth_oss;
    io_bandwidth_oss << std::setprecision(10) << er.device.io_bandwidth;
    this->RecordProperty("io_bandwidth(GB/s)", io_bandwidth_oss.str());

    std::ostringstream theory_ops_oss;
    theory_ops_oss << er.device.theory_ops;
    this->RecordProperty("theory_ops(Ops)", theory_ops_oss.str());

    std::ostringstream compute_force_oss;
    compute_force_oss << std::setprecision(10) << er.device.compute_force;
    this->RecordProperty("compute_force(op/s)", compute_force_oss.str());

    auto errors = er.errors;
    auto whats = er.what;
    // for (auto it : errors) {
    for (int i = 0; i < errors.size(); i++) {
        auto it = errors[i];
        auto name = it.name;
        auto teco_error = it.error_teco;
        auto gpu_error = it.error_gpu;
        auto func = showFormula(it.criterion.formula);
        // std::transform(func.begin(), func.end(), func.begin(), ::tolower);
        auto key = "error_" + name + "-" + func;

        auto gpu_key = "GPU_" + key;
        std::ostringstream gpu_key_oss;
        gpu_key_oss.setf(std::ios::scientific);
        gpu_key_oss << gpu_error.index << "," << gpu_error.baseline_value << ","
                    << gpu_error.compare_value << "," << gpu_error.max_error;
        this->RecordProperty(gpu_key, gpu_key_oss.str());

        auto teco_key = "TECO_" + key;
        std::ostringstream teco_key_oss;
        teco_key_oss.setf(std::ios::scientific);
        teco_key_oss << teco_error.index << "," << teco_error.baseline_value << ","
                     << teco_error.compare_value << "," << teco_error.max_error;
        this->RecordProperty(teco_key, teco_key_oss.str());

        auto threshold_key = "THRESHOLD_" + key;
        std::ostringstream threshold_key_oss;
        threshold_key_oss << it.criterion.golden_threshold << ","
                          << teco_error.max_error / (gpu_error.max_error + 1e-9);
        this->RecordProperty(threshold_key, threshold_key_oss.str());
    }

    if (optest::Context::instance()->showTestLog()) {
        json info;
        info["al_type"] = al_type_;
        info["op_name"] = op_name_;
        info["case_name"] = er.case_name;
        // if (er.is_passed) {
        //     info["status"] = "PASS";
        // } else {
        info["status"] = optest::showTestStatus(er.status);
        // }
        info["repeat_count"] = global_var.repeat_;
        info["date"] = ymd_time;  // date_oss.str();
        info["case_path"] = er.case_path;

        std::vector<std::string> dtypes;
        std::vector<std::string> layouts;
        std::vector<std::vector<std::vector<int>>> shapes;
        if (!er.tensors.empty()) {
            for (auto it = er.tensors.begin(); it != er.tensors.end(); ++it) {
                dtypes.push_back(DataType_Name(it->dtype));
                layouts.push_back(TensorLayout_Name(it->layout));
                shapes.push_back(std::vector<std::vector<int>>{it->shape, it->stride});
            }
        }
        info["shape"] = shapes;
        info["dtype"] = dtypes;
        info["layout"] = layouts;

        info["hardware_time(ms)"] = er.device.hardware_time / 1000;
        info["interface_time(ms)"] = er.device.interface_time / 1000;
        info["launch_time(ms)"] = std::to_string(er.device.launch_time / 1000) + "/" +
                                  std::to_string(er.device.launch_count);  // launch_oss.str();
        info["theory_ios(Bytes)"] = er.device.theory_io;
        info["io_bandwidth(GB/s)"] = er.device.io_bandwidth;
        info["theory_ops(Ops)"] = er.device.theory_ops;
        info["compute_force(op/s)"] = er.device.compute_force;
        info["kernel_details"] = er.device.kernel_details;
        info["cache_miss_details"] = er.device.cache_miss_details;
        info["hardware_times(ms)"] = er.device.hardware_times;
        info["result_hash"] = er.result_hash;

        for (int i = 0; i < errors.size(); i++) {
            auto it = errors[i];
            auto name = it.name;
            auto teco_error = it.error_teco;
            auto gpu_error = it.error_gpu;
            auto func = showFormula(it.criterion.formula);
            auto key = "error_" + name + "-" + func;

            if (func == "MSE_RELA") {
                info["diffs"][func][name] = {{"TECO",
                                              {{"index", teco_error.index},
                                               {"baseline_value", teco_error.baseline_value},
                                               {"compare_value", teco_error.compare_value},
                                               {"max_error", teco_error.max_error}}},
                                             {"THRESHOLD",
                                              {{"golden_threshold", it.criterion.golden_threshold},
                                               {"max_error", teco_error.max_error}}}};
            } else {
                info["diffs"][func][name] = {
                    {"TECO",
                     {{"index", teco_error.index},
                      {"baseline_value", teco_error.baseline_value},
                      {"compare_value", teco_error.compare_value},
                      {"max_error", teco_error.max_error}}},
                    {"GPU",
                     {{"index", gpu_error.index},
                      {"baseline_value", gpu_error.baseline_value},
                      {"compare_value", gpu_error.compare_value},
                      {"max_error", gpu_error.max_error}}},
                    {"THRESHOLD",
                     {{"golden_threshold", it.criterion.golden_threshold},
                      {"max_error", teco_error.max_error / (gpu_error.max_error + 1e-9)}}}};
            }
        }
        // compared device
        info["compared_device"] = er.compared_device.device;
        info["compared_device_interface_time(ms)"] = er.compared_device.interface_time / 1000;
        info["compared_device_api_details"] = er.compared_device.api_details;
        std::cout << "--------------test result json------------------" << std::endl;
        std::cout << info << std::endl;
    }
}
