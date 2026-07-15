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

#include <memory>
#include <iostream>
#include <utility>
#include <functional>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <chrono>
#include "common/variable.h"
#include "device/device.h"
#include "suite/executor.h"
#include "nlohmann/json.hpp"
using json = nlohmann::json;

#define GTEST_DEBUG_ENABLE 0  // todo(maliang)

extern optest::GlobalVar global_var;

namespace optest {
Executor::~Executor() { ALLOG(INFO) << "Executor end.."; }

void Executor::init(const std::shared_ptr<ExecuteContext> ectx) {
    ALLOG(INFO) << "Executor start.";
    exe_context_ = ectx;
    setContext();

    parser_ = std::make_shared<Parser>();
    eva_ = std::make_shared<Evaluator>();
}

static int save_to_file(void *data,  size_t size, const std::string& filename) {
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        std::cerr << "无法打开文件进行写入！" << std::endl;
        return 1;
    }

    outFile.write(reinterpret_cast<char*>(data), size);
    if (!outFile) {
        std::cerr << "写入文件时发生错误！" << std::endl;
        return 1;
    }
    outFile.close();
    return 0;
}

static int read_from_file(void *data,  size_t size, const std::string& filename) {
    std::ifstream inFile(filename, std::ios::binary);

    if (!inFile) {
        std::cerr << "无法打开文件进行读取！" << std::endl;
        return 1;
    }

   inFile.read(reinterpret_cast<char*>(data), size);

    if (!inFile) {
        std::cerr << "读取文件时发生错误！" << std::endl;
        return 1;
    }

    // 关闭文件
    inFile.close();
    return 0;
}

void Executor::setup(std::string file, const std::shared_ptr<ExecuteConfig> ecfg) {
    exe_config_ = ecfg;
    parser_->parse(file);
    eva_res_.case_path = file;
    ALLOG(INFO) << "Param check.";
    paramCheck();
    paramParse();
    ALLOG(INFO) << "Descriptor create.";
    createDesc();
    ALLOG(INFO) << "Host malloc and init.";
    pt_changed_ = false;
    is_bf16_ = false;
    for (int i = 0; i < parser_->inputs().size(); ++i) {
        if (parser_->getInputDataType(i) == 12) {
            is_bf16_ = true;
            break;
        }
    }
    for (int i = 0; i < parser_->outputs().size(); ++i) {
        if (parser_->getOutputDataType(i) == 12) {
            is_bf16_ = true;
            break;
        }
    }
    if (Context::instance()->onlyTestPerformance() && Context::instance()->notInitInput()) {
    } else {
        hostMalloc();
        initInput();
    }
    if (!Context::instance()->onlyTestPerformance()) {
        baselineMalloc();
        if (Context::instance()->compareWithGPU() && !is_bf16_) {
            gpuMalloc();
        }
    }
    ALLOG(INFO) << "Device malloc and init.";
    deviceMalloc();
    if (Context::instance()->onlyTestPerformance() && Context::instance()->notInitInput()) {
    } else {
        memcpyHost2Device();
    }
    ALLOG(INFO) << "Param generation.";
    paramGeneration();

    // last because depend on desc
    ALLOG(INFO) << "Device malloc and init (for workspace).";
    workspaceMalloc();
}

void Executor::runBaseline() {
    // call gpuCompute & save to baseline
#ifdef USE_CUDA
    gpuCompute();

    outputs_ = host_output;
    inputs_ = host_input;

    getPythonPath();
    getProtoPath();
    getDataPath();
    getDataParams();

    struct stat info;
    if (stat(data_path_.c_str(), &info) != 0) {
        mkdir(data_path_.c_str(), 0755);
    }

    // save gpu result
    // copy dev to host
    for (int index = 0; index < input_params_.size(); index++) {
        auto ts = parser_->input(index);
        if (ts->input_reuse && ts->total_count) {
            auto filename = data_path_ + std::string("cpu") + "_reuse_" + std::to_string(index);
            checkScdaErrors(
                scdaMemcpy((void*)(host_input[index]), (void*)(dev_input[index]), ts->size_in_bytes, MemcpyDeviceToHost));
            save_to_file((void*)(host_input[index]), ts->size_in_bytes, filename);
        }
    }
    for (int index = 0; index < output_params_.size(); index++) {
        auto ts = parser_->output(index);
        if (ts->total_count) {
            auto filename = data_path_ + std::string("cpu") + "_output_" + std::to_string(index);
            checkScdaErrors(
                scdaMemcpy((void*)(host_output[index]), (void*)(dev_output[index]), ts->size_in_bytes, MemcpyDeviceToHost));
            save_to_file((void*)(host_output[index]), ts->size_in_bytes, filename);
        }
    }
#endif
}

void Executor::getBaseline() {
#ifdef USE_TECO
    outputs_ = baseline_output;
    inputs_ = baseline_input;

    getPythonPath();
    getProtoPath();
    getDataPath();
    getDataParams();

    struct stat info;
    if (stat(data_path_.c_str(), &info) != 0) {
        // std::cout<<"need run gpu first"<<std::endl;
        cpuCompute();
        return;
    }

    for (int index = 0; index < input_params_.size(); index++) {
        auto ts = parser_->input(index);
        if (ts->input_reuse && ts->total_count) {
            auto filename = data_path_ + std::string("cpu") + "_reuse_" + std::to_string(index);
            read_from_file((void*)(baseline_input[index]), ts->size_in_bytes, filename);
        }
    }
    for (int index = 0; index < output_params_.size(); index++) {
        auto ts = parser_->output(index);
        if (ts->total_count) {
            auto filename = data_path_ + std::string("cpu") + "_output_" + std::to_string(index);
            read_from_file((void*)(baseline_output[index]), ts->size_in_bytes, filename);
        }
    }
#endif
}

void Executor::warmUp() {
    // for first inaccurate profiler data when using cuda
    Profiler::start();
    for (int i = 0; i < global_var.warm_repeat_; i++) {
        if (i == 0) {
            if (!Context::instance()->onlyTestPerformance()) forward();
        }
        compute();
    }
    checkScdaErrors(scdaStreamSynchronize(exe_context_->stream));
    Profiler::end();
    Profiler::duration();
}

void Executor::getPerformance() {
    eva_res_.device.interface_time = 0;
    eva_res_.device.hardware_time = 0;
    eva_res_.device.launch_time = 0;
    eva_res_.device.launch_count = 0;
    eva_res_.device.kernel_details.clear();
    eva_res_.device.hardware_times.clear();

    std::vector<ProfilePerfInfo> perf_results;
    if (global_var.repeat_ > 0) {
        Profiler::start();
        for (int i = 0; i < global_var.repeat_; i++) {
            compute();
        }
        checkScdaErrors(scdaStreamSynchronize(exe_context_->stream));
        Profiler::end();
        ProfilePerfInfo res = Profiler::duration();

        res = res / global_var.repeat_;
        eva_res_.device.interface_time = res.interface_time;
        eva_res_.device.hardware_time = res.kernel_time;
        eva_res_.device.launch_time = res.launch_time;
        eva_res_.device.launch_count = res.launch_count;
        int kernel_num = res.kernel_details.size() / global_var.repeat_;
        for (int i = 0; i < kernel_num; i++) {
            eva_res_.device.kernel_details.push_back(res.kernel_details[i]);
        }
        if (res.cache_miss_details.size() != res.kernel_details.size()) {
            ALLOG(ERROR) << "cache miss num != kernel num";
            eva_res_.status = TECOTEST_STATUS_KERNEL_NAME_ERROR;
        } else {
            for (int i = 0; i < kernel_num; i++) {
                eva_res_.device.cache_miss_details.push_back(res.cache_miss_details[i]);
            }
        }
        for (int i = 0; i < global_var.repeat_; i++) {
            double time = 0.0;
            for (int j = 0; j < kernel_num; j++) {
                time += res.kernel_details[i * kernel_num + j].second;
            }
            eva_res_.device.hardware_times.push_back(time / 1000);
        }
        int count = global_var.repeat_ / 10;
        std::sort(eva_res_.device.hardware_times.begin(), eva_res_.device.hardware_times.end());
        double tot_time = std::accumulate(eva_res_.device.hardware_times.begin() + count,
                                          eva_res_.device.hardware_times.end() - count, 0.0f);
        eva_res_.device.hardware_time =
            tot_time * 1000.0 / ((global_var.repeat_ - count * 2) * 1.0);
    }
}

void Executor::saveResultHash() {
    unsigned int *output_hash;
    scdaMalloc((void **)&output_hash, sizeof(unsigned int) * getOutTensorNum());
    unsigned int *next_once_hash = calcOutputHash(output_hash);
    int tensor_num = (next_once_hash - output_hash);
    unsigned int *host_hash = (unsigned int *)malloc(tensor_num * sizeof(unsigned int));
    checkScdaErrors(
        scdaMemcpy(host_hash, output_hash, tensor_num * sizeof(unsigned int), MemcpyDeviceToHost));
    std::vector<std::string> output_name;
    for (int index = 0; index < host_input.size(); index++) {
        MetaTensor *ts = parser_->input(index);
        if (ts->total_count == 0 || !ts->input_reuse) {
            continue;
        }
        output_name.push_back(ts->name);
    }
    for (int index = 0; index < host_output.size(); index++) {
        MetaTensor *ts = parser_->output(index);
        if (ts->total_count == 0) {
            continue;
        }
        output_name.push_back(ts->name);
    }
    for (int i = 0; i < tensor_num; i++) {
        eva_res_.result_hash.insert(std::make_pair(output_name[i], *(host_hash + i)));
    }
    free(host_hash);
    if (!scdaFree(output_hash)) {
        ALLOG(ERROR) << "hash workspace memory out of bounds!!!\n";
        ADD_FAILURE() << "hash workspace memory out of bounds!!!\n";
        eva_res_.status = TECOTEST_STATUS_MEMORY_OUT_ERROR1;
    }
}

void Executor::testStability() {
    if (global_var.stable_repeat_ > 0) {
        unsigned int *output_hash;
        scdaMalloc((void **)&output_hash, sizeof(unsigned int) * 2 * getOutTensorNum());

        checkScdaErrors(scdaSetDevice(global_var.dev_id_));
        memcpyHost2Device();
        checkScdaErrors(scdaSetDevice(global_var.kernel_id_));
        initLDM(exe_context_->stream);
        if (!Context::instance()->onlyTestPerformance()) forward();
        checkScdaErrors(scdaStreamSynchronize(exe_context_->stream));
        compute();
        unsigned int *first_once_hash = calcOutputHash(output_hash);
        if (Context::instance()->saveUnstableResult()) {
            // memcpyDevice2Host();
            saveUnstableResult(0);
        }
        checkScdaErrors(scdaStreamSynchronize(exe_context_->stream));
        bool is_stable = true;
        for (int i = 1; i < global_var.stable_repeat_; i++) {
            checkScdaErrors(scdaSetDevice(global_var.dev_id_));
            memcpyHost2Device();
            checkScdaErrors(scdaSetDevice(global_var.kernel_id_));
            initLDM(exe_context_->stream);
            if (!Context::instance()->onlyTestPerformance()) forward();
            checkScdaErrors(scdaStreamSynchronize(exe_context_->stream));
            compute();
            unsigned int *next_once_hash = calcOutputHash(first_once_hash);
            checkScdaErrors(scdaStreamSynchronize(exe_context_->stream));

            int tensor_num = (next_once_hash - output_hash) / 2;
            checkOutputHash(output_hash, tensor_num, 2);
            if (eva_res_.status == TECOTEST_STATUS_STABILITY_ERROR) {
                is_stable = false;
                if (Context::instance()->saveUnstableResult()) {
                    // memcpyDevice2Host();
                    saveUnstableResult(i);
                }
                ALLOG(ERROR) << "Result hash not same with the first time at the " << i
                             << "th time";
                break;
            }
        }

        if (!scdaFree(output_hash)) {
            ALLOG(ERROR) << "hash workspace memory out of bounds!!!\n";
            ADD_FAILURE() << "hash workspace memory out of bounds!!!\n";
            eva_res_.status = TECOTEST_STATUS_MEMORY_OUT_ERROR1;
        }
        if (Context::instance()->saveUnstableResult() && is_stable) {
            removeStableResult();
        }
    }
}

void Executor::launch() {
#ifdef USE_CUDA
return;
#elif defined USE_TECO
    ALLOG(VLOG) << "TECO compute";

    checkScdaErrors(scdaSetDevice(global_var.kernel_id_));
    ALLOG(VLOG) << "====>warm up repeat: " << global_var.warm_repeat_;
    warmUp();
    ALLOG(VLOG) << "====>performance repeat: " << global_var.repeat_;
    getPerformance();
    if (global_var.stable_) {
        if (global_var.stable_repeat_ <= 0) global_var.stable_repeat_ = global_var.repeat_;
    }
    ALLOG(VLOG) << "====>stablity repeat: " << global_var.stable_repeat_;
    testStability();
    checkScdaErrors(scdaSetDevice(global_var.dev_id_));

    ALLOG(VLOG) << "====>calc teco result";
    // device data may be changed, so init device data and clac result
    memcpyHost2Device();
    // initLDM(exe_context_->stream);
    // checkScdaErrors(scdaStreamSynchronize(exe_context_->stream));
    forward();
    checkScdaErrors(scdaStreamSynchronize(exe_context_->stream));
    if (eva_res_.device.kernel_details.empty()) {
        Profiler::start();
        compute();
        checkScdaErrors(scdaStreamSynchronize(exe_context_->stream));
        Profiler::end();
        auto res = Profiler::duration();
        eva_res_.device.kernel_details = res.kernel_details;
        eva_res_.device.cache_miss_details = res.cache_miss_details;
    } else {
        compute();
        checkScdaErrors(scdaStreamSynchronize(exe_context_->stream));
    }
#endif
}

void Executor::getDeviceOutputPath(std::string name, int index) {
    // std::time_t t = std::time(nullptr);
    // std::tm* now = std::localtime(&t);
    // std::string time_now = std::to_string((now->tm_year)+1900)+ std::to_string(now->tm_mon+1) +
    // std::to_string(now->tm_mday) + std::to_string(now->tm_hour) +
    // std::to_string(now->tm_min)+std::to_string(now->tm_sec); std::cout<<proto_path_<<std::endl;
    // std::string case_name = case_name_.substr(ldx+1, rdx-ldx-1);
    unsigned int case_hash = std::hash<std::string>()(proto_path_);
    device_output_path_ = device_output_dir_ + "/" + op_name_ + "/" + std::to_string(case_hash) +
                          "_" + name + "_" + std::to_string(index);
    device_output_casepath_.insert(std::make_pair(device_output_path_, proto_path_));
}

void Executor::saveDeviceOutput() {
    // static int count = 0;
    // std::cout<<"this is "<<++count<<" time"<<std::endl;
    char dir[1024] = "";
    std::string save_device_dir_name = Context::instance()->getDeviceOutputDir();
    // std::cout<<"this is proto path "<<proto_path_<<std::endl;
    if (getcwd(dir, 1024) != nullptr) {
        device_output_dir_ = std::string(dir) + "/" + save_device_dir_name;
        // std::cout<<teco_output_dir_<<std::endl;
    } else {
        size_t idx = proto_path_.find_last_of("/");
        device_output_dir_ = proto_path_.substr(0, idx + 1) + save_device_dir_name;
    }
    for (int index = 0; index < host_input.size(); index++) {
        MetaTensor *ts = parser_->input(index);
        if (ts->total_count == 0 || !ts->input_reuse) {
            continue;
        }
        getDeviceOutputPath(ts->name, index);
        saveDataToFile(device_output_path_, host_input[index],
                       parser_->input(index)->size_in_bytes);
    }
    for (int index = 0; index < host_output.size(); index++) {
        // std::cout<<"save "<<index<<std::endl;
        getDeviceOutputPath(parser_->output(index)->name, index);
        saveDataToFile(device_output_path_, host_output[index],
                       parser_->output(index)->size_in_bytes);
        // std::ofstream teco_file(device_output, std::ios::binary);
    }
}

void Executor::saveUnstableResult(int times) {
    if (times == 0) {
        getProtoPath();
        unstableMalloc();
    }
    char dir[1024] = "";
    std::string save_device_dir_name = Context::instance()->getUnstableResultDir();
    if (getcwd(dir, 1024) != nullptr) {
        device_output_dir_ = std::string(dir) + "/" + save_device_dir_name;
    } else {
        size_t idx = proto_path_.find_last_of("/");
        device_output_dir_ = proto_path_.substr(0, idx + 1) + save_device_dir_name;
    }
    device_output_dir_ = device_output_dir_ + "/" + op_name_;
    size_t idx_l = proto_path_.find_last_of("/"), idx_r = proto_path_.find_last_of(".");
    device_output_dir_ =
        device_output_dir_ + "/" + proto_path_.substr(idx_l + 1, idx_r - idx_l - 1);
    // createDir(device_output_dir_);
    auto getSaveTensorName = [&](std::string name, std::string save_dir, int times) {
        std::string save_name = save_dir + "/" + name + "_times_" + std::to_string(times);
        return save_name;
    };
    for (int index = 0; index < unstable_input.size(); index++) {
        MetaTensor *ts = parser_->input(index);
        if (ts->total_count == 0 || !ts->input_reuse) {
            continue;
        }
        checkScdaErrors(scdaMemcpy(unstable_input[index], dev_input[index], ts->size_in_bytes,
                                   MemcpyDeviceToHost));
        auto save_name = getSaveTensorName(parser_->input(index)->name, device_output_dir_, times);
        saveDataToFile(save_name, unstable_input[index], parser_->input(index)->size_in_bytes);
    }
    for (int index = 0; index < unstable_output.size(); index++) {
        // std::cout<<"save "<<index<<std::endl;
        MetaTensor *ts = parser_->output(index);
        checkScdaErrors(scdaMemcpy(unstable_output[index], dev_output[index], ts->size_in_bytes,
                                   MemcpyDeviceToHost));
        auto save_name = getSaveTensorName(parser_->output(index)->name, device_output_dir_, times);
        saveDataToFile(save_name, unstable_output[index], parser_->output(index)->size_in_bytes);
    }
    if (times > 0) {
        unstableFree();
    }
}

void Executor::removeStableResult() {
    char dir[1024] = "";
    std::string save_device_dir_name = Context::instance()->getUnstableResultDir();
    if (getcwd(dir, 1024) != nullptr) {
        device_output_dir_ = std::string(dir) + "/" + save_device_dir_name;
    } else {
        size_t idx = proto_path_.find_last_of("/");
        device_output_dir_ = proto_path_.substr(0, idx + 1) + save_device_dir_name;
    }
    device_output_dir_ = device_output_dir_ + "/" + op_name_;
    size_t idx_l = proto_path_.find_last_of("/"), idx_r = proto_path_.find_last_of(".");
    device_output_dir_ =
        device_output_dir_ + "/" + proto_path_.substr(idx_l + 1, idx_r - idx_l - 1);
    auto getSaveTensorName = [&](std::string name, std::string save_dir, int times) {
        std::string save_name = save_dir + "/" + name + "_times_" + std::to_string(times);
        return save_name;
    };
    for (int index = 0; index < unstable_input.size(); index++) {
        MetaTensor *ts = parser_->input(index);
        if (ts->total_count == 0 || !ts->input_reuse) {
            continue;
        }
        auto save_name = getSaveTensorName(parser_->input(index)->name, device_output_dir_, 0);
        std::remove(save_name.c_str());
    }
    for (int index = 0; index < unstable_output.size(); index++) {
        // std::cout<<"save "<<index<<std::endl;
        auto save_name = getSaveTensorName(parser_->output(index)->name, device_output_dir_, 0);
        std::remove(save_name.c_str());
    }
    std::remove(device_output_dir_.c_str());
}

void Executor::mapDeviceOutTocasepath() {
    std::string device_output_casepath_txt =
        device_output_dir_ + "/teco2case" + std::to_string(global_var.dev_id_) + ".txt";
    // createDir(tecooutput_casepath);
    std::ofstream cur_file(device_output_casepath_txt, std::ios::out | std::ios::app);
    // std::cout<<"size of cur case:"<<" "<< tecooutput_casepath_.size()<<std::endl;
    for (auto &pair : device_output_casepath_) {
        auto &deviceoutput = pair.first;
        auto &casepath = pair.second;
        // std::cout<<tecooutput<<" "<<casepath<<std::endl;
        cur_file << deviceoutput << " : " << casepath << std::endl;
    }
    cur_file.close();
}

void Executor::sync() {}

EvaluateResult Executor::teardown() {
    ALLOG(INFO) << "Copy data from device to host.";
    memcpyDevice2Host();

    // if (!Context::instance()->onlyTestPerformance()) {
    //     ALLOG(VLOG) << "CPU compute.";
    //     cpuCompute();
    //     if (Context::instance()->compareWithGPU() && !is_bf16_) {
    //         ALLOG(VLOG) << "GPU compute.";
    //         // gpuCompute();
    //     }
    // }

    // destroyDesc();
    ALLOG(INFO) << "Free workspace.";
    workspaceFree();

    // get criterions for each op
    getCriterionsUse();  // todo(maliang):
    // diff
    evaluate();

    // free
    ALLOG(INFO) << "Free device memory.";
    destroy();
    deviceFree();
    // std::cout<<"before save"<<std::endl;
    if (Context::instance()->saveDeviceOutput()) {
        // std::cout<<"start save"<<std::endl;
        getProtoPath();
        saveDeviceOutput();
        mapDeviceOutTocasepath();
    }
    if (Context::instance()->onlyTestPerformance() && Context::instance()->notInitInput()) {
    } else {
        hostFree();
    }
    if (!Context::instance()->onlyTestPerformance()) {
        baselineFree();
        if (Context::instance()->compareWithGPU() && !is_bf16_) {
            gpuFree();
        }
    }
    int check_status = scdaFreeCheck(exe_context_->stream);
    if (check_status == 1) {
        ALLOG(ERROR) << "device memory not all free";
    } else if (check_status == 2) {
        ALLOG(ERROR) << "all hbm check, memory out of bounds!!!";
        ADD_FAILURE() << "all hbm check, memory out of bounds!!!";
        eva_res_.status = TECOTEST_STATUS_MEMORY_OUT_ERROR2;
    } else if (check_status == 3) {
        ALLOG(ERROR) << "device memory not all free";
        ALLOG(ERROR) << "all hbm check, memory out of bounds!!!";
        ADD_FAILURE() << "all hbm check, memory out of bounds!!!";
        eva_res_.status = TECOTEST_STATUS_MEMORY_OUT_ERROR2;
    }

    if (!eva_res_.is_passed &&
        eva_res_.status != TECOTEST_STATUS_STABILITY_ERROR) {  // this is only check result
        if (Context::instance()->compareWithGPU()) {
            eva_res_.status = eva_->isPassed();
        } else {
            eva_res_.status = is_bf16_ ? eva_->isPassed_bf16() : eva_->isPassed_cpu();
        }
    }
    if (eva_res_.status != TECOTEST_STATUS_SUCCESS) {
        eva_res_.is_passed = false;
    }

    return eva_res_;
}

EvaluateResult Executor::evaluate() {
# ifdef USE_CUDA
eva_res_.is_passed = true;
    return eva_res_;
#else
    // get error func and threshold
    // it depends on func saved in pb or config
    getCriterion();

    eva_res_.errors.clear();
    if (!Context::instance()->onlyTestPerformance()) {
        ALLOG(VLOG) << "Calculate Diffs.";
        // input reuse
        for (int i = 0; i < parser_->inputs().size(); ++i) {
            MetaTensor *ts = parser_->input(i);
            if (ts->total_count == 0 || !ts->input_reuse) {
                continue;
            }
            bool same_nan_inf = true;
            if (Context::instance()->compareWithGPU()) {
                Error error_teco;
                Error error_gpu;
                same_nan_inf = eva_->resetNanOrInfToZero(
                    host_input[i], baseline_input[i], ts->total_count, ts->dtype, &error_teco);

                if (!same_nan_inf) {
                    ErrorWrap error_wrap;
                    error_wrap.name = ts->name;
                    error_wrap.num = ts->total_count;
                    error_wrap.error_teco = error_teco;
                    error_wrap.error_gpu = error_gpu;
                    eva_->error_vec_.push_back(error_wrap);
                }
            }
            if (same_nan_inf) {
                for (auto it = criterions_.begin(); it != criterions_.end(); ++it) {
                    if (Context::instance()->showAllErrors() || it->enable) {
                        Error error_teco, error_gpu;
                        error_teco = eva_->computeError(baseline_input[i], host_input[i],
                                                        ts->total_count, *it, ts->name, ts->dtype);
                        ErrorWrap error_wrap =
                            ErrorWrap(ts->name, ts->total_count, *it, error_teco, error_gpu);
                        eva_->error_vec_.push_back(error_wrap);
                    }
                }
            }
        }

        for (int i = 0; i < parser_->outputs().size(); ++i) {
            MetaTensor *ts = parser_->output(i);
            if (ts->total_count == 0) {
                continue;
            }

            bool same_nan_inf = true;
            if (Context::instance()->compareWithGPU()) {
                Error error_teco;
                Error error_gpu;
                same_nan_inf =
                    eva_->resetNanOrInfToZero(host_output[i], baseline_output[i],
                                              ts->total_count, ts->dtype, &error_teco);
                if (!same_nan_inf) {
                    ErrorWrap error_wrap;
                    error_wrap.name = ts->name;
                    error_wrap.num = ts->total_count;
                    error_wrap.error_teco = error_teco;
                    error_wrap.error_gpu = error_gpu;
                    eva_->error_vec_.push_back(error_wrap);
                }
            }
            if (same_nan_inf) {
                for (auto it = criterions_.begin(); it != criterions_.end(); ++it) {
                    if (Context::instance()->showAllErrors() || it->enable) {
                        Error error_teco, error_gpu;
                        error_teco = eva_->computeError(baseline_output[i], host_output[i],
                                                        ts->total_count, *it, ts->name, ts->dtype);
                        ErrorWrap error_wrap =
                            ErrorWrap(ts->name, ts->total_count, *it, error_teco, error_gpu);
                        eva_->error_vec_.push_back(error_wrap);
                    }
                }
            }
        }
        eva_res_.errors = eva_->errors();
    }

    ALLOG(VLOG) << "Calculate Performance.";
    getMluPerfInfo(&(eva_res_.device));
    getComparedPerfInfo(&(eva_res_.compared_device), eva_res_.case_path);
    getTensorInfo(&(eva_res_.tensors));

    if (is_bf16_) {
        eva_res_.is_passed = eva_->isPassed_bf16() == TECOTEST_STATUS_SUCCESS;
        eva_res_.what = std::move(eva_->what_bf16());
    // } else if (Context::instance()->compareWithGPU()) {
    //     eva_res_.is_passed = eva_->isPassed() == TECOTEST_STATUS_SUCCESS;
    //     eva_res_.what = std::move(eva_->what());
    } else {
        eva_res_.is_passed = eva_->isPassed_cpu() == TECOTEST_STATUS_SUCCESS;
        eva_res_.what = std::move(eva_->what_cpu());
    }

    // check baseline
    if (exe_config_->perf_baseline) {
        checkBaseline();  // update eva_res_  // todo(maliang):
    }
    return eva_res_;
#endif
}

unsigned int *Executor::calcOutputHash(unsigned int *once_hash) {
    for (size_t i = 0; i < parser_->inputs().size(); ++i) {
        MetaTensor *ts = parser_->input(i);
        if (unlikely(ts->total_count == 0)) {
            ALLOG(WARNING) << "Executor: skip " << ts->name << " calc output hash.";
            continue;
        }
        if (ts->input_reuse) {
            calcHash(exe_context_->stream, dev_input[i], ts->size_in_bytes, once_hash);
            once_hash = once_hash + 1;
        }
    }

    for (int i = 0; i < parser_->outputs().size(); ++i) {
        MetaTensor *ts = parser_->output(i);
        if (unlikely(ts->total_count == 0)) {
            ALLOG(WARNING) << "Executor: skip " << ts->name << " calc output hash.";
            continue;
        }
        calcHash(exe_context_->stream, dev_output[i], ts->size_in_bytes, once_hash);
        once_hash = once_hash + 1;
    }
    return once_hash;
}

void Executor::checkOutputHash(unsigned int *output_hash, int tensor_num, int times) {
    if (tensor_num * times > 0) {
        unsigned int *host_hash = (unsigned int *)malloc(tensor_num * times * sizeof(unsigned int));
        checkScdaErrors(scdaMemcpy(host_hash, output_hash,
                                   tensor_num * times * sizeof(unsigned int), MemcpyDeviceToHost));
        bool failed = false;
        for (int i = 1; i < times; i++) {
            for (int j = 0; j < tensor_num; j++) {
                if (host_hash[i * tensor_num + j] != host_hash[j]) {
                    ALLOG(ERROR) << "test stable failed, hash not same in multiple times";
                    ADD_FAILURE() << "test stable failed, hash not same in multiple times";
                    eva_res_.status = TECOTEST_STATUS_STABILITY_ERROR;
                    failed = true;
                    break;
                }
            }
            if (failed) break;
        }
        if (failed) {
            std::cout << "hash of multiple times: " << std::endl;
            for (int i = 0; i < times; i++) {
                std::cout << i << ": ";
                for (int j = 0; j < tensor_num; j++) {
                    std::cout << host_hash[i * tensor_num + j] << " ";
                }
                std::cout << std::endl;
            }
        }
        free(host_hash);
    } else {
        ALLOG(ERROR) << "test stable failed, hash vector is empty";
        ADD_FAILURE() << "test stable failed, hash vector is empty";
        eva_res_.status = TECOTEST_STATUS_STABILITY_ERROR;
    }
}

void Executor::getCriterion() {
    // only first call will calc it
    if (!criterions_.empty()) {
        std::cout<< "not empty"<< std::endl;
        return;
    }
    criterions_ = parser_->criterions();
    if (criterions_.empty()) {
        ALLOG(INFO) << "Using criterion in config.";

        unsigned out_type;
        if (parser_->outputs().size() > 0) {
            // 以第一个输出类型确定通过的标准
            out_type = parser_->getOutputDataType(0);
        } else {
            // 如果没有output，以第一个输入型确定通过的标准 (理论应该以第一个reused的tensor确定)
            out_type = parser_->getInputDataType(0);
        }

        if (is_bf16_) {
            Criterion criterion;
            criterion.max_error = 0.025;
            criterion.error_threshold = 1e-4;
            criterion.ratio_threshold = 1e-4;
            criterion.formula = MSE_RELA;
            criterion.enable = true;
            criterions_.insert(criterion);
        } else if (out_type == 0) {
            // float
            Criterion criterion;
            criterion.max_error = 1e-6;
            criterion.error_threshold = 1e-6;
            criterion.ratio_threshold = 1e-6;
            criterion.formula = DIFF3_MAX;
            criterion.enable = true;
            criterions_.insert(criterion);
        } else if (out_type == 1) {
            // half
            Criterion criterion;
            criterion.max_error = 1e-3;
            criterion.error_threshold = 1e-3;
            criterion.ratio_threshold = 1e-3;
            criterion.formula = DIFF3_MAX;
            criterion.enable = true;
            criterions_.insert(criterion);
        } else {
            Criterion criterion;
            criterion.max_error = 3e-3;
            criterion.error_threshold = 1e-3;
            criterion.ratio_threshold = 1e-3;
            criterion.formula = DIFF3_MAX;
            criterion.enable = true;
            criterions_.insert(criterion);
            criterion.max_error = 1e-6;
            criterion.formula = DIFF2;
            criterion.enable = true;
            criterions_.insert(criterion);
        }
    }
}

void Executor::getGPUError(MetaTensor *mt, std::string formula, Error *error) {
    auto diffs = mt->gpu_diffs_[formula];
    error->max_error = diffs["max_error"];
    if (formula == "DIFF3_MAX") {
        error->index = diffs["index"];
        error->baseline_value = diffs["baseline_value"];
        error->compare_value = diffs["compare_value"];
    }
}

void Executor::saveGPUErrors(EvaluateResult *er) {
    std::string filename = data_path_ + "cuda_cpu_diff.json";
    if (filename != "") {
        if (Context::instance()->reserveDataFlag()) {
            struct stat buf;
            if (stat(filename.c_str(), &buf) == -1) {
                json info;
                for (int i = 0; i < er->errors.size(); i++) {
                    auto it = er->errors[i];
                    auto name = it.name;
                    auto formula = showFormula(it.criterion.formula);
                    auto gpu_error = it.error_gpu;

                    info[name][formula]["max_error"] = gpu_error.max_error;
                    if (formula == "DIFF3_MAX") {
                        info[name][formula]["index"] = gpu_error.index;
                        info[name][formula]["baseline_value"] = gpu_error.baseline_value;
                        info[name][formula]["compare_value"] = gpu_error.compare_value;
                    }
                }
                createDir(filename);
                std::ofstream(filename) << info;
            }
        } else {
            struct stat buf;
            if (stat(filename.c_str(), &buf) != -1) {
                remove(filename.c_str());
            }
        }
    }
}

void Executor::getMluPerfInfo(PerfInfo *res) {
    // compute
    if (parser_->node()->has_theory_compute_ops()) {
        res->theory_ops = parser_->node()->theory_compute_ops();
    } else {
        res->theory_ops = getTheoryOps();
    }

    // op / ( (latency(us) / 1000 / 1000)  op/s
    res->compute_force = res->theory_ops / res->hardware_time / 1e+6;

    // io
    if (parser_->node()->has_theory_io_size()) {
        res->theory_io = parser_->node()->theory_io_size();
    } else {
        res->theory_io = getTheoryIoSize();
    }

    // io_size(byte) / ( (latency(us)) GB/s
    res->io_bandwidth = res->theory_io / (res->hardware_time * 1000);
}

void Executor::getComparedPerfInfo(ComparedPerfInfo *res, std::string case_path) {
    std::string filename = subReplaceFirst(case_path, ".prototxt", "_cuda.json");
    if (filename != "") {
        struct stat buf;
        if (stat(filename.c_str(), &buf) != -1) {
            std::ifstream in(filename);
            json info;
            in >> info;
            in.close();

            res->interface_time = (double)(info["summary"]["total_device_time"]);  // us
            auto details = info["tracing"];
            for (auto &item : details.items()) {
                std::string key = item.value()["op_name"];
                double value = (double)item.value()["time"];  // us
                res->api_details.push_back(std::pair<std::string, double>(key, value));
            }
            res->device = info["summary"]["device"];

        } else {
            ALLOG(INFO) << "Compared device performance data file not exists";
        }
    }
}

void Executor::getTensorInfo(std::vector<MetaTensor> *res) {
    for (int i = 0; i < parser_->inputs().size(); ++i) {
        MetaTensor *ts = parser_->input(i);
        if (ts->total_count == 0) {
            continue;
        }
        res->push_back(*ts);
    }
    for (int i = 0; i < parser_->outputs().size(); ++i) {
        MetaTensor *ts = parser_->output(i);
        if (ts->total_count == 0) {
            continue;
        }
        res->push_back(*ts);
    }
}

// call this func after getMluHardwareTime()
// deal with baseline check of perf test
void Executor::checkBaseline() {
    GTEST_CHECK(eva_res_.op_name != "", "Executor: missing op name, didn't set it. We need know it "
                                        "when get performance "
                                        "baseline threshold");

    double hw_time_base = 0;
    double hw_time_mean = eva_res_.device.hardware_time;  // eva_->getMluHardwareTime();
    double scale_bound = 0;
    double threshold_absolute = 0;
    double threshold_relative = 0;
    double workspace_size = 0;

    hw_time_base = hw_time_mean;
    eva_res_.is_passed = eva_res_.is_passed;
}

// input_size + output_size
int64_t Executor::getIoSize() {
    size_t total_size = 0;
    for (size_t i = 0; i < parser_->inputs().size(); ++i) {
        MetaTensor *ts = parser_->input(i);
        if (unlikely(ts->empty())) {
            continue;
        }
        total_size += ts->shape_count * ts->sizeof_dtype;
    }
    for (size_t i = 0; i < parser_->outputs().size(); ++i) {
        MetaTensor *ts = parser_->output(i);
        if (unlikely(ts->empty())) {
            continue;
        }
        total_size += ts->shape_count * ts->sizeof_dtype;
    }

    return total_size;
}

// input_size * (1 + reused) + output size
int64_t Executor::getIoSizeWithReused() {
    size_t total_size = 0;
    for (size_t i = 0; i < parser_->inputs().size(); ++i) {
        MetaTensor *ts = parser_->input(i);
        if (unlikely(ts->empty())) {
            continue;
        }
        total_size += ts->shape_count * ts->sizeof_dtype;
        if (ts->input_reuse) {
            total_size += ts->shape_count * ts->sizeof_dtype;
        }
    }
    for (size_t i = 0; i < parser_->outputs().size(); ++i) {
        MetaTensor *ts = parser_->output(i);
        if (unlikely(ts->empty())) {
            continue;
        }
        total_size += ts->shape_count * ts->sizeof_dtype;
    }

    return total_size;
}

// input_size * (1 + reused * (beta!=0)) + output size
int64_t Executor::getIoSizeWithBeta(float beta) {
    size_t total_size = 0;
    for (size_t i = 0; i < parser_->inputs().size(); ++i) {
        MetaTensor *ts = parser_->input(i);
        if (unlikely(ts->empty())) {
            continue;
        }
        total_size += ts->shape_count * ts->sizeof_dtype;
        if (ts->input_reuse && fabs(beta) > 1e-5) {
            total_size += ts->shape_count * ts->sizeof_dtype;
        }
    }
    for (size_t i = 0; i < parser_->outputs().size(); ++i) {
        MetaTensor *ts = parser_->output(i);
        if (unlikely(ts->empty())) {
            continue;
        }
        total_size += ts->shape_count * ts->sizeof_dtype;
    }

    return total_size;
}

int Executor::getOutTensorNum() {
    int num = 0;
    for (size_t i = 0; i < parser_->inputs().size(); ++i) {
        MetaTensor *ts = parser_->input(i);
        if (ts->input_reuse) num += 1;
    }
    for (size_t i = 0; i < parser_->outputs().size(); ++i) {
        MetaTensor *ts = parser_->output(i);
        num += 1;
    }

    return num;
}

void Executor::baselineMalloc() {
    for (size_t i = 0; i < parser_->inputs().size(); ++i) {
        MetaTensor *ts = parser_->input(i);
        if (unlikely(ts->empty())) {
            baseline_input.push_back(nullptr);
            continue;
        }
        if (!ts->input_reuse) {
            baseline_input.push_back(host_input[i]);
        } else {
            baseline_input.push_back(malloc(ts->size_in_bytes));
            memcpy(baseline_input[i], host_input[i], ts->size_in_bytes);
        }
    }
    for (size_t i = 0; i < parser_->outputs().size(); ++i) {
        MetaTensor *ts = parser_->output(i);
        if (unlikely(ts->empty())) {
            baseline_output.push_back(nullptr);
            continue;
        }
        baseline_output.push_back(malloc(ts->size_in_bytes));
        memset(baseline_output[i], 0x04, ts->size_in_bytes);
    }
}

void Executor::baselineFree() noexcept {
    for (size_t i = 0; i < parser_->inputs().size(); ++i) {
        MetaTensor *ts = parser_->input(i);
        if (ts->input_reuse) {
            if (i < baseline_input.size()) {
                if (baseline_input[i] != nullptr) {
                    free(baseline_input[i]);
                    baseline_input[i] = nullptr;
                }
            }
        }
    }
    for (size_t i = 0; i < parser_->outputs().size(); ++i) {
        MetaTensor *ts = parser_->output(i);
        if (i < baseline_output.size()) {
            if (baseline_output[i] != nullptr) {
                free(baseline_output[i]);
                baseline_output[i] = nullptr;
            }
        }
    }
}

void Executor::gpuMalloc() {
    for (size_t i = 0; i < parser_->inputs().size(); ++i) {
        MetaTensor *ts = parser_->input(i);
        if (unlikely(ts->empty())) {
            gpu_input.push_back(nullptr);
            continue;
        }
        if (!ts->input_reuse) {
            gpu_input.push_back(host_input[i]);
        } else {
            gpu_input.push_back(malloc(ts->size_in_bytes));
            memcpy(gpu_input[i], host_input[i], ts->size_in_bytes);
        }
    }
    for (size_t i = 0; i < parser_->outputs().size(); ++i) {
        MetaTensor *ts = parser_->output(i);
        if (unlikely(ts->empty())) {
            gpu_output.push_back(nullptr);
            continue;
        }
        gpu_output.push_back(malloc(ts->size_in_bytes));
        memset(gpu_output[i], 0x04, ts->size_in_bytes);
    }
}

void Executor::gpuFree() noexcept {
    for (size_t i = 0; i < parser_->inputs().size(); ++i) {
        MetaTensor *ts = parser_->input(i);
        if (ts->input_reuse) {
            if (i < gpu_input.size()) {
                if (gpu_input[i] != nullptr) {
                    free(gpu_input[i]);
                    gpu_input[i] = nullptr;
                }
            }
        }
    }
    for (size_t i = 0; i < parser_->outputs().size(); ++i) {
        MetaTensor *ts = parser_->output(i);
        if (i < gpu_output.size()) {
            if (gpu_output[i] != nullptr) {
                free(gpu_output[i]);
                gpu_output[i] = nullptr;
            }
        }
    }
}

void Executor::unstableMalloc() {
    for (size_t i = 0; i < parser_->inputs().size(); ++i) {
        MetaTensor *ts = parser_->input(i);
        if (unlikely(ts->empty())) {
            unstable_input.push_back(nullptr);
            continue;
        }
        if (!ts->input_reuse) {
            unstable_input.push_back(host_input[i]);
        } else {
            unstable_input.push_back(malloc(ts->size_in_bytes));
            memcpy(unstable_input[i], host_input[i], ts->size_in_bytes);
        }
    }
    for (size_t i = 0; i < parser_->outputs().size(); ++i) {
        MetaTensor *ts = parser_->output(i);
        if (unlikely(ts->empty())) {
            unstable_output.push_back(nullptr);
            continue;
        }
        unstable_output.push_back(malloc(ts->size_in_bytes));
        memset(unstable_output[i], 0x04, ts->size_in_bytes);
    }
}

void Executor::unstableFree() noexcept {
    for (size_t i = 0; i < parser_->inputs().size(); ++i) {
        MetaTensor *ts = parser_->input(i);
        if (ts->input_reuse) {
            if (i < unstable_input.size()) {
                if (unstable_input[i] != nullptr) {
                    free(unstable_input[i]);
                    unstable_input[i] = nullptr;
                }
            }
        }
    }
    for (size_t i = 0; i < parser_->outputs().size(); ++i) {
        MetaTensor *ts = parser_->output(i);
        if (i < unstable_output.size()) {
            if (unstable_output[i] != nullptr) {
                free(unstable_output[i]);
                unstable_output[i] = nullptr;
            }
        }
    }
}

void Executor::hostMalloc() {
    hostInputMalloc();
    hostOutputMalloc();
}

void Executor::hostFree() noexcept {
    hostInputFree();
    hostOutputFree();
}

void Executor::hostInputMalloc() {
    for (size_t i = 0; i < parser_->inputs().size(); ++i) {
        MetaTensor *ts = parser_->input(i);
        if (unlikely(ts->empty())) {
            host_input.push_back(nullptr);
            continue;
        }
        host_input.push_back(malloc(ts->size_in_bytes));
        memset(host_input[i], 0x01, ts->size_in_bytes);
    }
}

void Executor::hostInputFree() noexcept {
    for (size_t i = 0; i < parser_->inputs().size(); ++i) {
        if (i < host_input.size()) {
            if (host_input[i] != nullptr) {
                free(host_input[i]);
                host_input[i] = nullptr;
            }
        }
    }
}

void Executor::hostOutputMalloc() {
    for (size_t i = 0; i < parser_->outputs().size(); ++i) {
        MetaTensor *ts = parser_->output(i);
        if (unlikely(ts->empty())) {
            host_output.push_back(nullptr);
            continue;
        }
        host_output.push_back(malloc(ts->size_in_bytes));
        memset(host_output[i], 0x02, ts->size_in_bytes);
    }
}

void Executor::hostOutputFree() noexcept {
    for (size_t i = 0; i < parser_->outputs().size(); ++i) {
        if (i < host_output.size()) {
            if (host_output[i] != nullptr) {
                free(host_output[i]);
                host_output[i] = nullptr;
            }
        }
    }
}

bool Executor::isBetaZero() {
    float beta = -1;
    auto node = parser_->getProtoNode();
    return fabs(beta) < 1e-6;
}

// read file or random
// read file datas include stride
// nethier random
void Executor::initInput() {
    for (size_t i = 0; i < parser_->inputs().size(); ++i) {
        MetaTensor *ts = parser_->input(i);
        if (unlikely(ts->empty())) {
            continue;
        }

        if (ts->inplace >= 0) {
            if (ts->inplace < i) {
                memcpy(host_input[i], host_input[ts->inplace], ts->size_in_bytes);
            } else {
                ALLOG(ERROR) << "Only allow inplace previous tensor";
            }
        } else {
            if (Context::instance()->setNanWithBeta() && ts->input_reuse && isBetaZero()) {
                memset(host_input[i], 0xff, ts->size_in_bytes);
            } else {
                parser_->getInputTensorValue(i, host_input[i], ts->total_count);
            }
        }
    }
}

void Executor::deviceInputMalloc() {
    for (size_t i = 0; i < parser_->inputs().size(); ++i) {
        MetaTensor *ts = parser_->input(i);
        dev_input.push_back(nullptr);
        if (unlikely(ts->empty())) {
            continue;
        }
        if (ts->inplace >= 0) {
            if (ts->inplace < i) {
                dev_input[i] = dev_input[ts->inplace];
            } else {
                ALLOG(ERROR) << "Only allow inplace previous input tensor";
            }
        } else {
            scdaMalloc(&(dev_input[i]), ts->size_in_bytes);
            checkScdaErrors(scdaMemset(dev_input[i], 0xff, ts->size_in_bytes));
        }
    }
}

void Executor::deviceOutputMalloc() {
    for (size_t i = 0; i < parser_->outputs().size(); ++i) {
        MetaTensor *ts = parser_->output(i);
        dev_output.push_back(nullptr);
        if (unlikely(ts->empty())) {
            continue;
        }
        if (ts->inplace >= 0) {
            if (ts->inplace < dev_input.size()) {
                dev_output[i] = dev_input[ts->inplace];
                // } else if (ts->inplace < dev_input.size() + i) {
                //     dev_output[i] = dev_output[ts->inplace - dev_input.size()];
            } else {
                ALLOG(ERROR) << "Only allow inplace previous input tensor";
            }
        } else {
            scdaMalloc(&(dev_output[i]), ts->size_in_bytes);
            checkScdaErrors(scdaMemset(dev_output[i], 0xff, ts->size_in_bytes));
        }
    }
}
void Executor::deviceMalloc() {
    deviceInputMalloc();
    deviceOutputMalloc();
}

void Executor::deviceInputFree() noexcept {
    for (size_t i = 0; i < parser_->inputs().size(); ++i) {
        if (i < dev_input.size()) {
            if (dev_input[i] == nullptr) {
                continue;
            }
            if (parser_->input(i)->inplace >= 0) {
                continue;
            }
            if (!scdaFree(dev_input[i])) {
                std::string name = parser_->input(i)->name;
                ALLOG(ERROR) << "input" << i << " [" + name + "] memory out of bounds!!!\n";
                ADD_FAILURE() << "input" << i << " [" + name + "] memory out of bounds!!!\n";
                eva_res_.status = TECOTEST_STATUS_MEMORY_OUT_ERROR1;
            }
            dev_input[i] = nullptr;
        }
    }
}

void Executor::deviceOutputFree() noexcept {
    for (size_t i = 0; i < parser_->outputs().size(); ++i) {
        if (i < dev_output.size()) {
            if (dev_output[i] == nullptr) {
                continue;
            }
            if (parser_->output(i)->inplace >= 0) {
                continue;
            }
            if (!scdaFree(dev_output[i])) {
                std::string name = parser_->output(i)->name;
                ALLOG(ERROR) << "output" << i << " [" + name + "] memory out of bounds!!!\n";
                ADD_FAILURE() << "output" << i << " [" + name + "] memory out of bounds!!!\n";
                eva_res_.status = TECOTEST_STATUS_MEMORY_OUT_ERROR1;
            }
            dev_output[i] = nullptr;
        }
    }
}

void Executor::deviceFree() noexcept {
    deviceInputFree();
    deviceOutputFree();
}

void Executor::memcpyHost2Device() {
    for (size_t i = 0; i < parser_->inputs().size(); ++i) {
        MetaTensor *ts = parser_->input(i);
        if (unlikely(ts->total_count == 0)) {
            ALLOG(WARNING) << "Executor: skip " << ts->name << " memcpy host => device.";
            continue;
        }
        checkScdaErrors(
            scdaMemcpy(dev_input[i], host_input[i], ts->size_in_bytes, MemcpyHostToDevice));
    }
    for (size_t i = 0; i < parser_->outputs().size(); ++i) {
        MetaTensor *ts = parser_->output(i);
        // when output has stride param
        if (unlikely(ts->total_count == 0)) {
            ALLOG(WARNING) << "Executor: skip " << ts->name << " memcpy host => device.";
            continue;
        }
    }
}

void Executor::memcpyDevice2Host() {
    for (size_t i = 0; i < parser_->inputs().size(); ++i) {
        MetaTensor *ts = parser_->input(i);
        if (unlikely(ts->total_count == 0)) {
            ALLOG(WARNING) << "Executor: skip " << ts->name << " memcpy device => host.";
            continue;
        }
        if (ts->input_reuse) {
            checkScdaErrors(
                scdaMemcpy(host_input[i], dev_input[i], ts->size_in_bytes, MemcpyDeviceToHost));
        }
    }

    for (int i = 0; i < parser_->outputs().size(); ++i) {
        MetaTensor *ts = parser_->output(i);
        // memcpy only for output
        if (unlikely(ts->total_count == 0)) {
            ALLOG(WARNING) << "Executor: skip " << ts->name << " memcpy device => host.";
            continue;
        }
        // memcpy dev to host
        checkScdaErrors(
            scdaMemcpy(host_output[i], dev_output[i], ts->size_in_bytes, MemcpyDeviceToHost));
    }
}

bool Executor::hasInputData() {
    if (parser_->getProtoNode()->input_size() > 0) {
        if (parser_->getProtoNode()->input(0).has_prev_path() ||
            parser_->getProtoNode()->input(0).has_prev_value()) {
            return true;
        }
    }
    return false;
}

void Executor::extraDataMallocAndInit(void **dev_ptr, size_t count, testpt::DataType dtype,
                                      testpt::RandomData *random_param) {
    size_t size = getSizeOfDataType(dtype) * count;
    void *host_ptr = malloc(size);
    setValue(host_ptr, count, dtype, random_param);

    scdaMalloc(dev_ptr, size);
    checkScdaErrors(scdaMemcpy(*dev_ptr, host_ptr, size, MemcpyHostToDevice));
    extra_dev_input.push_back(*dev_ptr);
    free(host_ptr);
}

void Executor::extraDataFree() {
    for (int i = 0; i < extra_dev_input.size(); ++i) {
        if (extra_dev_input[i] != nullptr) {
            scdaFree(extra_dev_input[i]);
            extra_dev_input[i] = nullptr;
        }
    }
    extra_dev_input.clear();
}

void stride_map(void *dst,                           // dst ptr
                void *src,                           // src ptr
                const std::vector<int> &shape,       // shape
                const std::vector<int> &dst_stride,  // stride
                const std::vector<int> &src_stride,  // stride
                size_t dst_offset, size_t src_offset, size_t d, size_t sizeof_dtype,
                const size_t dst_max, const size_t src_max) {
    if (d == shape.size() - 1) {  // the last dim
        for (size_t i = 0; i < shape[d]; ++i) {
            size_t dst_idx = src_offset + i * src_stride[d];
            size_t src_idx = dst_offset + i * dst_stride[d];
            memcpy((char *)dst + dst_idx * sizeof_dtype, (char *)src + src_idx * sizeof_dtype,
                   sizeof_dtype);
        }
    } else {
        for (size_t i = 0; i < shape[d]; ++i) {
            stride_map(dst, src, shape, dst_stride, src_stride, dst_offset + i * dst_stride[d],
                       src_offset + i * src_stride[d], d + 1, sizeof_dtype, dst_max, src_max);
        }
    }
}

// src(strided) -> dst(shape)
// dst should malloc by shape_count
// src should malloc by stride_count
void Executor::remove_stride(void *dst, void *src, const std::vector<int> &shape,
                             const std::vector<int> &dst_stride,  // dst_stride
                             size_t sizeof_dtype) {
    GTEST_CHECK(shape.size() == dst_stride.size(),
                "Executor: shape's size is not equal to stride's size.");

    size_t shape_total =
        std::accumulate(shape.begin(), shape.end(), (int)1, std::multiplies<int>());
    size_t stride_total = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        stride_total += (shape[i] - 1) * dst_stride[i];
    }

    std::vector<int> src_stride(shape.size());
    int stride_base = 1;
    for (ssize_t i = shape.size() - 1; i >= 0; --i) {
        src_stride[i] = stride_base;
        stride_base *= shape[i];
    }
    stride_map(dst, src, shape, dst_stride, src_stride, 0, 0, 0, sizeof_dtype, stride_total,
               shape_total);
}

// src(shape) -> dst(strided)
// dst should malloc by stride_count
// src should malloc by shape_count
void Executor::include_stride(void *dst, void *src, const std::vector<int> &shape,
                              const std::vector<int> &src_stride,  // src_stride
                              size_t sizeof_dtype) {
    GTEST_CHECK(shape.size() == src_stride.size(),
                "Executor: shape's size is not equal to stride's size.");

    size_t shape_total =
        std::accumulate(shape.begin(), shape.end(), (int)1, std::multiplies<int>());

    size_t stride_total = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        stride_total += (shape[i] - 1) * src_stride[i];
    }

    std::vector<int> dst_stride(shape.size());
    int stride_base = 1;

    for (int i = shape.size() - 1; i >= 0; --i) {
        dst_stride[i] = stride_base;
        stride_base *= shape[i];
    }

    stride_map(dst, src, shape, dst_stride, src_stride, 0, 0, 0, sizeof_dtype, shape_total,
               stride_total);
}

void Executor::gpuCompute() {
    for (int i = 0; i < parser_->inputs().size(); ++i) {
        MetaTensor *ts = parser_->input(i);
        if (ts->total_count == 0 || !ts->input_reuse) {
            continue;
        }
        memcpy(gpu_input[i], baseline_input[i], ts->size_in_bytes);
    }

    for (int i = 0; i < parser_->outputs().size(); ++i) {
        MetaTensor *ts = parser_->output(i);
        if (ts->total_count == 0) {
            continue;
        }
        memcpy(gpu_output[i], baseline_output[i], ts->size_in_bytes);
    }
}

// run python cmd
void Executor::getProtoPath() {
    char dir[1024] = "";
    // eva_res_.case_path= ../zoo/tecoal/add_tensor/test_case/case_0.prototxt
    realpath(eva_res_.case_path.c_str(), dir);
    proto_path_ = dir;
}

void Executor::getPythonPath() {
    std::string current_dir;
    char *buffer = NULL;
    if ((buffer = getcwd(NULL, 0)) == NULL) {
        ALLOG(ERROR) << "Get current dir failed.";
        throw std::invalid_argument(std::string(__FILE__) + " +" + std::to_string(__LINE__));
    } else {
        current_dir = std::string(buffer) + "/";
        free(buffer);
    }

    const char *env = std::getenv("TECOAL_TECOTEST_PROJECT");
    if (env == nullptr) {
        python_path_ = current_dir + "../zoo/" + al_type_ + "/" + op_name_;
    } else {
        python_path_ = std::string(env) + "/zoo/" + al_type_ + "/" + op_name_;
    }
}

void Executor::getDataPath() {
    char dir[1024] = "";
    realpath(eva_res_.case_path.c_str(), dir);
    std::string case_path = dir;
    case_path = case_path.substr(0, case_path.find_last_of("."));
    data_path_ = case_path + "/";
}

void Executor::getDataParams() {
    input_params_.clear();
    reuse_params_.clear();
    output_params_.clear();
    for (int index = 0; index < inputs_.size(); index++) {
        input_params_.push_back(data_path_ + "input_" + std::to_string(index));
    }

    int i = 0;
    for (int index = 0; index < inputs_.size(); index++) {
        if (parser_->input(index)->input_reuse) {
            reuse_params_.push_back(data_path_ + device_ + "_reuse_" + std::to_string(i));
            i++;
        }
    }

    for (int index = 0; index < outputs_.size(); index++) {
        output_params_.push_back(data_path_ + device_ + "_output_" + std::to_string(index));
    }
}

void Executor::inputToFile() {
    for (int index = 0; index < inputs_.size(); index++) {
        if (parser_->input(index)->dtype != 12) {
            saveDataToFile(input_params_[index], inputs_[index],
                           parser_->input(index)->size_in_bytes);
        } else {
            uint16_t *input_16 = (uint16_t *)inputs_[index];
            float *input_32 = (float *)malloc(parser_->input(index)->size_in_bytes * 2);
            for (size_t i = 0; i < parser_->input(index)->size_in_bytes / 2; i++) {
                input_32[i] = BFloat16::bfloat162float(input_16[i]);
            }
            saveDataToFile(input_params_[index], (void *)input_32,
                           parser_->input(index)->size_in_bytes * 2);
            free(input_32);
        }
    }
}

void Executor::outputFromFile() {
    int i = 0;
    for (int index = 0; index < inputs_.size(); index++) {
        if (parser_->input(index)->input_reuse) {
            if (!parser_->input(index)->empty()) {
                if (parser_->input(index)->dtype != 12) {
                    readDataFromFile(reuse_params_[i], inputs_[index],
                                     parser_->input(index)->size_in_bytes);
                } else {
                    float *output_32 = (float *)malloc(parser_->input(index)->size_in_bytes * 2);
                    readDataFromFile(reuse_params_[i], output_32,
                                     parser_->input(index)->size_in_bytes * 2);
                    uint16_t *tmp = (uint16_t *)malloc(parser_->input(index)->size_in_bytes);
                    for (int j = 0; j < parser_->input(index)->total_count; j++) {
                        tmp[j] = BFloat16::float2bfloat16Rn(output_32[j]);
                    }
                    memcpy(inputs_[index], tmp, parser_->input(index)->size_in_bytes);
                    free(output_32);
                    free(tmp);
                }
            }
            i++;
        }
    }
    for (int index = 0; index < outputs_.size(); index++) {
        if (!parser_->output(index)->empty()) {
            if (parser_->output(index)->dtype != 12) {
                readDataFromFile(output_params_[index], outputs_[index],
                                 parser_->output(index)->size_in_bytes);
            } else {
                float *output_32 = (float *)malloc(parser_->output(index)->size_in_bytes * 2);
                readDataFromFile(output_params_[index], output_32,
                                 parser_->output(index)->size_in_bytes * 2);
                uint16_t *tmp = (uint16_t *)malloc(parser_->output(index)->size_in_bytes);
                for (int j = 0; j < parser_->output(index)->total_count; j++) {
                    tmp[j] = BFloat16::float2bfloat16Rn(output_32[j]);
                }
                memcpy(outputs_[index], tmp, parser_->output(index)->size_in_bytes);
                free(output_32);
                free(tmp);
            }
        }
    }
}

void Executor::removeCaseData() {
    removeDir(data_path_);
    // for (int index = 0; index < reuse_params_.size(); index++) {
    //     remove(reuse_params_[index].c_str());
    // }
    // for (int index = 0; index < output_params_.size(); index++) {
    //     remove(output_params_[index].c_str());
    // }
    // std::string filename = subReplaceFirst(eva_res_.case_path, ".prototxt", "/cuda_cpu_diff.json");
    // struct stat buf;
    // if (stat(filename.c_str(), &buf) != -1) {
    //     remove(filename.c_str());
    // }
}

bool Executor::isDataExists() {
    struct stat buffer;
    for (int index = 0; index < reuse_params_.size(); index++) {
        if (stat(reuse_params_[index].c_str(), &buffer) != 0) return false;
    }
    for (int index = 0; index < output_params_.size(); index++) {
        if (stat(output_params_[index].c_str(), &buffer) != 0) return false;
    }
    return true;
}

void Executor::callPythonCmd() {
    // write params to json file
    std::string params_list_file = data_path_ + device_ + "_list";
    createDir(params_list_file);
    std::ofstream fout(params_list_file, std::ios::out | std::ios::binary);
    if (!fout) {
        ALLOG(ERROR) << params_list_file << " open fault, and can not create";
        throw std::invalid_argument(std::string(__FILE__) + " +" + std::to_string(__LINE__));
    }
    fout << "{\n";

    fout << "\"param_path\":\"" << proto_path_ << "\",\n";

    fout << "\"input_lists\":[\n";
    for (int i = 0; i < (int)input_params_.size() - 1; i++)
        fout << "\"" << input_params_[i] << "\",\n";
    if (input_params_.size() > 0) fout << "\"" << input_params_[input_params_.size() - 1] << "\"\n";
    fout << "],\n";

    fout << "\"reuse_lists\":[\n";
    for (int i = 0; i < (int)reuse_params_.size() - 1; i++)
        fout << "\"" << reuse_params_[i] << "\",\n";
    if (reuse_params_.size() > 0) fout << "\"" << reuse_params_[reuse_params_.size() - 1] << "\"\n";
    fout << "],\n";

    fout << "\"output_lists\":[\n";
    for (int i = 0; i < (int)output_params_.size() - 1; i++)
        fout << "\"" << output_params_[i] << "\",\n";
    if (output_params_.size() > 0)
        fout << "\"" << output_params_[output_params_.size() - 1] << "\"\n";
    fout << "]\n";

    fout << "}\n";

    fout.close();

    // run cmd
    std::string cmd =
        "cd " + python_path_ + "/; python " + module_ + ".py " + params_list_file + " " + device_;
    ALLOG(INFO) << cmd;
    system(cmd.c_str());
}

void Executor::pythonCompute() {
    if (device_ != "cpu" && device_ != "cuda") {
        ALLOG(ERROR) << "Don't support this device, now only support cpu,cuda.";
        throw std::invalid_argument(std::string(__FILE__) + " +" + std::to_string(__LINE__));
    }

    if (module_ == "") module_ = op_name_;
    if (func_ == "") func_ = "test_" + op_name_;
    getProtoPath();
    getPythonPath();
    getDataPath();
    getDataParams();

    size_t name_length = proto_path_.find(".prototxt");
    std::string key_path = proto_path_.substr(0, name_length) + "/" + device_ + ".key";
    std::ifstream key_file(key_path);
    std::ifstream proto_file(proto_path_);
    bool is_data_exists = isDataExists();
    if (is_data_exists) {
        if (key_file.is_open() && isProtoChanged(proto_path_, key_path)) {
            pt_changed_ = true;
            ALLOG(VLOG) << device_ + "proto data has changed, regenerated";
            removeCaseData();
            inputToFile();
            callPythonCmd();
            key_file.close();
            std::ofstream key_file(key_path, std::ios::binary);
            key_file << proto_file.rdbuf();
        } else {
            ALLOG(VLOG) << device_ + " data already exists, no need to run compute.";
            if (!key_file.is_open()) {
                key_file.close();
                std::ofstream key_file(key_path, std::ios::binary);
                key_file << proto_file.rdbuf();
            }
        }

    } else {
        inputToFile();
        callPythonCmd();
        key_file.close();
        std::ofstream key_file(key_path, std::ios::binary);
        key_file << proto_file.rdbuf();
    }
    key_file.close();
    proto_file.close();
    outputFromFile();

    // remove input data
    for (int index = 0; index < input_params_.size(); index++) {
        remove(input_params_[index].c_str());
    }

    if (!Context::instance()->reserveDataFlag() && !is_data_exists) {
        removeCaseData();
    } else {
        ALLOG(VLOG) << device_ + " data saved at " << data_path_;
    }

    /* delete gpu output data later
    if(device_ == "cuda"){
        for (int index = 0; index < reuse_params_.size(); index++) {
            remove(reuse_params_[index].c_str());
        }
        for (int index = 0; index < output_params_.size(); index++) {
            remove(output_params_[index].c_str());
        }
    }
    */
}

void Executor::pythonComputeCPU(std::string device) {
    device_ = device;
    inputs_ = baseline_input;
    outputs_ = baseline_output;
    pythonCompute();
}

void Executor::pythonComputeGPU(std::string device) {
    device_ = device;
    inputs_ = gpu_input;
    outputs_ = gpu_output;
    pythonCompute();
}

// for old, will be removed later
void Executor::pythonCompute(std::string module, std::string func, bool use_cmd) {
    module_ = module;
    func_ = func;
    pythonComputeCPU("cpu");
}

void Executor::pythonComputeGPU(std::string module, std::string func, bool use_cmd) {
    module_ = module;
    func_ = func;
    pythonComputeGPU("cuda");
}

}  // namespace optest
