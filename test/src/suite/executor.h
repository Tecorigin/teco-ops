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

#ifndef SUITE_EXECUTOR_H_  // NOLINT
#define SUITE_EXECUTOR_H_

#include <time.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <functional>
#include <sstream>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <memory>
#include <unordered_set>
#include <unordered_map>
#include <set>
#include "gtest/gtest.h"
#include "common/variable.h"
#include "common/context.h"
#include "common/tecoallog.h"
#include "case/parser.h"
#include "case/pb_tools.h"
#include "case/criterion.h"
#include "case/evaluator.h"
#include "device/device.h"
#include "teco_kernel_context.h"

extern optest::GlobalVar global_var;

namespace optest {

const int interface_time_repeat = 4;

// io bandwidth GB/s
const float IO_BANDWIDTH = 240;

// runtime config
struct ExecuteConfig {
    ExecuteConfig() {
        dump_data = getEnv("TEST_DUMP_DATA", false);
        fixed_criterion = getEnv("TEST_ALL_CRITERION", false);
        perf_baseline = getEnv("TEST_PERF_BASELINE", false);
    }

    void print() {
        std::cout << "Execution config:\n";
        std::cout << std::left << std::setw(25) << "show diff1~3: " << fixed_criterion << std::endl;
        std::cout << std::left << std::setw(25) << "dump data: " << dump_data << std::endl;
        std::cout << std::left << std::setw(25) << "check perf baseline: " << perf_baseline
                  << std::endl;
    }

    bool fixed_criterion = false;
    bool dump_data = true;
    bool perf_baseline = false;
};

// common variable.
// like handle/queue/tensors ...
// create outside (executor) and just create once.
// all case share these variable.
struct ExecuteContext {
    ~ExecuteContext() { destroy(); }

    void init() {
        checkScdaErrors(scdaSetDevice(global_var.kernel_id_));
        checkScdaErrors(scdaStreamCreate(&stream));
#ifdef USE_TECO
        teco_kernel::TecoHandle& handle = teco_kernel::TecoHandle::getInstance();
        handle.stream = stream;
#endif
#ifdef USE_CUDA
        cuda_kernel::CudaKernelHandle& handle = cuda_kernel::CudaKernelHandle::getInstance();
        handle.stream = stream;
#endif
        checkScdaErrors(scdaSetDevice(global_var.dev_id_));
    }

    void destroy() {
        checkScdaErrors(scdaSetDevice(global_var.kernel_id_));
        if (stream != nullptr) {
            checkScdaErrors(scdaStreamDestroy(stream));
        } else {
            ALLOG(WARNING) << "stream is null";
        }
        checkScdaErrors(scdaSetDevice(global_var.dev_id_));
        stream = nullptr;
    }
    void reset() {
        ALLOG(WARNING) << "Executor: reset sdaart and go on running, this may caused "
                          "by sdaart failed.";
        destroy();
        init();
    }

    scdaStream_t stream{nullptr};
};

class Executor {
 public:
    Executor() { criterions_.clear(); }
    virtual ~Executor();

    void init(const std::shared_ptr<ExecuteContext> ctx);  // set config param by
                                                           // init().
    // set execute variable by setup().
    void setup(std::string file, const std::shared_ptr<ExecuteConfig> ecfg);
    void warmUp();
    void getPerformance();
    void testStability();
    void launch();
    void runBaseline();
    void getBaseline();
    bool ready();
    void sync();
    EvaluateResult teardown();
    inline EvaluateResult *result() { return &eva_res_; }

    //  protected:
    bool is_bf16_;
    bool pt_changed_;
    std::string op_name_;
    std::string al_type_;
    std::string case_name_;

    std::shared_ptr<Parser> parser_ = nullptr;
    std::shared_ptr<Evaluator> eva_ = nullptr;
    std::shared_ptr<ExecuteContext> exe_context_ = nullptr;
    std::shared_ptr<ExecuteConfig> exe_config_ = nullptr;

    // allocate by device_runtime

    std::vector<void *> workspace_;
    std::vector<void *> dev_input;
    std::vector<void *> dev_output;

    std::vector<void *> host_input;
    std::vector<void *> host_output;

    std::vector<void *> baseline_input;
    std::vector<void *> baseline_output;

    std::vector<void *> gpu_input;
    std::vector<void *> gpu_output;

    std::vector<void *> extra_dev_input;

    std::vector<void *> unstable_input;
    std::vector<void *> unstable_output;

    // placeholder used to identify whether the criterion is used or not
    std::vector<int> criterions_use_ = {1, 1, 1, 1};
    virtual void setContext() {}
    virtual void paramCheck() {}  // check op params
    virtual void paramParse() {}
    virtual void paramGeneration() {}
    virtual void workspaceMalloc() {}
    virtual void workspaceFree() {}
    virtual void cpuCompute() = 0;
    virtual void gpuCompute();
    virtual void compute() = 0;
    virtual void forward() {}
    virtual void setQuantizedParam() {}
    virtual void diffPreprocess() {}
    virtual int64_t getTheoryOps() { return 0; }
    virtual int64_t getTheoryIoSize() { return 0; }
    virtual std::vector<int> getCriterionsUse() { return criterions_use_; }

    virtual void createDesc() {}
    virtual void destroyDesc() {}

    virtual void destroy() {}
    // void syncQueueAndGetHardwareTime(int repeat = 1);
    // private:
 protected:
    void baselineMalloc();
    void baselineFree() noexcept;
    void initInput();  // read or random data
    bool isBetaZero();

    void hostInputMalloc();
    void hostInputFree() noexcept;
    void hostOutputMalloc();
    void hostOutputFree() noexcept;
    void hostMalloc();
    void hostFree() noexcept;

    void unstableMalloc();
    void unstableFree() noexcept;
    void deviceInputMalloc();
    void deviceInputFree() noexcept;

    void deviceOutputMalloc();
    void deviceOutputFree() noexcept;
    void deviceMalloc();
    void deviceFree() noexcept;

    void gpuMalloc();
    void gpuFree() noexcept;

    void extraDataMallocAndInit(void **dev_ptr, size_t count, testpt::DataType dtype,
                                testpt::RandomData *random_param);
    void extraDataFree();

    bool hasInputData();

    void saveResultHash();

    int64_t getIoSize();
    int64_t getIoSizeWithReused();
    int64_t getIoSizeWithBeta(float beta);

    int getOutTensorNum();

    // switch data for perf test
    void switchDataToOrigin();
    void switchDataToPerf();
    // memcpy
    void memcpyHost2Device();
    void memcpyDevice2Host();

    void checkBaseline();
    EvaluateResult evaluate();

    // efficiency
    double getComputeForce();
    double getIoBandwidth();
    void getMluPerfInfo(PerfInfo *info);
    void getComparedPerfInfo(ComparedPerfInfo *res, std::string case_path);
    void getTensorInfo(std::vector<MetaTensor> *info);

    // chose criterion by op_name
    std::set<Criterion> criterions_;
    void getCriterion();
    void getGPUError(MetaTensor *mt, std::string formula, Error *error);
    void saveGPUErrors(EvaluateResult *er);
    void getDeviceOutputPath(std::string name, int index);
    void saveUnstableResult(int times);
    void removeStableResult();
    unsigned int *calcOutputHash(unsigned int *once_hash);
    void checkOutputHash(unsigned int *output_hash, int tensor_num, int times);

    EvaluateResult eva_res_;

 protected:
    void remove_stride(void *dst, void *src, const std::vector<int> &shape,
                       const std::vector<int> &dst_stride, size_t sizeof_dtype);
    void include_stride(void *dst, void *src, const std::vector<int> &shape,
                        const std::vector<int> &src_stride, size_t sizeof_dtype);

 private:
    // this is for python generate data in run time
    // todo(maliang)
    std::string device_;
    std::vector<void *> inputs_;
    std::vector<void *> outputs_;
    std::string module_ = "";
    std::string func_ = "";
    std::string python_path_;
    std::string proto_path_;
    std::string data_path_;
    std::string device_output_path_;
    std::string device_output_dir_;
    std::vector<std::string> input_params_;
    std::vector<std::string> reuse_params_;
    std::vector<std::string> output_params_;
    std::unordered_map<std::string, std::string> device_output_casepath_;

    void getDataPath();
    void getProtoPath();
    void getPythonPath();

    void getDataParams();
    void inputToFile();
    void saveDeviceOutput();
    void outputFromFile();
    void mapDeviceOutTocasepath();
    bool isDataExists();

    void removeCaseData();

    void callPythonCmd();
    // void callPython();
    void pythonCompute();

 public:
    void pythonComputeCPU(std::string device);
    void pythonComputeGPU(std::string device);
    // for old, will be removed later
    void pythonCompute(std::string module, std::string func, bool use_cmd);
    void pythonComputeGPU(std::string module, std::string func, bool use_cmd);
};

}  // namespace optest

#endif  // SUITE_EXECUTOR_H_  // NOLINT
