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
 
#include "device/sdaa/version.h"
#include <stdlib.h>
#include <sdaa_runtime.h>
#include <sdpti.h>
#include <unistd.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <cstring>
#include "common/variable.h"
#include "common/tecotest.h"
#include "zoo/tecozoo.h"
#include "device/sdaa/common.h"

extern optest::GlobalVar global_var;

namespace optest {
namespace sdaa {

std::string formatVersion(int version) {
    int major = version / 100000000;
    int minor = (version % 100000000) / 1000000;
    int patch = (version % 1000000) / 10000;
    int type = (version % 10000) / 100;
    int type_no = version % 100;
    std::string out_version =
        std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
    if (type == 0) {
        return out_version;
    } else if (type == 1) {
        out_version += "rc" + std::to_string(type_no);
    } else if (type == 2) {
        out_version += "b" + std::to_string(type_no);
    } else if (type == 3) {
        out_version += "a" + std::to_string(type_no);
    }
    return out_version;
}

void printVersions() {
    int dev_num = 0;
    checkScdaErrors(scdaGetDeviceCount(&dev_num));
    if (global_var.dev_id_ < 0 || global_var.dev_id_ >= dev_num) {
        ALLOG(ERROR) << "device id " << global_var.dev_id_ << " must >= 0 and < " << dev_num;
        throw std::invalid_argument(std::string(__FILE__) + " +" + std::to_string(__LINE__));
    }
    if (global_var.kernel_id_ < 0 || global_var.kernel_id_ >= dev_num) {
        ALLOG(ERROR) << "kernel id " << global_var.kernel_id_ << " must >= 0 and < " << dev_num;
        throw std::invalid_argument(std::string(__FILE__) + " +" + std::to_string(__LINE__));
    }

    int device = global_var.kernel_id_;
    sdaaDeviceProp_t props;
    checkScdaErrors(sdaaGetDeviceProperties(&props, device));

    int driver_version = 0, runtime_version = 0;
    uint32_t sdpti_version = 0;
    checkScdaErrors(sdaaDriverGetVersion(&driver_version));
    checkScdaErrors(sdaaRuntimeGetVersion(&runtime_version));
#ifndef SDPTI_VERSION_MAJOR
    sdptiGetVersion(&sdpti_version);
#endif

    int dnn_version = 0, blas_version = 0, custom_version = 0;
#if DNN_ENABLE
    dnn_version = tecoalGetVersion();
#endif

#if BLAS_ENABLE
    tblasHandle_t handle;
    tecoblasCreate(&handle);
    tecoblasGetVersion(handle, &blas_version);
    tecoblasDestroy(handle);
#endif

// todo because of tecocustom bug
#if CUSTOM_ENABLE
    // custom_version = tecoalExtGetVersion();
#endif

    int tecotest_version = tecotestGetVersion();

    // get .so path
    std::string libtecdnn_path = "", libtecoblas_path = "", libtecocustom_path = "";
    std::string libsdaart_path = "", libsdpti_path = "";

    std::string current_dir = "./";
    const char *env = std::getenv("TECOAL_TECOTEST_PROJECT");
    if (env != nullptr) {
        current_dir = std::string(env) + "/build/";
    }
    std::string cmd = "cd " + current_dir + "; ldd demo";

    FILE *fp = NULL;
    char buffer[512];
    if ((fp = popen(cmd.c_str(), "r")) != NULL) {
        while (fgets(buffer, sizeof(buffer), fp)) {
            std::string res = std::string(buffer);
            if (res.find("tecoal.so => ") != std::string::npos) {
                size_t index = res.find("=>") + 3;
                libtecdnn_path = res.substr(index);
                replace(libtecdnn_path.begin(), libtecdnn_path.end(), '\n', ' ');
            } else if (res.find("tecoblas.so => ") != std::string::npos) {
                size_t index = res.find("=>") + 3;
                libtecoblas_path = res.substr(index);
                replace(libtecoblas_path.begin(), libtecoblas_path.end(), '\n', ' ');
            } else if (res.find("tecoal_ext.so => ") != std::string::npos) {
                size_t index = res.find("=>") + 3;
                libtecocustom_path = res.substr(index);
                replace(libtecocustom_path.begin(), libtecocustom_path.end(), '\n', ' ');
            } else if (res.find("libsdaart.so.0 => ") != std::string::npos ||
                       res.find("libsdaart.so => ") != std::string::npos) {
                size_t index = res.find("=>") + 3;
                libsdaart_path = res.substr(index);
                replace(libsdaart_path.begin(), libsdaart_path.end(), '\n', ' ');
            } else if (res.find("libsdpti.so.0 => ") != std::string::npos ||
                       res.find("libsdpti.so => ") != std::string::npos) {
                size_t index = res.find("=>") + 3;
                libsdpti_path = res.substr(index);
                replace(libsdpti_path.begin(), libsdpti_path.end(), '\n', ' ');
            }
        }
        pclose(fp);
    }

    std::cout << "----------------------------" << std::endl;
    std::cout << "device          : " << props.name << std::endl;
    std::cout << "spe clock (MHz) : " << props.clockRate << std::endl;
    std::cout << "sdaadriver      : " << std::setw(10) << formatVersion(driver_version)
              << std::endl;
    std::cout << "sdaart          : " << std::setw(10) << formatVersion(runtime_version) << " "
              << libsdaart_path << std::endl;
    std::cout << "sdpti           : " << std::setw(10) << formatVersion(sdpti_version) << " "
              << libsdpti_path << std::endl;
    std::cout << "tecoal         : " << std::setw(10) << formatVersion(dnn_version) << " "
              << libtecdnn_path << std::endl;
    std::cout << "tecoblas        : " << std::setw(10) << formatVersion(blas_version) << " "
              << libtecoblas_path << std::endl;
    std::cout << "tecocustom      : " << std::setw(10) << formatVersion(custom_version) << " "
              << libtecocustom_path << std::endl;
    std::cout << "tecotest        : " << std::setw(10) << formatVersion(tecotest_version)
              << std::endl;
    std::cout << "----------------------------" << std::endl;
}

}  // namespace sdaa
}  // namespace optest
