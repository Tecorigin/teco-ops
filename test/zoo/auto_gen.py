import os
import argparse
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--al_type', dest='al_type', type=str, help='dnn, blas, custom or customblas.')
parser.add_argument('--op_name', dest='op_name', type=str, help='if al_type is custom or customblas, op_name should start with custom_.')
args = parser.parse_args()

def upper_camel(op_name):
    res = ""
    slices = op_name.split('_')
    for item in slices:
        res += item.capitalize()
    return res

def gen_proto(al_type, op_name):
    al_type_convert = {"tecoal": "tecoal"}
    package_convert = {"tecoal": "testpt"}
    op_file = os.path.join("test_proto", al_type_convert[al_type], f"{op_name}.proto")
    
    op_name = op_name.replace("custom_", "")
    header = f"""syntax = "proto2";

package {package_convert[al_type]};
import "{al_type_convert[al_type]}/common.proto";
    """

    op_message = f"""
message {upper_camel(op_name)}Param{{

}}
    """

    with open(op_file, 'w') as f:
        f.write(header)
        f.write(op_message)


    return op_file


def gen_op_path(al_type, op_name):
    op_path = os.path.join("zoo", al_type, op_name)
    if not os.path.exists(op_path):
        os.makedirs(op_path)

def gen_header_file(al_type, op_name):
    header_str = f"""#ifndef ZOO_{al_type.upper()}_{op_name.upper()}_{op_name.upper()}_H_  // NOLINT
#define ZOO_{al_type.upper()}_{op_name.upper()}_{op_name.upper()}_H_

#include "zoo/{al_type}/{al_type}_executor.h"

namespace optest {{

class {upper_camel(op_name)}Executor : public {al_type.capitalize()}Executor {{
 public:
    void paramCheck();
    void paramParse();
    void paramGeneration();
    void compute();
    void cpuCompute();
    void gpuCompute();
    int64_t getTheoryOps() override;
    int64_t getTheoryIoSize() override;
    // void destroy();  // for destroy Descriptor

 private:
    // dnn/custom example
    // tecoalActivationDescriptor_t activationDesc_;
    // tecoalActivationMode_t mode_;
    // float alpha_;
    // tecoalTensorDescriptor_t xDesc_;
    // void *x_;
    // float beta_;
    // tecoalTensorDescriptor_t yDesc_;
    // void *y_;
    // double coef_;
    // tecoalNanPropagation_t nanopt_;

    // blas/customblas example
    //  int m_;
    //  int n_;
    //  int k_;
    //  float alpha_;
    //  float beta_;
    //  tblasDataType_t dtype_;
    //  tblasOperation_t transa_;
    //  tblasOperation_t transb_;
    //  int lda_;
    //  int ldb_;
    //  int ldc_;
    //  void *a_;
    //  void *b_;
    //  void *c_;
}};

}}  // namespace optest

#endif  // ZOO_{al_type.upper()}_{op_name.upper()}_{op_name.upper()}_H_  // NOLINT
"""

    op_file = os.path.join("zoo", al_type, op_name, f"{op_name}.h")
    with open(op_file, 'w') as f:
        f.write(header_str)

    return op_file

def gen_cpp_file(al_type, op_name):
    executor = f"{upper_camel(op_name)}Executor"

    cpp_str = f"""#include "zoo/{al_type}/{op_name}/{op_name}.h"
#include <stdio.h>
#include <iostream>
#include <string>
#include "zoo/{al_type}/convert.h"
namespace optest {{

// void {executor}::destroy() {{
    // checktecoal(tecoalDestroyActivationDescriptor(activationDesc_));
// }}

void {executor}::paramCheck() {{
    // todo
    if (parser_->inputs().size() != 2) {{
        ALLOG(ERROR) << "input num is wrong.";
        throw std::invalid_argument(std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }}

    if (parser_->outputs().size() != 0) {{
        ALLOG(ERROR) << "output num is wrong.";
        throw std::invalid_argument(std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }}
}}

void {executor}::paramParse() {{
    // todo
    // dnn activation_forward example
    // auto activation_forward_param = parser_->getProtoNode()->dnn_param().activation_forward_param();
    // auto activation_desc_param =
    //     parser_->getProtoNode()->dnn_param().activation_forward_param().act_desc_param();
    // alpha_ = activation_forward_param.alpha();
    // beta_ = activation_forward_param.beta();
    // mode_ = convert::toDnnActivationMode(activation_desc_param.mode());
    // nanopt_ = convert::toDnnNanPropagation(activation_desc_param.relu_nanopt());
    // coef_ = activation_desc_param.coef();
}}

void {executor}::paramGeneration() {{
    // todo
    // dnn activation_forward example
    // xDesc_ = getInputDesc<tecoalTensorDescriptor_t>(0);
    // x_ = dev_input[0];
    // yDesc_ = getInputDesc<tecoalTensorDescriptor_t>(1);
    // y_ = dev_input[1];

    // checktecoal(tecoalCreateActivationDescriptor(&activationDesc_));
    // checktecoal(tecoalSetActivationDescriptor(activationDesc_, mode_, nanopt_, coef_));
}}

void {executor}::compute() {{
    // todo
    // dnn activation_forward example
    // checktecoal(tecoalActivationForward(handle_, activationDesc_, &alpha_, xDesc_, x_, &beta_,
    //                                       yDesc_, y_));
}}

int64_t {executor}::getTheoryOps() {{
    // todo
}}

int64_t {executor}::getTheoryIoSize() {{
    // todo
 }}

void {executor}::cpuCompute() {{ pythonComputeCPU("cpu"); }}

void {executor}::gpuCompute() {{ pythonComputeGPU("cuda"); }}

}}  // namespace optest
    """

    op_file = os.path.join("zoo", al_type, op_name, f"{op_name}.cpp")
    with open(op_file, 'w') as f:
        f.write(cpp_str)

    return op_file

def gen_python_file(al_type, op_name):
    now_time = datetime.datetime.now().strftime("%Y-%m-%d")
    username = os.getlogin()
    py_str = f"""# encoding:utf-8
'''
Create on {now_time}
@author: {username}
Describe: {op_name} in cpu/cuda
'''

import os
import sys
import json
import torch
import numpy as np
sys.path.append("../zoo/{al_type}/")
sys.path.append("../")
from {al_type}_executor import *

def check_inputs(param_path, input_lists, reuse_lists, output_lists):
    # TODO
    if param_path == "":
        print("The path of prototxt file is empty.")
        return False
    if len(input_lists) != 2:
        print("The number of input data is wrong.")
        return False
    if len(reuse_lists) != 1:
        print("The number of reuse data is wrong.")
        return False
    if len(output_lists) != 0:
        print("The number of output data is wrong.")
        return False
    return True


def test_{op_name}(param_path, input_lists, reuse_lists, output_lists, device):
    if not check_inputs(param_path, input_lists, reuse_lists, output_lists):
        return

    params = read_prototxt(param_path)
    # todo
    # this is example of dnn add_tensor
    # add_tensor_params = params["dnn_param"]["add_tensor_param"]
    # alpha = add_tensor_params["alpha"]
    # beta = add_tensor_params["beta"]

    # input_params = params["input"]
    # x = to_tensor(input_lists[0], input_params[0], device=device)
    # y = alpha * x
    # if beta != 0:
    #     y_in = to_tensor(input_lists[1], input_params[1], device=device)
    #     y = y + beta * y_in

    # y_dtype = input_params[1]["dtype"]
    # with open(reuse_lists[0], "wb") as f:
    #     save_tensor(f, y, y_dtype)

def parse_params(filename):
    with open(filename, "r") as f:
        params = json.load(f)
    return params

if __name__ == "__main__":
    params = parse_params(sys.argv[1])
    device = sys.argv[2]
    test_{op_name}(params["param_path"], params["input_lists"], params["reuse_lists"], params["output_lists"], device)
    """

    op_file = os.path.join("zoo", al_type, op_name, f"{op_name}.py")
    with open(op_file, 'w') as f:
        f.write(py_str)

    return op_file

if __name__ == "__main__":
    if args.al_type is None or args.op_name is None:
        print("please use --help")
        exit(-1)

    al_type = args.al_type
    op_name = args.op_name
    if al_type.startswith("custom"):
        if not op_name.startswith("custom_"):
            op_name = "custom_" + op_name
            print(f"[WARNING] op_name is {op_name} when using custom/customblas")
    files = []
    files.append(gen_proto(al_type, op_name))
    gen_op_path(al_type, op_name)
    files.append(gen_header_file(al_type, op_name))
    files.append(gen_cpp_file(al_type, op_name))
    files.append(gen_python_file(al_type, op_name))

    print("The following files need to be compiled:")
    for f in files:
        print(f"    {f}")
