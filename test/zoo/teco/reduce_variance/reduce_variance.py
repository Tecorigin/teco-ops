# encoding:utf-8
'''
Create on
@author:
Describe:
'''

import os
import sys
import json
import torch
import numpy as np
sys.path.append("../zoo/teco/")
sys.path.append("../")
from executor import *

def check_inputs(param_path, input_lists, reuse_lists, output_lists):
    # TODO
    if param_path == "":
        print("The path of prototxt file is empty.")
        return False
    if len(input_lists) != 1:
        print("The number of input data is wrong.")
        return False
    if len(reuse_lists) != 0:
        print("The number of reuse data is wrong.")
        return False
    if len(output_lists) != 1:
        print("The number of output data is wrong.")
        return False
    return True

def test_reduce_variance(param_path, input_lists, reuse_lists, output_lists, device):
    if not check_inputs(param_path, input_lists, reuse_lists, output_lists):
        return

    if device == "cuda":
        if torch.cuda.is_available():
            used_device = torch.device("cuda:0")
        else:
            print("not found cuda device")
            return

    params = read_prototxt(param_path)
    input_params = params["input"]
    output_params = params["output"]
    reduce_variance_params = params["tecokernel_param"]["reduce_variance_param"]
    axis = int(reduce_variance_params["axis"])
    correction = int(reduce_variance_params["correction"])
    x_dtype = input_params["dtype"]
    y_dtype = output_params["dtype"]

    input_params = params["input"]
    x = to_tensor(input_lists[0], input_params, device=device)
    output = torch.var(x, dim=axis, correction=correction)

    with open(output_lists[0], "wb") as f:
        save_tensor(f, output, y_dtype)

def parse_params(filename):
    with open(filename, "r") as f:
        params = json.load(f)
    return params

if __name__ == "__main__":
    params = parse_params(sys.argv[1])
    device = sys.argv[2]
    test_reduce_variance(params["param_path"], params["input_lists"], params["reuse_lists"], params["output_lists"], device)