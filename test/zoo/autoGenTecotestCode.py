#!/usr/bin/python
# encoding:utf-8
'''
Create on 2023-01-09
@author: sundaoheng
Describe: Auto generate code for tecotest
'''
import os
import string
import re

input_path = "../../../../dnn/ops"

def GenPath():
    print("-------- Generate path start --------")
    for root, dirs, files in os.walk(input_path, topdown=False):
        for path in dirs:
            if "conv" in path or "multi" in path or "rnn" in path or "fuse" in path:
                continue
            for file_name in os.listdir(input_path + "/"  + path):
                if "tablefor_activation_forward" in file_name:
                    continue
                file_name = file_name.split('.')[0]
                if file_name == "concat_tensor":       # maliang
                    file_name = "concat"
                if not os.path.exists("./todo/" + file_name):
                    print("zoo/tecoal/" + file_name)
                    os.makedirs("./todo/" + file_name + "/test_case")
    print("-------- Generate path end --------")

def GetTecoalParam():
    tecoal_file = "../test_proto/tecoal.proto"
    param_dist_list = {}
    with open(tecoal_file, 'r') as f:
        lines = f.readlines()
        mark = 0
        for line in lines:
            if "}" in line:
                mark = 0
            if mark is 1:
                if len(line.strip().split()) >= 3:
                    class_name = line.strip().split()[1]
                    param_name = line.strip().split()[2]
                    param_dist_list[class_name] = param_name
            if "message TecoalParam" in line:
                mark = 1
    return param_dist_list
def GetArithParam(param_name):
    tecoal_file = "../test_proto/tecoal.proto"
    param_list = []
    param_mark = "message " + param_name
    with open(tecoal_file, 'r') as f:
        lines = f.readlines()
        mark = 0
        for line in lines:
            if "}" in line:
                mark = 0
            if mark == 1:
                if "handle" in line:
                    continue
                if len(line.strip().split()) >= 3:
                    param_list.append(line.strip().split()[2])
            if param_mark in line:
                mark = 1
    return param_list

def GenHeadFile():
    print("-------- Generate head file start --------")
    output_path = "./todo"
    for root, dirs, files in os.walk(output_path, topdown=False):
        if "test_case" in dirs:
            continue
        for path in dirs:
            file_name = input_path + "/" + path + "/" + path + ".cpp"
            func_name = ""
            params = []
            if os.path.exists(file_name):
                with open(file_name, 'r') as f:
                    lines = f.read().replace('\n', ' ').replace('    ', '').replace('( ', '(').replace(') ', ')')
                    lines = re.findall(r'tecoalStatus_t tecoalWINAPI tecoal(.+?){', lines)
                    for line in lines:
                        if ("Get" not in line) and ("Set" not in line) and ("Create" not in line) and ("Destroy" not in line):
                            #func = line[0]
                            params = line.split('(')[1].split(',')
            #print(func_name)
            #print(params)
            macro_name = "TECOTEST_ZOO_DNN_" + path.upper() + "_" + path.upper() + "_H_"
            class_name = ""
            for substr in path.split('_'):
                #print(substr)
                class_name += substr.capitalize()
            class_name += "Executor"
            #print(class_name)
            head_file = "./todo/" + path + "/" + path + ".h"
            if not os.path.exists(head_file):
                with open(head_file, 'a+') as fd:
                    fd.write("#ifndef " + macro_name + "\n")
                    fd.write("#define " + macro_name + "\n\n")
                    fd.write("#include \"dnn_executor.h\"\n")
                    fd.write("#include \"tecoal.h\"\n\n")
                    fd.write("namespace optest {\n\n")
                    fd.write("class " + class_name + " : public Executor {\n")
                    fd.write("public:\n")
                    fd.write("    " + class_name + "() {}\n")
                    fd.write("    ~" + class_name + "() {}\n\n")
                    fd.write("    void paramCheck();\n")
                    fd.write("    void paramParse();\n")
                    fd.write("    void paramGeneration();\n")
                    fd.write("    void compute();\n")
                    fd.write("    void cpuCompute();\n")
                    fd.write("    void gpuCompute();\n")
                    fd.write("    int64_t getTheoryOps() override;\n\n")
                    fd.write("private:\n")
                    for param in params:
                        print(param)
                        param = param.strip().strip(')')
                        if "handle" in param:
                            continue
                        if "*alpha" in param or "*beta" in param:
                            param = "float " + param.split()[-1].strip('*')
                        else:
                            param = param.split()[-2] + " " + param.split()[-1]
                        if "[]" in param:
                            fd.write("    " + param[0:-2] + "_[4];\n")
                        else:
                            fd.write("    " + param + "_;\n")
                    fd.write("};\n")
                    fd.write("}; // namespace optest\n\n")
                    fd.write("#endif  // " + macro_name + "\n")
    print("-------- Generate head file end --------")

def GenCppFile():
    print("-------- Generate CPP file start --------")
    output_path = "./todo"
    param_dist_list = GetTecoalParam()

    for root, dirs, files in os.walk(output_path, topdown=False):
        if "test_case" in dirs:
            continue
        for path in dirs:
            if path == "concat":
                file_name = input_path + "/" + path + "/" + "concat_tensor" + ".cpp"
                # print("filename:",file_name)
            else:
                file_name = input_path + "/" + path + "/" + path + ".cpp"
            func_name = ""
            params = []
            if os.path.exists(file_name):
                with open(file_name, 'r') as f:
                    lines = f.read().replace('\n', ' ').replace('    ', '').replace('( ', '(').replace(') ', ')')
                    lines = re.findall(r'tecoalStatus_t tecoalWINAPI tecoal(.+?){', lines)
                    for line in lines:
                        if ("Get" not in line) and ("Set" not in line) and ("Create" not in line) and ("Destroy" not in line) or ("SetTensor(" in line):
                            func_name = "tecoal" + line.split('(')[0]
                            params = line.split('(')[1].split(',')
            #print(func_name)
            #print(params)
            class_name = ""
            key_name = ""
            for substr in path.split('_'):
                #print(substr)
                class_name += substr.capitalize()
                if substr != "forward" and substr != "backward":
                    key_name += substr.capitalize()
            key_name += "Param"
            if key_name not in param_dist_list:
                continue;
            dnn_param = param_dist_list[key_name]
            class_name += "Executor"
            cpp_file = "./todo/" + path + "/" + path + ".cpp"
            if not os.path.exists(cpp_file):
                #print(cpp_file)
                with open(cpp_file, 'a+') as fd:
                    fd.write("#include <stdio.h>\n")
                    fd.write("#include <iostream>\n")
                    fd.write("#include \""+ path +".h\"\n")
                    fd.write("#include \"tecoal.h\"\n")
                    fd.write("#include \"convert.h\"\n")
                    fd.write("namespace optest {\n\n")
                    fd.write("void " + class_name + "::paramCheck() {\n")
                    fd.write("    //TODO\n")
                    fd.write("    if (parser_->inputs().size() != 0) {\n")
                    fd.write("        ALLOG(ERROR) << \"input num is wrong.\";\n")
                    fd.write("        throw std::invalid_argument(std::string(__FILE__) + \":\" + std::to_string(__LINE__));\n")
                    fd.write("    }\n\n")
                    fd.write("    if (parser_->outputs().size() != 0) {\n")
                    fd.write("        ALLOG(ERROR) << \"output num is wrong.\";\n")
                    fd.write("        throw std::invalid_argument(std::string(__FILE__) + \":\" + std::to_string(__LINE__));\n")
                    fd.write("    }\n\n")
                    fd.write("}\n\n")
                    fd.write("void " + class_name + "::paramParse() {\n")
                    fd.write("    auto " + path + "_param = parser_->getProtoNode()->dnn_param()." + dnn_param + "();\n")
                    param_list = GetArithParam(key_name)
                    for param in param_list:
                        fd.write("    " + param + "_ = " + path + "_param." + param + "();\n")
                    fd.write("}\n\n")
                    fd.write("void " + class_name + "::paramGeneration() {\n")
                    index = 0
                    sign = 0
                    for param in params:
                        param = param.strip().strip(')')
                        if sign == 1:
                            param = param.split()[-1].strip('*') + "_"
                            fd.write("    " + param + " = dev_input[" + str(index) + "];\n")
                            sign = 0
                            index = index + 1
                        if "Desc" in param:
                            paramDesc = param.split()[-1].strip('*') + "_"
                            fd.write("    " + paramDesc + " = getInputDesc<tecoalTensorDescriptor_t>(" + str(index) + ");\n")
                            sign = 1
                    fd.write("}\n\n")
                    fd.write("void " + class_name + "::compute() {\n")
                    param_list = ""
                    for param in params:
                        param = param.strip().strip(')')
                        if "*alpha" in param or "*beta" in param:
                            param = "&" + param.split()[-1].strip('*')
                        else:
                            param = param.split()[-1].strip("*")
                        param_list = param_list + param + "_, "
                    param_list = param_list.strip().strip(",")
                    fd.write("    checktecoal(" + func_name + "(" + param_list + "));\n")
                    fd.write("}\n\n")
                    fd.write("int64_t " + class_name + "::getTheoryOps() {\n")
                    fd.write("    int64_t theory_ops = parser_->input(0)->shape_count;\n")
                    fd.write("    return theory_ops;\n")
                    fd.write("}\n\n")
                    fd.write("void " + class_name + "::cpuCompute() {\n")
                    fd.write("    pythonComputeCPU(\"cpu\");\n")
                    fd.write("}\n\n")
                    fd.write("void " + class_name + "::gpuCompute() {\n")
                    fd.write("    pythonComputeGPU(\"cuda\");\n")
                    fd.write("}\n\n")
                    fd.write("}  // namespace optest\n")

    print("-------- Generate CPP file end --------")

def GenPythonFile():
    print("-------- Generate Python file start --------")
    output_path = "./todo"
    param_dist_list = GetTecoalParam()

    for root, dirs, files in os.walk(output_path, topdown=False):
        if "test_case" in dirs:
            continue
        for path in dirs:
            python_file = "./todo/" + path + "/" + path + ".py"
            class_name = ""
            key_name = ""
            for substr in path.split('_'):
                #print(substr)
                class_name += substr.capitalize()
                if substr != "forward" and substr != "backward":
                    key_name += substr.capitalize()
            key_name += "Param"
            if key_name not in param_dist_list:
                continue;
            dnn_param = param_dist_list[key_name]
            print(dnn_param)
            param_list = GetArithParam(key_name)
            print(param_list)
            if not os.path.exists(python_file):
                with open(python_file, 'a+') as fd:
                    fd.write("# encoding:utf-8\n")
                    fd.write("'''\n")
                    fd.write("Create on 2023-01-09\n")
                    fd.write("@author: \n")
                    fd.write("Describe: \n")
                    fd.write("'''\n")
                    fd.write("\n")

                    fd.write("import os\n")
                    fd.write("import sys\n")
                    fd.write("import json\n")
                    fd.write("import torch\n")
                    fd.write("import numpy as np\n")
                    fd.write("sys.path.append(\"../zoo/tecoal/\")\n")
                    fd.write("sys.path.append(\"../\")\n")
                    fd.write("from dnn_executor import *\n")
                    fd.write("\n")

                    fd.write("def check_inputs(param_path, input_lists, reuse_lists, output_lists):\n")
                    fd.write("    # TODO\n")
                    fd.write("    if param_path == \"\":\n")
                    fd.write("        print(\"The path of prototxt file is empty.\")\n")
                    fd.write("        return False\n")
                    fd.write("    if len(input_lists) != 0:\n")
                    fd.write("        print(\"The number of input data is wrong.\")\n")
                    fd.write("        return False\n")
                    fd.write("    if len(reuse_lists) != 0:\n")
                    fd.write("        print(\"The number of reuse data is wrong.\")\n")
                    fd.write("        return False\n")
                    fd.write("    if len(output_lists) != 0:\n")
                    fd.write("        print(\"The number of output data is wrong.\")\n")
                    fd.write("        return False\n")
                    fd.write("    return True\n")
                    fd.write("\n")               
    
                    fd.write("def test_" + path + "(param_path, input_lists, reuse_lists, output_lists, device):\n")
                    fd.write("    if not check_inputs(param_path, input_lists, reuse_lists, output_lists):\n")
                    fd.write("        return\n")
                    fd.write("\n")
                    fd.write("    params = read_prototxt(param_path)\n")
                    fd.write("    " + path + "_params = params[\"dnn_param\"][\"" + dnn_param + "\"]\n")
                    for param in param_list:
                        fd.write("    " + param + " = " + path + "_params[\"" + param + "\"]\n")
                    fd.write("\n")
                    fd.write("    # TODO this is AddTensor demo\n")
                    fd.write("    input_params = params[\"input\"]\n")
                    fd.write("    x = to_tensor(input_lists[0], input_params[0], device=device)\n")
                    fd.write("    y = alpha * x\n")
                    fd.write("    if beta != 0:\n")
                    fd.write("        y_in = to_tensor(input_lists[1], input_params[1], device=device)\n")
                    fd.write("        y += beta * y_in\n")
                    fd.write("\n")
                    fd.write("    y_dtype = input_params[1][\"dtype\"]\n")
                    fd.write("    with open(reuse_lists[0], \"wb\") as f:\n")
                    fd.write("        save_tensor(f, y, y_dtype)\n")
                    fd.write("\n")

                    fd.write("def parse_params(filename):\n")
                    fd.write("    with open(filename, \"r\") as f:\n")
                    fd.write("        params = json.load(f)\n")
                    fd.write("    return params\n")
                    fd.write("\n")

                    fd.write("if __name__ == \"__main__\":\n")
                    fd.write("    params = parse_params(sys.argv[1])\n")
                    fd.write("    device = sys.argv[2]\n")
                    fd.write("    test_" + path + "(params[\"param_path\"], params[\"input_lists\"], params[\"reuse_lists\"], params[\"output_lists\"], device)\n")
                    fd.write("\n")
                    
    
    
       
    print("-------- Generate Python file end --------")

if __name__ == "__main__":
    print("-------- Autogen start --------")
    GenPath()
    GenHeadFile()
    GenCppFile()
    GenPythonFile()
    # #GetArithParam("AddTensorParam")
    # print("-------- Autogen end --------")