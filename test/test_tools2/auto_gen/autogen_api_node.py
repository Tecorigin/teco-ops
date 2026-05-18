import os, shutil, re

code = """\
import sys
import os
import numpy as np
import json
from google.protobuf import json_format

sys.path.append(
    os.path.split(os.path.realpath(__file__))[0] + '/../../test_proto')
import optest_pb2 as optest

sys.path.append(os.path.split(os.path.realpath(__file__))[0])
sys.path.append(os.path.split(os.path.realpath(__file__))[0] + '/../')
from tecoalNode import *

class Tecoal{}Node(TecoalNode):
    op_name = "{}"

    def check(self):
        return True

    def reRand(self):
        if self.node is None:
            print("node is None")
            return

        for i in range(len(self.node.input)):
            randData(self.node.input[i])
 
    def getParamInfo(self):
        if self.node.HasField('tecoal_param'):
            return 'tecoal_param {{{{\\n{{}}}}}}'.format(self.node.tecoal_param)
"""

def find_prototxt_path(path, prototxt_list):
    dir_or_files = os.listdir(path)
    for dir_file_path in dir_or_files:
        new_path= os.path.join(path, dir_file_path)
        if os.path.isdir(new_path):
            find_prototxt_path(new_path,prototxt_list)
        else:
            if new_path.endswith(".prototxt"):
                prototxt_list.append(new_path)

def gen_template(op_name):
    class_name = ""
    for tmp in op_name.split("_"):
        class_name += tmp.capitalize()

    dst_file = "../nodes/todo/{}_node.py".format(op_name)
    with open(dst_file, "w") as f:
        f.write(code.format(class_name, op_name))

def gen_template_prototxt(api_path, op_name):
    dst_file = "../nodes/template/{}.prototxt".format(op_name)
    prototxts = []
    find_prototxt_path(api_path, prototxts)
    if len(prototxts) == 0:
        print(f"[ERROR] {op_name} prototxt not found in {api_path}")
    else:
        prototxts.sort()
        src_file = prototxts[0]
        shutil.copyfile(src_file, dst_file)


        # delete test_param in prototxt
        with open(dst_file, 'r') as f:
            lines = f.readlines()
        
        start, end = 0, 0
        for i, line in enumerate(lines):
            if "test_param" in line:
                start = i
            if start != 0 and end == 0 and "}" in line:
                end = i
                break
        lines = lines[0:start] + lines[end+1:]
        
        if start !=0 and end != 0:
            with open(dst_file, 'w') as f:
                f.write("".join(lines))



def getApiList():
    filepath = os.path.split(os.path.realpath(__file__))[0]
    result = os.popen('find {}/../../zoo/tecoal/ -mindepth 1 -maxdepth 1 -type d ! -name "__pycache__"'.format(filepath))
    tmp = result.read().strip().split("\n")
    result.close()
    tecoal_api_list = tmp


    return  tecoal_api_list 

api_path_list = getApiList()
api_list = [api[api.rfind("/") + 1 :] for api in api_path_list]

os.system("mkdir -p ../nodes/todo")
for op_name in api_list:
    gen_template(op_name)

for i in range(len(api_path_list)):
    gen_template_prototxt(api_path_list[i], api_list[i])
