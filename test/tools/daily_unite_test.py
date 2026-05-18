import time
import os
import json
import multiprocessing
from multiprocessing import Process, Manager

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gid', dest='gid', type=str, help='gid')
parser.add_argument('--op_name', dest='op_name', type=str, help='op_name')
parser.add_argument('--cases_list', dest='cases_list', type=str, help='cases_list')
parser.add_argument('--cases_dir', dest='cases_dir', type=str, help='cases_dir')
parser.add_argument('--warm_repeat', dest='warm_repeat', type=int, default=1, help='warm_repeat')
parser.add_argument('--perf_repeat', dest='perf_repeat', type=int, default=1, help='perf_repeat')
parser.add_argument("--test_stable", action="store_true", help="whether to test stablity")
parser.add_argument("--test_multi_core", action="store_true", help="whether to test multi core")
args = parser.parse_args()

cases_dir = "/data/hpe_data/testcase/daily_test"

def get_ops_cases(path):
    cases = []
    if not os.path.exists(path):
        print("cases list file {} not found".format(path))
        return cases
    if os.path.isdir(path):
        print("CASES LIST MUST BE A FILE")
        return cases
    if os.path.isfile(path):
        with open(path, 'r') as f:
            lines = f.readlines()
        if len(lines) > 0:
            cases = [x.strip() for x in lines]
        return cases
    
def find_prototxt_path(path,prototxt_list):
    dir_or_files = os.listdir(path)
    for dir_file_path in dir_or_files:
        new_path= os.path.join(path, dir_file_path)
        if os.path.isdir(new_path):
            find_prototxt_path(new_path,prototxt_list)
        else:
            if new_path.endswith(".prototxt"):
                prototxt_list.append(new_path)

def get_cases_for_dir(path):
    cases = []
    if not os.path.exists(path):
        print("cases dir file {} not found".format(path))
        return cases
    if os.path.isfile(path):
        print("CASES DIR MUST BE A DIR")
        return cases
    if os.path.isdir(path):
        find_prototxt_path(path,cases)
        return cases
    
def split_ops_list(origin_list,num):
    if len(origin_list) % num == 0:
        cnt = len(origin_list) // num
    else:
        cnt = len(origin_list) // num + 1

    for i in range(0,num):
        yield origin_list[i*cnt:(i+1)*cnt]

def get_api_list():
    result = os.popen('ls ../zoo/tecoal/')
    tmp = result.read()
    result.close()
    tmp = tmp.split('\n')
    del tmp[-1]
    blas_api_list = tmp
    
    result = os.popen('ls ../zoo/blas/')
    tmp = result.read()
    result.close()
    tmp = tmp.split('\n')
    del tmp[-1]
    dnn_api_list = tmp

    return  blas_api_list + dnn_api_list

def run_cmd(gid,op_names,cases_list, cases_dir):
    print("====start time is", time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time())))
    test_name = ";".join(i for i in op_names)
    test_name = "\"" + test_name + "\""
    cmd = "cd ../build; python ../tools/unit_test.py --gid {} --test_name {}".format(gid, test_name)
    if cases_list is not None:
        cmd += " --cases_list {}".format(cases_list)
    elif cases_dir is not None:
        cmd += " --cases_dir {}".format(cases_dir)
    cmd += " --warm_repeat {} --perf_repeat {}".format(args.warm_repeat, args.perf_repeat)
    if args.test_stable:
        cmd += " --test_stable"
    print(cmd)
    os.system(cmd)



def daily_test():  
    print(args)

    gid_list = args.gid.split(',')

    if args.cases_list is not None:
        cases = get_ops_cases(args.cases_list)
    elif args.cases_dir is not None:
        cases = get_cases_for_dir(args.cases_dir)
    with open("cases_list.txt", 'w') as f:
        for case in cases:
            f.write(case)
            f.write("\n")


    if len(cases) == 0:
        print("no cases found")
        exit(1)
    print("case num is:", len(cases))

    api_list = get_api_list()
    if args.op_name is not None:
        op_names = args.op_name.split(",")
    else:
        op_names = api_list

    op_cases = {}
    for api in api_list:
        # pass if not needed
        if api not in op_names:
            continue
        if api not in op_cases:
            op_cases[api] = []
        for case in cases:
            if "/"+api+"/" in case:
                op_cases[api].append(case)
        # delete api if case num is 0
        if len(op_cases[api]) == 0:
            del op_cases[api]

    op_names = list(op_cases.keys())

    # case_op_num_dict = {}
    # for item in cases:
    #     words = item.split("/")
    #     for word_index in range(len(words)):
    #         if words[word_index] == "dnn" or words[word_index] == "blas":
    #             opname = words[word_index+2]
    #             if opname not in case_op_num_dict:
    #                 case_op_num_dict[opname] = 1
    #             else:
    #                 case_op_num_dict[opname] += 1

    # op_names = list(case_op_num_dict.keys())

    all_num = 0
    for key, value in op_cases.items():
        all_num += len(value)

    print("case num is:", all_num)
    print(op_names)
    

    # if args.op_name is not None:
    #     op_names = args.op_name.split(",")

    processes = []
    split_opcase = [item for item in split_ops_list(op_names,len(gid_list))]

    for i in range(len(gid_list)):
        gid_num = gid_list[i]
        cut_opcase = split_opcase[i]
        if len(cut_opcase) == 0:
            continue
        son_p = Process(target = run_cmd, args=(gid_num,cut_opcase,args.cases_list, args.cases_dir))
        son_p.start()
        processes.append(son_p)
    for son_p in processes:
        son_p.join()

if __name__ == "__main__":
    daily_test()