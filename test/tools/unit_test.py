#!/usr/bin/python3

import time
import os
import subprocess
import psutil
import pdb
import argparse
import random
import pandas as pd
import json
import multiprocessing
from multiprocessing import Process, Manager


def deal_log(log_name):
    result = {}
    tmp = os.popen('grep "Interface Time" {}'.format(log_name))
    a = tmp.read().split('\n')
    tmp.close()
    del a[-1]
    if (len(a) >= 1):
        result['interface_time(ms)'] = a[0].split(":")[-1].rstrip(' (ms)')
    else:
        result['interface_time(ms)'] = 'error'

    tmp = os.popen('grep "Hardware Time" {}'.format(log_name))
    a = tmp.read().split('\n')
    tmp.close()
    del a[-1]
    if (len(a) >= 1):
        result['hardware_time(ms)'] = a[0].split(":")[-1].rstrip(' (ms)')
    else:
        result['hardware_time(ms)'] = 'error'

    tmp = os.popen('grep "Launch Time" {}'.format(log_name))
    a = tmp.read().split('\n')
    tmp.close()
    del a[-1]
    if (len(a) == 1):
        result['launch_time(ms)'] = a[0].split(":")[-1].rstrip(' (ms)')
    else:
        result['launch_time(ms)'] = 'error'

    tmp = os.popen('grep "TheoryIOs" {}'.format(log_name))
    a = tmp.read().split('\n')
    tmp.close()
    del a[-1]
    if (len(a) == 1):
        result['theory_ios(Bytes)'] = a[0].split(":")[-1].rstrip(' (Bytes)')
    else:
        result['theory_ios(Bytes)'] = 'error'

    tmp = os.popen('grep "IoBandWidth" {}'.format(log_name))
    a = tmp.read().split('\n')
    tmp.close()
    del a[-1]
    if (len(a) == 1):
        result['io_bandwidth(GB/s)'] = a[0].split(":")[-1].rstrip(' (GB/s)')
    else:
        result['io_bandwidth(GB/s)'] = 'error'

    tmp = os.popen('grep -r "kernel_name = " {}'.format(log_name))
    a = tmp.read()
    tmp.close()
    result['kernel_details'] = a

    b = a.split("\n")
    result['main_kernel'] = ""
    for kernel in b:
        if "convolution" in kernel:
            # op_name = memset : kernel_name = teco_slave_memset_4B_align : time = 63.0 us
            result['main_kernel'] = kernel.split(":")[1].strip().split(
                "=")[1].strip()
            break

    tmp = os.popen('grep "\[  FAILED  \]" {}'.format(log_name))
    fail_log = tmp.read().split('\n')
    tmp.close()
    del fail_log[-1]

    tmp = os.popen('grep "\[  PASSED  \]" {}'.format(log_name))
    pass_log = tmp.read().split('\n')
    tmp.close()
    del pass_log[-1]

    if (len(fail_log) >= 1 or len(pass_log) < 1):
        result['result'] = "FAILED"
    else:
        result['result'] = "PASSED"

    tmp = os.popen('grep -A8 -B2 "DIFF3_MAX" {}'.format(log_name))
    a = tmp.read()
    tmp.close()
    result['DIFF3_MAX'] = a

    tmp = os.popen('grep -A5 "DIFF3_MEAN" {}'.format(log_name))
    a = tmp.read()
    tmp.close()
    result['DIFF3_MEAN'] = a

    return result


# time(second)
time_out_threshold = 300


def cmd_time_guard(cmd, log_name):
    # tempfile_name = './log/' +  str(abs(hash(cmd))) + '.log'
    tempfile_name = log_name
    retry_time = 0
    while retry_time < 10:
        time_out_flag = False
        f = open(tempfile_name, 'w')
        process = subprocess.Popen('exec {}'.format(cmd),
                                   shell=True,
                                   stdout=f,
                                   stderr=f)
        start = time.time()
        while (process.poll() is None):
            end = time.time()
            if (end - start >= time_out_threshold):
                time_out_flag = True
                process.kill()
                break
        f.close()

        if os.path.exists(tempfile_name) and os.path.getsize(
                tempfile_name) != 0:
            tmp = os.popen('grep "device id must" {}'.format(tempfile_name))
            a = tmp.read().split('\n')
            tmp.close()
            del a[-1]
            if (len(a) < 1):
                break
        retry_time += 1

    # save log
    result = {}
    result['cmd'] = cmd
    result['user_time'] = end - start
    result['time_out_flag'] = str(time_out_flag)
    result.update(deal_log(tempfile_name))

    # os.system('rm -f {}'.format(tempfile_name))

    return result


def grep_prototxt(dir):
    result = os.popen('find {} -name "*prototxt"'.format(dir))
    tmp = result.read()
    result.close()
    tmp = tmp.split('\n')
    del tmp[-1]
    return tmp


def add_case_path(cases_list, test_name, cur_list):
    for i in range(len(cur_list)):
        tmp = {'test_name': test_name, 'case_path': cur_list[i]}
        cases_list.append(tmp)


def gen_cases_list(cases_list, dir, test_name):
    if not os.path.isdir(dir):
        print("gen_cases_list: not dir")
        return

    dirlist = os.listdir(dir)
    for dirret in dirlist:
        fullname = dir + "/" + dirret

        if (dirret == test_name):
            add_case_path(cases_list, test_name, grep_prototxt(fullname))
            continue

        if os.path.isdir(fullname):
            gen_cases_list(cases_list, fullname, test_name)
        else:
            pass


def get_test_name_in_path(case_path, api_list):
    for tmp in os.path.abspath(case_path).split('/'):
        if tmp in api_list:
            return tmp
    return None


def run_case(i, gid, test_name, case_path, queue_result=None):
    cur_result = {}
    cmd = './demo'
    cmd += ' --gid=' + str(gid)
    cmd += ' --gtest_filter="*' + test_name + '*"'
    cmd += ' --case_path=' + case_path
    cmd += ' --warm_repeat={} --perf_repeat={}'.format(args.warm_repeat,
                                                       args.perf_repeat)
    if args.test_stable:
        cmd += " --test_stable"

    cur_result['id'] = "{}_gid{}".format(i, gid)
    cur_result['test_name'] = test_name
    cur_result['case_path'] = case_path
    cur_result['cmd'] = cmd
    print("[{}][DO]: {}".format(i, cmd))
    log_name = './log/' + test_name + '/' + str(i) + "_" + str(gid) + '.log'
    cur_result['log_name'] = log_name
    os.system('mkdir -p ./log/{}'.format(test_name))
    cur_result.update(cmd_time_guard(cmd, log_name))

    tmp = os.popen('grep "dtype" {}'.format(case_path))
    cur_result['dtype'] = tmp.read()
    tmp.close()

    if queue_result is not None:
        queue_result.put(cur_result)
    else:
        return cur_result


def run_cases_list(gids, cases_list):
    gid_list = gids.split(",")
    result = []
    for i in range(len(cases_list)):
        test_name = cases_list[i]["test_name"]
        case_path = cases_list[i]["case_path"]

        if len(gid_list) == 1:
            result.append(run_case(i, gid_list[0], test_name, case_path))
        else:
            queue_result = multiprocessing.Queue()
            processes = []
            for gid in gid_list:
                son_p = Process(target=run_case,
                                args=(i, gid, test_name, case_path,
                                      queue_result))
                son_p.start()
                processes.append(son_p)
            for son_p in processes:
                son_p.join()

            result += [queue_result.get() for son_p in processes]
    return result


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

    return blas_api_list + dnn_api_list


parser = argparse.ArgumentParser()
parser.add_argument('--case_path',
                    dest='case_path',
                    type=str,
                    help='case path')
parser.add_argument('--cases_dir',
                    dest='cases_dir',
                    type=str,
                    help='cases dir')
parser.add_argument('--cases_list',
                    dest='cases_list',
                    type=str,
                    help='cases list')
parser.add_argument('--test_name',
                    dest='test_name',
                    type=str,
                    help='test name')
parser.add_argument('--rand_n', dest='rand_n', type=int, help='rand_n')
parser.add_argument('--gid', dest='gid', type=str, help='gid')
parser.add_argument('--warm_repeat',
                    dest='warm_repeat',
                    type=int,
                    default=1,
                    help='warm_repeat')
parser.add_argument('--perf_repeat',
                    dest='perf_repeat',
                    type=int,
                    default=1,
                    help='perf_repeat')
parser.add_argument("--test_stable",
                    action="store_true",
                    help="whether to test stablity")
parser.add_argument("--test_multi_core",
                    action="store_true",
                    help="whether to test multi core")
parser.add_argument('--time_out_threshold',
                    dest='time_out_threshold',
                    type=int,
                    help='time_out_threshold')

args = parser.parse_args()

if __name__ == "__main__":
    os.environ['MKL_SERVICE_FORCE_INTEL'] = "1"
    os.environ['DNN_RESERVE_DATA'] = "1"
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    os.environ['TECO_ENABLE_PROFILING'] = "1"
    os.environ['DNN_TEST_LOG'] = "1"

    dir_name = '../zoo'
    # dir_name = '../testcase/sdpti'

    if args.time_out_threshold is not None:
        time_out_threshold = args.time_out_threshold

    if args.test_name is not None:
        # api_list = [args.test_name]
        api_list = args.test_name.strip().split(";")
    else:
        api_list = get_api_list()

    if args.cases_dir is not None:
        dir_name = args.cases_dir

    cases_list = []
    if args.case_path is not None:
        if args.test_name is not None:
            cases_list.append({
                'test_name': args.test_name,
                'case_path': args.case_path
            })
        else:
            print("need test_name")
    else:
        for i in range(len(api_list)):
            test_name = api_list[i]
            gen_cases_list(cases_list, dir_name, test_name)

    if args.cases_list is not None:
        cases_list = []
        if os.path.exists(args.cases_list):
            with open(args.cases_list, "r") as f:
                cases = f.readlines()
            cases = [x.strip() for x in cases]
            for i in range(len(cases)):
                test_name = get_test_name_in_path(cases[i], api_list)
                if (test_name is not None):
                    cases_list.append({
                        'test_name': test_name,
                        'case_path': cases[i]
                    })

    rand_n = len(cases_list)
    if args.rand_n is not None:
        rand_n = min(rand_n, args.rand_n)
        random.shuffle(cases_list)
        cases_list = cases_list[0:rand_n]

    for i in range(len(cases_list)):
        print(cases_list[i])

    result = run_cases_list(args.gid, cases_list)
    # print(result)

    save_name = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    result_json = json.dumps(result, ensure_ascii=False, indent=4)
    # save_name_json = "result_{}_{}.json".format(args.test_name, args.gid)
    # save_name_xlsx = "result_{}_{}.xlsx".format(args.test_name, args.gid)
    save_name_json = "result_{}_gid{}.json".format(save_name, args.gid)
    save_name_xlsx = "result_{}_gid{}.xlsx".format(save_name, args.gid)

    with open(save_name_json, "w") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4))

    df = pd.read_json(result_json)
    df.to_excel(save_name_xlsx)
