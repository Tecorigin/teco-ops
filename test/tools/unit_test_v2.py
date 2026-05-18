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
from tecoalTest import TecoalTest
import sys

# from tecoalLog import merge_log


def grep_prototxt(dir):
    result = os.popen('find {} -name "*prototxt"'.format(dir))
    tmp = result.read()
    result.close()
    tmp = tmp.split("\n")
    del tmp[-1]
    return tmp


def add_case_path(cases_list, test_name, cur_list):
    for i in range(len(cur_list)):
        tmp = {"test_name": test_name, "case_path": cur_list[i]}
        cases_list.append(tmp)


def gen_cases_list(cases_list, dir, test_name):
    if not os.path.isdir(dir):
        print("gen_cases_list: not dir")
        return

    dirlist = os.listdir(dir)
    for dirret in dirlist:
        fullname = dir + "/" + dirret

        if dirret == test_name:
            add_case_path(cases_list, test_name, grep_prototxt(fullname))
            continue

        if os.path.isdir(fullname):
            gen_cases_list(cases_list, fullname, test_name)
        else:
            pass


def get_test_name_in_path(case_path, api_list):
    for tmp in os.path.abspath(case_path).split("/"):
        if tmp in api_list:
            return tmp
    return None


def get_api_list():
    result = os.popen("ls ../zoo/tecoal/")
    tmp = result.read()
    result.close()
    tmp = tmp.split("\n")
    del tmp[-1]
    blas_api_list = tmp

    result = os.popen("ls ../zoo/blas/")
    tmp = result.read()
    result.close()
    tmp = tmp.split("\n")
    del tmp[-1]
    dnn_api_list = tmp

    return blas_api_list + dnn_api_list


parser = argparse.ArgumentParser()
parser.add_argument("--case_path", dest="case_path", type=str, help="case path")
parser.add_argument("--cases_dir", dest="cases_dir", type=str, help="cases dir")
parser.add_argument("--cases_list", dest="cases_list", type=str, help="cases list")
parser.add_argument("--test_name", dest="test_name", type=str, help="test name")
parser.add_argument("--rand_n", dest="rand_n", type=int, help="rand_n")
parser.add_argument("--gid", dest="gid", type=str, help="gid")
parser.add_argument(
    "--warm_repeat", dest="warm_repeat", type=int, default=1, help="warm_repeat"
)
parser.add_argument(
    "--perf_repeat", dest="perf_repeat", type=int, default=100, help="perf_repeat"
)
parser.add_argument(
    "--gtest_repeat", dest="gtest_repeat", type=int, default=1, help="gtest_repeat"
)
parser.add_argument(
    "--test_stable",
    dest="test_stable",
    type=str,
    default="off",
    help="whether to test stablity, on/off",
)
parser.add_argument(
    "--test_multi_core", action="store_false", help="whether to test multi core"
)
parser.add_argument(
    "--time_out_threshold",
    dest="time_out_threshold",
    type=int,
    help="time_out_threshold",
)
parser.add_argument(
    "--merge_log", dest="merge_log", action="store_false", help="whether to merge log"
)
parser.add_argument(
    "--use_cuda",
    dest="use_cuda",
    action="store_true",
    help="whether to get cuda result",
)
parser.add_argument(
    "--subprocess_case",
    dest="subprocess_case",
    action="store_true",
    help="whether to subprocess_case",
)
parser.add_argument("--output", dest="output", type=str, help="output name")

args = parser.parse_args()

if __name__ == "__main__":
    os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
    os.environ["DNN_RESERVE_DATA"] = "1"
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    # os.environ["TECO_ENABLE_PROFILING"] = "1"
    os.environ["DNN_TEST_LOG"] = "1"

    dir_name = "./zoo"
    # dir_name = '../testcase/sdpti'

    test = TecoalTest()
    print(args)
    path = ""
    if args.gid is not None:
        if args.gid == 'all':
            core_num = int(subprocess.getstatusoutput('ls /dev/tcaicard* | wc -l')[1])
            test.gids = [int(gid) for gid in range(core_num)]
        elif "-" in args.gid:
            gid_min, gid_max = args.gid.split("-")
            test.gids = [int(gid) for gid in range(int(gid_min), int(gid_max) + 1)]
        else:
            gids = args.gid.split(",")
            print(gids)
            test.gids = [int(gid) for gid in gids]

    if args.time_out_threshold is not None:
        test.time_out_threshold = args.time_out_threshold

    if args.test_name is not None:
        # api_list = [args.test_name]
        test.op_names = args.test_name.strip().split(";")
        print(test.op_names)
    # else:
    #     api_list = get_api_list()

    if args.cases_dir is not None:
        path = args.cases_dir

    if args.warm_repeat is not None:
        test.warm_repeat = args.warm_repeat

    if args.perf_repeat is not None:
        test.perf_repeat = args.perf_repeat

    if args.gtest_repeat is not None:
        test.gtest_repeat = args.gtest_repeat

    if args.test_stable == "on":
        test.test_stable = True
    else:
        test.test_stable = False

    if args.test_multi_core is True:
        test.async_gids = True

    # cases_list = []
    if args.case_path is not None:
        path = args.case_path
    #     if args.test_name is not None:
    #         cases_list.append( {'test_name': args.test_name, 'case_path': args.case_path})
    #     else:
    #         print("need test_name")
    # else:
    #     for i in range(len(api_list)):
    #         test_name = api_list[i]
    #         gen_cases_list(cases_list, dir_name, test_name)

    if args.cases_list is not None:
        print(args.cases_list)
        path = args.cases_list

    if not path or not os.path.exists(path):
        print(f"{path} not find")
        # cases_list = []
        # if os.path.exists(args.cases_list):
        #     with open(args.cases_list, "r") as f:
        #         cases = f.readlines()
        #     cases = [x.strip() for x in cases]
        #     for i in range(len(cases)):
        #         test_name = get_test_name_in_path(cases[i], api_list)
        #         if(test_name is not None):
        #             cases_list.append( {'test_name': test_name, 'case_path': cases[i]})
    test.merge = args.merge_log
    test.use_cuda = args.use_cuda
    test.subprocess_case = args.subprocess_case
    # assert(len(cases_list) > 0)
    # if isinstance(cases_list[0], dict):
    #     cases_list = [dic["case_path"] for dic in cases_list]
    # rand_n = len(cases_list)
    if args.rand_n is not None:
        test.rand_n = args.rand_n
        # print(test.rand_n)

    save_name = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    if args.output is not None:
        save_name = args.output
    if not save_name.endswith(".xlsx"):
        save_name = save_name + ".xlsx"

    # with open(f"{save_name}_case_list.txt", 'w') as f:
    #     for case in cases_list:
    #         f.write(case)
    #         f.write("\n")
    result = test.unit_test(path)
    accu_data = result.getLogResult("accu")
    test_status = True
    for cur_data in accu_data:
        if "PASS" not in cur_data["result"]:
            test_status = False
            break
    accu_data = pd.DataFrame(accu_data)
    perf_data = result.getLogResult("perf")
    json_data = {}
    passed_count = 0
    for res in perf_data:
        case_path, ht = res["case_path"], res["hardware_time(ms)"]
        ar = res["result"]
        json_data[case_path] = {"hardware_time(ms)": ht}
        passed_count += 1 if ar in ["PASS_S1", "PASS_S2"] else 0
    accu_score = passed_count / len(perf_data) * 6.0
    json_data["accu_score"] = f"{accu_score:.5f}"
    jn = save_name.replace("xlsx", "json")
    perf_data = pd.DataFrame(perf_data)
    result.getSummary()
    # sum0 = pd.DataFrame([sum_data[0]])
    # sum3 = pd.DataFrame(sum_data[3])
    # sum1 = pd.DataFrame(sum_data[1]) if len(sum_data[1]) > 0 else []
    # sum2 = pd.DataFrame(sum_data[2]) if len(sum_data[2]) > 0 else []
    summary = pd.DataFrame(result.summary)
    failed_cases = (
        pd.DataFrame(result.failed_casepath) if result.failed_casepath else []
    )
    op_details = pd.DataFrame(result.op_details)
    kernel_details = (
        pd.DataFrame(result.kernel_details) if result.kernel_details else []
    )

    # accu_data.to_excel(f"./{save_name}_accu.xlsx")
    # perf_data.to_excel(f"./{save_name}_perf.xlsx")
    with pd.ExcelWriter(f"./{save_name}", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        if isinstance(failed_cases, pd.DataFrame):
            failed_cases.to_excel(writer, sheet_name="failed_casepath", index=False)
        if isinstance(kernel_details, pd.DataFrame):
            kernel_details.to_excel(writer, sheet_name="kernel_details", index=False)
        op_details.to_excel(writer, sheet_name="op_details", index=False)
        accu_data.to_excel(writer, sheet_name="accu", index=False)
        perf_data.to_excel(writer, sheet_name="perf", index=False)
    with open(jn, 'w') as f:
        f.write(json.dumps(json_data, indent=4))
    if not test_status:
        print("TEST FAILED!!!")
        sys.exit(-1)
    else:
        print("TEST PASSED!!!")
        sys.exit(0)

    # to_txt = save_name[:save_name.find(".xlsx")]+"_fault.txt"
    # fault_txt = {}
    # with open(to_txt, 'w') as f:
    #     for index, row in accu_data.iterrows():
    #         if "ERROR_FAULT" in row["result"] and row["case_path"] not in fault_txt:
    #             fault_txt[row["case_path"]] = 1
    #             f.write(row["case_path"])
    #             f.write("\n")
    # if len(fault_txt) == 0:
    #     os.remove(to_txt)
    # if merge:
    #     log_dir = os.listdir("./log")
    #     log_dir = sorted(log_dir)[-1]
    #     log_path = os.path.join("./log",log_dir)
    #     merge_log(log_path, save_name)

    # with open(f"./{save_name}_accu.json", "w") as f:
    #     f.write(accu_data.to_json(orient="records"))
    # with open(f"./{save_name}_perf.json", "w") as f:
    #     f.write(perf_data.to_json(orient="records"))
