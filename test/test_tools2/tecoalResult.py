import logging
import os
import sys
import time
import logging
import copy
import pandas as pd
import statistics
from tecoalCase import TecoalCase
from tecoalLog import *
import subprocess

# # only support list[ dict enum ] (not check)
# def matchResult(cases, in_result, keys):
#     result = []

#     if keys is None:
#         keys = cases.kesy()[0]

#     cases_keys = []
#     for record in cases:
#         cases_keys.append(hash(str([record.get(key, "NAN") for key in keys])))

#     for record in in_result:
#         cur_hash = hash(str([record.get(key, "NAN") for key in keys]))

#         if cur_hash in cases_keys:
#             index = cases_keys.index(cur_hash)
#             result.update(record)

#     return result


#
# def updateResult()
def form(keys, result):
    # assert(keys is not None)
    cur_result = [
        dict(zip(keys, [record.get(key, "NAN") for key in keys])) for record in result
    ]
    return cur_result


def read_sheet(excel_path, sheet_name):
    if not os.path.exists(excel_path):
        return []
    df_sheet = pd.read_excel(excel_path, sheet_name)
    data = []
    for index, row in df_sheet.iterrows():
        data_dict = {col: val for col, val in row.items()}
        data.append(data_dict)
    # self.perf_result = data
    return data


def kernel_valid(kernel):
    if not kernel or kernel == "NAN":
        return False
    return True


# def diff(result, compar_result, keys, mode="diff1"):
#     cases_list = [record.get(keys, default="NAN") for record in result]
#     comapre_result = matchResult(cases_list, result)

#     result = copy.deepcopy(result)
#     for i in range(len(cases_list)):
#         if keys in comapre_result[i].keys():
#             diff1 = (
#                 (comapre_result[i][keys] - result[i][keys])
#                 / comapre_result[i][keys]
#                 * 100
#             )
#         else:
#             diff1 = "NAN"
#         result[i]["diff1({})".format(keys)] = diff1

#     return result


def deduplicate(result, case_times={}):
    #  cases_list = {}
    perf_result = {}
    assert result is not None
    case_path = {}
    output = {}
    for cur_res in result:
        if "case_path" not in cur_res:
            cur_res["case_path"] = cur_res["log"]
            cur_res["times"] = cur_res.get("times", 1)
        else:
            if case_times and cur_res["case_path"] in case_times:
                cur_res["times"] = case_times[cur_res["case_path"]]
            else:
                cur_res["times"] = cur_res.get("times", 1)

        if "output" not in cur_res:
            cur_res["output"] = cur_res["log"]
        out = cur_res["case_path"] + cur_res["output"]
        if "gid" in cur_res["log"]:
            idx = cur_res["log"].rfind("/")
            tmp = cur_res["log"][idx + 1 :]
            out = out + tmp[tmp.find("_") + 1 : tmp.rfind("_")]
        if cur_res["case_path"] not in case_path:
            case_path[cur_res["case_path"]] = 1
            output[out] = 1
            perf_result[cur_res["case_path"]] = cur_res
        else:
            if out in output:
                perf_result[cur_res["case_path"]] = cur_res
            else:
                if cur_res["result"] not in ["PASS_S1", "PASS_S2"]:
                    perf_result[cur_res["case_path"]] = cur_res
    return list(perf_result.values())


#  if "gid" in result[0]["log"]:
#     case_path = {}
#     for cur_res in result:
#         if "case_path" not in cur_res: cur_res["case_path"] = cur_res["log"]
#         if cur_res["case_path"] not in case_path:
#             case_path[cur_res["case_path"]] = [cur_res["hardware_time(ms)"]]
#         else:
#             case_path[cur_res["case_path"]].append(cur_res["hardware_time(ms)"])
#     for cur_res in result:
#         if cur_res["case_path"] not in cases_list:
#             cases_list[cur_res["case_path"]] = 1
#             mean, var = statistics.mean(case_path[cur_res["case_path"]]), statistics.variance(case_path[cur_res["case_path"]])
#             cur_res["hardware_time(ms)"] = mean
#             cur_res["hardware_time_var"] = var
#             perf_result.append(cur_res)

#  else:
#     for cur_res in result:
#         if "case_path" not in cur_res: cur_res["case_path"] = cur_res["log"]
#         if cur_res["case_path"] not in cases_list:
#             cases_list[cur_res["case_path"]] = 1
#             perf_result.append(cur_res)


def isvalid(key):
    if isinstance(key, str):
        # print(key)
        if not key.strip().isdigit():
            return False
    if float(key) <= 0:
        # print(key)
        return False
    return True


def get_err_from_str(data):
    # 1、str => list
    # [cur,bench,index,err]
    # 需要去掉末尾的']'
    *others, err = data.split(",")
    return err[:-1]


def get_spe_from_cmd(used_gids):
    # used_gids [0,1,2...]
    # return xxx mhz
    cmd = "source /opt/tecoai/setvars.sh && teco-smi --showspeclk"
    process = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if process.returncode == 0:
        gid_spe = {}
        spes = []
        speclk_list = process.stdout.strip().split("\n")[4:-1]
        for content in speclk_list:
            content = [val.strip() for val in content.strip().split("|") if val]
            _, gids, spe, max_spe = content
            gid_min, gid_max = (int(gids[0]), int(gids[-1]))
            for gid in range(gid_min, gid_max + 1):
                gid_spe[gid] = spe
                if gid in used_gids and spe not in spes:
                    spes.append(spe)
        return spes

    else:
        return []


def op_details_extend(cur_op_details, pre_op_details):
    ##only support list with dict
    cur_op_details = sorted(cur_op_details, key=lambda x: x["op_name"])
    pre_op_details = sorted(pre_op_details, key=lambda x: x["op_name"])
    to_extend_op = []
    i, j = 0, 0
    while i < len(cur_op_details) and j < len(pre_op_details):
        if cur_op_details[i]["op_name"] == pre_op_details[j]["op_name"]:
            if cnt := pre_op_details[j]["op_cases"]:
                cur_op_details[i]["op_cases"] += cnt
            if isvalid(cur_op_details[i]["tot_hardware_time(ms)"]) and isvalid(
                pre_op_details[j]["tot_hardware_time(ms)"]
            ):
                cur_op_details[i]["tot_hardware_time(ms)"] += pre_op_details[j][
                    "tot_hardware_time(ms)"
                ]
            i, j = i + 1, j + 1
        elif cur_op_details[i]["op_name"] < pre_op_details[j]["op_name"]:
            i += 1
        else:
            to_extend_op.append(pre_op_details[j])
            j += 1
    if j < len(pre_op_details):
        to_extend_op.extend(pre_op_details[j:])
    if to_extend_op:
        cur_op_details.extend(to_extend_op)


def kernel_details_extend(cur_kernel_details, pre_kernel_details):
    def kernel_sort(item):
        return (item["op_name"], item["kernel_details"])

    cur_kernel_details = sorted(cur_kernel_details, key=kernel_sort)
    pre_kernel_details = sorted(pre_kernel_details, key=kernel_sort)
    i, j = 0, 0
    to_extend_kernel = []
    while i < len(cur_kernel_details) and j < len(pre_kernel_details):
        if cur_kernel_details[i]["op_name"] > pre_kernel_details[j]["op_name"]:
            to_extend_kernel.append(pre_kernel_details[j])
            j += 1
        elif cur_kernel_details[i]["op_name"] < pre_kernel_details[j]["op_name"]:
            i += 1
        else:
            if (
                cur_kernel_details[i]["kernel_details"]
                > pre_kernel_details[j]["kernel_details"]
            ):
                to_extend_kernel.append(pre_kernel_details[j])
                j += 1
            elif (
                cur_kernel_details[i]["kernel_details"]
                < pre_kernel_details[j]["kernel_details"]
            ):
                i += 1
            else:
                if cnt := pre_kernel_details[j]["kernel_count"]:
                    cur_kernel_details[i]["kernel_count"] += cnt
                i += 1
                j += 1
    if j < len(pre_kernel_details):
        to_extend_kernel.extend(pre_kernel_details[j:])
    if to_extend_kernel:
        cur_kernel_details.extend(to_extend_kernel)


class KernelDetails:
    kernel_details = "unknown"
    kernel_count = 0

    def __init__(self, kernel_details, kernel_count) -> None:
        self.kernel_details = kernel_details
        self.kernel_count = kernel_count

    def kernel_add(self, times):
        self.kernel_count += times


class OpDetails:
    op_name = "unknown"
    op_cases = 0
    tot_hardware_time = 0
    failed_count = 0
    insuf_count = 0
    # kernel_name : kernel_obj
    kernels = {}

    # only support kernel: obj or [obj1,obj2]
    def __init__(self, op_name, op_cases, hardware_time, kernel=None) -> None:
        self.op_name = op_name
        self.op_cases = op_cases
        if isvalid(hardware_time):
            if float(hardware_time) >= 0:
                self.tot_hardware_time = float(hardware_time)
            else:
                self.tot_hardware_time = "NAN"
        else:
            self.tot_hardware_time = "NAN"
        if isinstance(kernel, KernelDetails):
            kernel = [kernel]
        if kernel and isinstance(kernel, list):
            try:
                for ker_obj in kernel:
                    self.kernels[ker_obj.kernel_details] = ker_obj
            except Exception as e:
                print(e)

    def op_add(self, op_cases=0, hardware_time=0, kernel=None):
        self.op_cases += op_cases
        if isvalid(hardware_time) and isvalid(self.tot_hardware_time):
            if float(hardware_time) >= 0:
                self.tot_hardware_time += hardware_time
            else:
                self.tot_hardware_time = "NAN"
        else:
            self.tot_hardware_time = "NAN"

        if isinstance(kernel, KernelDetails):
            kernel = [kernel]
        if kernel and isinstance(kernel, list):
            for ker_obj in kernel:
                try:
                    if ker_obj.kernel_details not in self.kernels:
                        self.kernels[ker_obj.kernel_details] = ker_obj
                    else:
                        self.kernels[ker_obj.kernel_details].kernel_add(
                            ker_obj.kernel_count
                        )
                except Exception as e:
                    print(e)


# now, do not support repeat case_path
class TecoalResult:
    # test info
    test_name = "simple test"
    test_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    test_info = {}
    keys = "case_path"
    gids = []
    default_format = [
        "case_path",
        "op_name",
        "user_time",
        "interface_time(ms)",
        "hardware_time(ms)",
        "launch_time(ms)",
        "theory_ios(Bytes)",
        "io_bandwidth(GB/s)",
        "kernel_details",
        "result",
    ]
    accu_format = [
        "log",
        "case_path",
        "op_name",
        "shape",
        "dtype",
        "layout",
        "param",
        "output",
        "kernel_details",
        "DIFF3_MAX_GPU",
        "DIFF3_MAX_TECO",
        "DIFF3_MAX_THRESHOLD",
        "DIFF3_MEAN_GPU",
        "DIFF3_MEAN_TECO",
        "DIFF3_MEAN_THRESHOLD",
        "result",
        "result_hash",
        "host_result",
    ]

    perf_format = [
        "log",
        "case_path",
        "op_name",
        "shape",
        "dtype",
        "layout",
        "kernel_details",
        "interface_time(ms)",
        "hardware_time(ms)",
        "launch_time(ms)",
        "cache_miss_details",
        "io_bandwidth(GB/s)",
        "theory_ios(Bytes)",
        "compute_force(op/s)",
        "theory_ops(Ops)",
        "minmax_hardware_time(ms)",
        "minmax_hardware_time_gap(ms)",
        "minmax_hardware_time_gap/avg(%)",
        "result",
        "host_result",
        "times",
    ]

    # test result
    result = None  # list with dict enum
    perf_result = None
    summary = None  # test env and conclude test result
    failed_casepath = None
    op_details = None  # op_name tot_cases hardware_time
    kernel_details = None  # op_name kernel_name kernel_count
    case_times = {}

    def extend(self, tecoal_result):
        if isinstance(tecoal_result, TecoalResult):
            if tecoal_result.result:
                self.result.extend(tecoal_result.result)

            if tecoal_result.perf_result:
                self.perf_result.extend(tecoal_result.perf_result)

            if tecoal_result.failed_casepath:
                if not self.failed_casepath:
                    self.failed_casepath = tecoal_result.failed_casepath
                else:
                    self.failed_casepath.extend(tecoal_result.failed_casepath)
                    # self.failed_casepath = list(set(self.failed_casepath))

            if tecoal_result.op_details:
                if not self.op_details:
                    self.op_details = tecoal_result.op_details
                else:
                    op_details_extend(self.op_details, tecoal_result.op_details)

            if tecoal_result.summary:
                if not self.summary:
                    self.summary = tecoal_result.summary
                else:
                    cur_summary, pre_summary = self.summary[0], tecoal_result.summary[0]
                    for key in cur_summary:
                        cur_val, pre_val = cur_summary[key], pre_summary[key]
                        if isinstance(cur_val, str) or key == "spe clock (MHz)":
                            if cur_val != pre_val:
                                cur_summary[key] = f"{cur_val} {pre_val}"
                        else:
                            if key != "tot_ops":
                                cur_summary[key] += pre_val
                            else:
                                if self.op_details:
                                    cur_summary[key] = len(self.op_details)

            if tecoal_result.kernel_details:
                if not self.kernel_details:
                    self.kernel_details = tecoal_result.kernel_details
                else:
                    kernel_details_extend(
                        self.kernel_details, tecoal_result.kernel_details
                    )

        else:
            logging.warning("not support now")

    def addResult(self, result):
        tecoal_result = TecoalResult(result)
        self.extend(tecoal_result)

    # only support one key now
    # def updateResult(self, result, keys='case_path'):
    # if isinstance(cases, TecoalCase):
    #     cases_list = cases.cases_list.deepcopy()
    # elif isinstance(cases, list):
    #     cases_list = cases
    # else:
    #     logging.error('matchResult error')

    #     self_keys_list = [ i.get(keys, default='NAN') for i in self.result]
    #     for tmp in result:
    #         if keys not in tmp.keys():
    #             continue

    #         idx = self_keys_list.index(tmp[kyes])
    #         self.result[idx].update(tmp)

    def resetResult(self, result):
        if isinstance(result, TecoalResult):
            self = copy.deepcopy(result)
        elif isinstance(result, str):
            if result.endswith("xlsx"):
                self.readFromExcel(result)
                # if len(res) == 1:
                #     self.result = res[0]
                # else:
                #     self.result, self.perf_result = res
                # print(len(self.result))

            else:
                self.result = getResult(result)
        else:
            self.result = getResult(result)

    def getLogResult(self, mode="accu"):
        assert (mode == "accu") or (mode == "all") or (mode == "perf")
        if mode == "perf":
            # if "gid" in self.result[0]["log"]:
            #     self.perf_format.append("hardware_time_var")
            if not self.perf_result:
                self.perf_result = deduplicate(self.result, self.case_times)
            return form(self.perf_format, self.perf_result)
        else:
            return form(self.accu_format, self.result)

    # not check list detail
    def __init__(self, result=None):
        if result is None:
            self.result = []
            self.perf_result = []
            self.summary = []
            self.failed_casepath = []
            self.op_details = []
            self.kernel_details = []
            return

        self.resetResult(result)

    def readFromExcel(self, path):
        if os.path.isfile(path):
            file = pd.ExcelFile(path)
            sheet_names = file.sheet_names
            # sheet_name = sheet_names[0] if len(sheet_names) == 1 else "perf"
            # df = pd.read_excel(path, sheet_name)
            #
            if len(sheet_names) == 1:
                self.result = read_sheet(path, 0)
            else:
                if "accu" in sheet_names:
                    self.result = read_sheet(path, "accu")
                if "perf" in sheet_names:
                    self.perf_result = read_sheet(path, "perf")
                if "summary" in sheet_names:
                    self.summary = read_sheet(path, "summary")
                if "op_details" in sheet_names:
                    self.op_details = read_sheet(path, "op_details")
                if "kernel_details" in sheet_names:
                    self.kernel_details = read_sheet(path, "kernel_details")
                if "failed_casepath" in sheet_names:
                    self.failed_casepath = read_sheet(path, "failed_casepath")

        else:
            logging.error(f"{path} not exists")

    # keys is list
    def format(self, keys=None):
        if keys is None:
            keys = self.default_format
        cur_result = self.perf_result if self.perf_result is not None else self.result
        result = [
            dict(zip(keys, [record.get(key, "NAN") for key in keys]))
            for record in cur_result
        ]
        return result

    # diff1: (new - old)/old*100%
    def diff_perf(
        self,
        result,
        keys="hardware_time(ms)",
        mode="diff1",
        threshold=5,
        is_ignore=False,
    ):
        # cases_list = [record['case_path'] for record in self.result]
        # comapre_result = sortResult(self.result, result)

        # self.format()
        # result = copy.deepcopy(self.result)
        # for i in range(len(cases_list)):
        #     if keys in comapre_result[i].keys():
        #         diff1 = (comapre_result[i][keys] -
        #                  result[i][keys]) / comapre_result[i][keys] * 100
        #     else:
        #         diff1 = 'NAN'
        #     result[i]['diff1({})'.format(keys)] = diff1

        # return result
        # print(type(result))
        if isinstance(result, TecoalResult):
            if result.perf_result is not None:
                result = result.perf_result
            else:
                # result = result.getLogResult("perf")
                # print(len(result))
                result = deduplicate(result.result)

        else:
            result = deduplicate(result)

        if self.perf_result is None:
            self.perf_result = deduplicate(self.result)
        # print(len(self.perf_result))
        result_new = self.perf_result
        result_new.sort(key=lambda x: x["case_path"])
        result.sort(key=lambda x: x["case_path"])
        len_new, len_old = len(result_new), len(result)
        i, j = 0, 0
        not_matched, rise, fall_back, flat = 0, 0, 0, 0
        tot_hardware_time_new, tot_hardware_time_old = 0.0, 0.0
        rise_path, fall_path = [], []
        op_details = {}
        rise_hardware_time, fallback_hardware_time = 0, 0
        while i < len_new and j < len_old:
            dic_new, dic = result_new[i], result[j]
            if dic_new["case_path"] < dic["case_path"]:
                result_new[i]["performance"] = "invalid"
                not_matched += 1
                i += 1
            elif dic_new["case_path"] > dic["case_path"]:
                j += 1
            else:
                if keys in result[j] and keys in result_new[i]:
                    if isvalid(result_new[i][keys]) and isvalid(result[j][keys]):
                        tot_hardware_time_new += result_new[i][keys]
                        tot_hardware_time_old += result[j][keys]
                        diff1 = (
                            (float(-result_new[i][keys]) + float(result[j][keys]))
                            / float(result[j][keys])
                            * 100
                        )
                        result_new[i]["hardware_time(base_line)"] = result[j][keys]
                        result_new[i]["rate(%)"] = diff1
                        result_new[i]["kernel_details(base_line)"] = result[j][
                            "kernel_details"
                        ]
                        result_new[i]["interface_time(base_line)"] = result[j].get(
                            "interface_time(ms)", "NAN"
                        )
                        try:
                            result_new[i]["cache_miss_time(ns)"] = 0
                            cache_misses = eval(result_new[i]["cache_miss_details"])
                            for cache_miss in cache_misses:
                                result_new[i]["cache_miss_time(ns)"] += int(
                                    cache_miss.split(",")[4].split(" ")[-1]
                                )
                            result_new[i]["cache_miss_details(base_line)"] = result[j][
                                "cache_miss_details"
                            ]
                            result_new[i]["cache_miss_time(ns)(base_line)"] = 0
                            cache_misses = eval(
                                result_new[i]["cache_miss_details(base_line)"]
                            )
                            for cache_miss in cache_misses:
                                result_new[i]["cache_miss_time(ns)(base_line)"] += int(
                                    cache_miss.split(",")[4].split(" ")[-1]
                                )
                        except Exception as e:
                            result_new[i]["cache_miss_time(ns)"] = result_new[i].get(
                                "cache_miss_time(ns)", 0
                            )
                            result_new[i]["cache_miss_details(base_line)"] = 0
                            result_new[i]["cache_miss_time(ns)(base_line)"] = (
                                result_new[i].get("cache_miss_time(ns)(base_line)", 0)
                            )

                        try:
                            cur_min_hwtime, cur_max_hwtime = eval(
                                result_new[i]["minmax_hardware_time(ms)"]
                            )
                            pre_min_hwtime, pre_max_hwtime = eval(
                                result[j]["minmax_hardware_time(ms)"]
                            )
                            min_hwtime_diff = cur_min_hwtime - pre_min_hwtime
                            max_hwtime_diff = cur_max_hwtime - pre_max_hwtime
                            result_new[i]["min_hardwaretime_diff(ms)"] = min_hwtime_diff
                            result_new[i]["max_hardwaretime_diff(ms)"] = max_hwtime_diff
                            result_new[i]["hardwaretime_iou"] = (
                                min(cur_max_hwtime, pre_max_hwtime)
                                - max(cur_min_hwtime, pre_min_hwtime)
                            ) / (
                                max(cur_max_hwtime, pre_max_hwtime)
                                - min(cur_min_hwtime, pre_min_hwtime)
                            )
                            if result_new[i]["hardwaretime_iou"] < 0:
                                result_new[i]["hardwaretime_iou"] = 0
                        except Exception as e:
                            pass

                        if "op_name" in result_new[i]:
                            op = result_new[i]["op_name"]
                            if op not in op_details:
                                op_details[op] = {
                                    "cur_hardware_time": 0,
                                    "pre_hardware_time": 0,
                                    "rise": [0, "", 0, 0, 0, 0],
                                    "fall_back": [0, "", 0, 0, 0, 0],
                                }
                            op_details[op]["cur_hardware_time"] += result_new[i][keys]
                            op_details[op]["pre_hardware_time"] += result[j][keys]
                        if diff1 > threshold:
                            if (
                                is_ignore
                                and abs(result_new[i][keys] - result[j][keys]) > 0.01
                            ) or not is_ignore:
                                result_new[i]["performance"] = "rise"
                                rise += 1
                                rise_hardware_time += abs(
                                    result_new[i][keys] - result[j][keys]
                                )
                                rise_path.append(result_new[i]["case_path"])
                                if "op_name" in result_new[i]:
                                    op_details[op]["rise"][-1] += abs(
                                        result_new[i][keys] - result[j][keys]
                                    )
                                    op_details[op]["rise"][-2] += 1
                                    if diff1 > op_details[op]["rise"][0]:
                                        op_details[op]["rise"][0] = diff1
                                        op_details[op]["rise"][1] = result_new[i][
                                            "case_path"
                                        ]
                                        op_details[op]["rise"][2] = result_new[i][keys]
                                        op_details[op]["rise"][3] = result[j][keys]
                            else:
                                result_new[i]["performance"] = "ignore"

                        elif diff1 < -threshold:
                            if (
                                is_ignore
                                and abs(result_new[i][keys] - result[j][keys]) > 0.01
                            ) or not is_ignore:
                                fallback_hardware_time += abs(
                                    result_new[i][keys] - result[j][keys]
                                )
                                cur_cache_miss_time = result_new[i].get(
                                    "cache_miss_time(ns)", 0
                                )
                                pre_cache_miss_time = result_new[i].get(
                                    "cache_miss_time(ns)(base_line)", 0
                                )
                                if cur_cache_miss_time == "NAN":
                                    cur_cache_miss_time = 0
                                if pre_cache_miss_time == "NAN":
                                    pre_cache_miss_time = 0
                                cache_miss_time_diff = (
                                    cur_cache_miss_time - pre_cache_miss_time
                                ) * 1e-6
                                hardware_time_diff = (
                                    result_new[i][keys]
                                    - result[j][keys] * (100 + threshold) / 100
                                )
                                if (
                                    cache_miss_time_diff
                                    and cache_miss_time_diff >= hardware_time_diff
                                ):
                                    result_new[i]["performance"] = "cache_miss"
                                else:
                                    result_new[i]["performance"] = "fall_back"
                                    fall_back += 1
                                    fall_path.append(result_new[i]["case_path"])
                                    op_details[op]["fall_back"][-1] += abs(
                                        result_new[i][keys] - result[j][keys]
                                    )
                                    op_details[op]["fall_back"][-2] += 1
                                    if diff1 < op_details[op]["rise"][0]:
                                        op_details[op]["fall_back"][0] = diff1
                                        op_details[op]["fall_back"][1] = result_new[i][
                                            "case_path"
                                        ]
                                        op_details[op]["fall_back"][2] = result_new[i][
                                            keys
                                        ]
                                        op_details[op]["fall_back"][3] = result[j][keys]
                            else:
                                result_new[i]["performance"] = "ignore"
                        else:
                            result_new[i]["performance"] = "flat"
                            flat += 1
                    else:
                        result_new[i]["performance"] = "invalid"
                        not_matched += 1
                else:
                    result_new[i]["performance"] = "invalid"
                    not_matched += 1
                cur_kernel = result_new[i].get("kernel_details", "NAN")
                pre_kernel = result[j].get("kernel_details", "NAN")
                if kernel_valid(cur_kernel) and kernel_valid(pre_kernel):
                    try:
                        cur_kernel = fix_kernel_str(cur_kernel.strip())
                        pre_kernel = fix_kernel_str(pre_kernel.strip())
                        if cur_kernel and pre_kernel:
                            if cur_kernel != pre_kernel:
                                result_new[i]["kernel_diff"] = "different_kernel"
                            else:
                                result_new[i]["kernel_diff"] = "same_kernel"
                        else:
                            result_new[i]["kernel_diff"] = "invalid"
                    except Exception as e:
                        result_new[i]["kernel_diff"] = "invalid"
                else:
                    result_new[i]["kernel_diff"] = "invalid"
                i += 1
                j += 1
        if i < len_new:
            for k in range(i, len_new):
                result_new[k]["performance"] = "invalid"
                result_new[k]["kernel_diff"] = "invalid"
                not_matched += 1
        diff_result = [
            {
                "tot_cases": len_new,
                "not_matched": not_matched,
                "rise": [rise, rise / (len_new - not_matched) * 100.0],
                "flat": [flat, flat / (len_new - not_matched) * 100.0],
                "fall_back": [fall_back, fall_back / (len_new - not_matched) * 100.0],
                "tot_hardware_time_new": tot_hardware_time_new,
                "tot_hardware_time_old": tot_hardware_time_old,
                "abs_diff": tot_hardware_time_new - tot_hardware_time_old,
                "rela_diff(%)": (
                    (tot_hardware_time_new - tot_hardware_time_old)
                    / tot_hardware_time_old
                )
                * 100.0,
                "rise_hardware_time": rise_hardware_time,
                "fallback_hardware_time": fallback_hardware_time,
            }
        ]
        for _ in range(4):
            diff_result.append([])
        # print(len(diff_result))
        for op in op_details:
            cur_data = {}
            cur_data["op_name"] = op
            cur_data["abs_diff"] = (
                op_details[op]["cur_hardware_time"]
                - op_details[op]["pre_hardware_time"]
            )
            cur_data["rela_diff(%)"] = (
                (
                    op_details[op]["cur_hardware_time"]
                    - op_details[op]["pre_hardware_time"]
                )
                / op_details[op]["pre_hardware_time"]
                * 100.0
            )
            cur_data["tot_rise_hardware_time"] = (
                op_details[op]["rise"][-1] if op_details[op]["rise"][-1] != 0 else ""
            )
            cur_data["tot_rise_count"] = (
                op_details[op]["rise"][-2] if op_details[op]["rise"][-2] != 0 else ""
            )
            cur_data["tot_fallback_hardware_time"] = (
                op_details[op]["fall_back"][-1]
                if op_details[op]["fall_back"][-1] != 0
                else ""
            )
            cur_data["tot_fallback_count"] = (
                op_details[op]["fall_back"][-2]
                if op_details[op]["fall_back"][-2] != 0
                else ""
            )
            diff_result[1].append(cur_data)

            detail_data = {}
            detail_data["op_name"] = op
            detail_data["rise_casepath"] = op_details[op]["rise"][1]
            detail_data["rise_diff1"] = op_details[op]["rise"][0] if diff1 > 0 else ""
            detail_data["rise_hardware_time"] = (
                op_details[op]["rise"][2] if op_details[op]["rise"][2] > 0 else ""
            )
            detail_data["rise_harware_time_base"] = (
                op_details[op]["rise"][3] if op_details[op]["rise"][3] > 0 else ""
            )
            detail_data["fallback_casepath"] = op_details[op]["fall_back"][1]
            detail_data["fallback_diff1"] = (
                op_details[op]["fall_back"][0] if diff1 < 0 else ""
            )
            detail_data["fallback_hardware_time"] = (
                op_details[op]["fall_back"][2]
                if op_details[op]["fall_back"][2] > 0
                else ""
            )
            detail_data["fallback_harware_time_base"] = (
                op_details[op]["fall_back"][3]
                if op_details[op]["fall_back"][3] > 0
                else ""
            )
            diff_result[2].append(detail_data)
        for rp in rise_path:
            cur_data = {}
            cur_data["case_path"] = rp
            diff_result[3].append(cur_data)
        for fp in fall_path:
            cur_data = {}
            cur_data["case_path"] = fp
            diff_result[4].append(cur_data)
        # diff_result.append(result_new)
        # print(diff_result[0])
        return diff_result

    def diff_accu(
        self, bench_result, keys=["DIFF3_MAX_TECO", "DIFF3_MEAN_TECO"], is_ci=False
    ):
        assert self.result
        if isinstance(bench_result, TecoalResult):
            bench_result = bench_result.result
        else:
            bench_obj = TecoalResult(bench_result)
            bench_result = bench_obj.result
        assert bench_result

        def accu_sort(item):
            return (item["case_path"], item["output"])

        self.result.sort(key=accu_sort)
        bench_result.sort(key=accu_sort)
        i, j = 0, 0
        len_new, len_bench = len(self.result), len(bench_result)
        while i < len_new and j < len_bench:
            if self.result[i]["case_path"] < bench_result[j]["case_path"]:
                self.result[i]["accu_performance"] = "invalid"
                i += 1
            elif self.result[i]["case_path"] > bench_result[j]["case_path"]:
                j += 1
            else:
                if self.result[i]["output"] > bench_result[j]["output"]:
                    j += 1
                elif self.result[i]["output"] < bench_result[j]["output"]:
                    self.result[i]["accu_performance"] = "invalid"
                    i += 1
                else:  # self.result[i]["output"] == bench_result[j]["output"]:
                    key_max, key_mean = keys
                    try:
                        cur_teco_max, bench_teco_max = (
                            self.result[i][key_max],
                            bench_result[j][key_max],
                        )
                        cur_teco_mean, bench_teco_mean = (
                            self.result[i][key_mean],
                            bench_result[j][key_mean],
                        )
                        try:
                            cur_teco_max_err, bench_teco_max_err = float(
                                get_err_from_str(cur_teco_max)
                            ), float(get_err_from_str(bench_teco_max))
                            cur_teco_mean_err, bench_teco_mean_err = float(
                                get_err_from_str(cur_teco_mean)
                            ), float(get_err_from_str(bench_teco_mean))
                            self.result[i]["diff_err"] = [
                                cur_teco_max_err,
                                cur_teco_mean_err,
                            ]
                            self.result[i]["diff_err(base_line)"] = [
                                bench_teco_max_err,
                                bench_teco_mean_err,
                            ]
                            self.result[i]["kernel_details(base_line)"] = bench_result[
                                j
                            ]["kernel_details"]
                            if (cur_teco_max_err > bench_teco_max_err) or (
                                cur_teco_mean_err > bench_teco_mean_err
                            ):
                                self.result[i]["accu_performance"] = "fall_back"
                                if self.result[i]["result"] == "PASS_S2":
                                    self.result[i][
                                        "accu_performance"
                                    ] = "fall_back(PASS_S2)"
                                elif is_ci and self.result[i]["result"] == "PASS_S3":
                                    self.result[i][
                                        "accu_performance"
                                    ] = "fall_back(PASS_S3)"

                            elif (cur_teco_max_err == bench_teco_max_err) and (
                                cur_teco_mean_err == bench_teco_mean_err
                            ):
                                self.result[i]["accu_performance"] = "flat"
                                result_hash = self.result[i].get("result_hash", None)
                                result_hash_bench = bench_result[j].get(
                                    "result_hash", None
                                )
                                if (
                                    result_hash
                                    and result_hash_bench
                                    and result_hash != result_hash_bench
                                ):
                                    self.result[i]["accu_performance"] = "diff"

                            else:
                                self.result[i]["accu_performance"] = "rise"
                        except Exception as e:
                            if "kernel_details" in bench_result[j]:
                                self.result[i]["kernel_details(base_line)"] = (
                                    bench_result[j]["kernel_details"]
                                )
                            result_hash = self.result[i].get("result_hash", None)
                            result_hash_bench = bench_result[j].get("result_hash", None)
                            if result_hash and result_hash_bench:
                                if result_hash != result_hash_bench:
                                    self.result[i]["accu_performance"] = "diff"
                                else:
                                    self.result[i]["accu_performance"] = "flat"
                            else:
                                self.result[i]["accu_performance"] = "invalid"
                            print(e)
                    except Exception as e:
                        self.result[i]["accu_performance"] = "invalid"
                        print(e)
                    cur_kernel = self.result[i].get("kernel_details", "NAN")
                    pre_kernel = bench_result[j].get("kernel_details", "NAN")
                    if kernel_valid(cur_kernel) and kernel_valid(pre_kernel):
                        try:
                            cur_kernel = fix_kernel_str(cur_kernel.strip())
                            pre_kernel = fix_kernel_str(pre_kernel.strip())
                            if cur_kernel and pre_kernel:
                                if cur_kernel != pre_kernel:
                                    self.result[i]["kernel_diff"] = "different_kernel"
                                else:
                                    self.result[i]["kernel_diff"] = "same_kernel"
                            else:
                                self.result[i]["kernel_diff"] = "invalid"
                        except Exception as e:
                            self.result[i]["kernel_diff"] = "invalid"
                    else:
                        self.result[i]["kernel_diff"] = "invalid"

                    i += 1
                    j += 1

    def getSummary(self):
        assert self.result is not None
        insufficient = {}
        success_log = ""
        failed = {}
        ops = {}
        # kernel_details = {}
        for res in self.result:
            if "case_path" not in res:
                res["case_path"] = res["log"]
            if "PASS" not in res["result"]:
                if (
                    "ERROR_FAULT" not in res["result"]
                    and res["case_path"] not in failed
                ):
                    failed[res["case_path"]] = 1
                    continue
                if res["case_path"] not in insufficient:
                    insufficient[res["case_path"]] = 1
            else:
                success_log = os.path.realpath(res["log"])
        cur_result = (
            self.perf_result
            if self.perf_result is not None
            else self.getLogResult("perf")
        )
        tot_cases_len = len(cur_result)
        # op_name : op obj
        for i, res in enumerate(cur_result):
            cur_case_times = res["times"] if "times" in res else 1
            hardware_time = (
                res["hardware_time(ms)"] if "hardware_time(ms)" in res else "NAN"
            )
            if hardware_time != "NAN":
                hardware_time *= cur_case_times
            if "op_name" in res:
                op = res["op_name"]
                cur_kernels = res["kernel_details"] if "kernel_details" in res else []
                ker_objs = []
                for kernel in cur_kernels:
                    ker_objs.append(
                        KernelDetails(
                            kernel_details=kernel, kernel_count=cur_case_times
                        )
                    )

                if op not in ops:
                    cur_op_obj = OpDetails(
                        op_name=op,
                        op_cases=cur_case_times,
                        hardware_time=hardware_time,
                        kernel=ker_objs,
                    )
                    ops[op] = cur_op_obj
                else:
                    ops[op].op_add(
                        op_cases=cur_case_times,
                        hardware_time=hardware_time,
                        kernel=ker_objs,
                    )

                # if op not in kernel_details:
                #     kernel_details[op] = {"hardware_time(ms)": 0, "op_count": 0}
                # kernel_details[op]["op_count"] += cur_case_times
                # if (
                #     (
                #         isinstance(hardware_time, str)
                #         and not hardware_time.strip().isdigit()
                #     )
                #     or hardware_time <= 0
                #     or kernel_details[op]["hardware_time(ms)"] == "NAN"
                # ):
                #     kernel_details[op]["hardware_time(ms)"] = "NAN"
                # else:
                #     kernel_details[op]["hardware_time(ms)"] += hardware_time
                # if "kernel_details" in res:
                #     for kernel in res["kernel_details"]:
                #         if kernel not in kernel_details[op]:
                #             kernel_details[op][kernel] = [0, 0, []]
                #         kernel_details[op][kernel][0] += cur_case_times
                #         if "case_path" not in res:
                #             res["case_path"] = res["log"]
                #         kernel_details[op][kernel][2].append(res["case_path"])
                # if (isinstance(hardware_time, str) and not hardware_time.strip().isdigit()) or hardware_time <= 0 or kernel_details[op][kernel][1] == "NAN":
                #     kernel_details[op][kernel][1] = "NAN"
                # else:
                #     kernel_details[op][kernel][1] += hardware_time
            # if op not in op_name:
            #     if (isinstance(hardware_time, str) and not hardware_time.strip().isdigit()) or hardware_time <= 0:
            #         op_name[op] = [1, "NAN"]
            #     else:
            #         op_name[op] = [1, hardware_time]
            # else:
            #     op_name[op][0] +=1
            #     if (isinstance(hardware_time, str) and not hardware_time.strip().isdigit()) or hardware_time <= 0 or op_name[op][1] == "NAN":
            #         op_name[op][1] = "NAN"
            #     else:
            #         op_name[op][1] = hardware_time + op_name[op][1]
        self.summary = [
            {
                "tot_cases": tot_cases_len,
                "tot_ops": len(ops),
                "tot_time(s)": self.test_time,
                "tot_passed": tot_cases_len - len(failed) - len(insufficient),
                "tot_insufficent": len(insufficient),
                "tot_failed": len(failed),
            }
        ]
        if len(success_log) > 0:
            with open(success_log, "r") as f:
                lines = f.readlines()
            version_list = {
                "device": 1,
                "spe clock (MHz)": 1,
                "sdaadriver": 1,
                "sdaart": 1,
                "tecoal": 1,
                "tecoblas": 1,
                "tecocustom": 1,
                "tecotest": 1,
            }
            for i in range(1, 10):
                if i >= len(lines):
                    continue
                version = lines[i][: lines[i].find(":")].strip()
                if version in version_list:
                    line = lines[i][lines[i].find(":") + 1 :].strip()
                    self.summary[0][version] = line
                    if version == "device":
                        if self.gids:
                            self.summary[0][version] = self.gids
                    if version == "spe clock (MHz)":
                        if self.gids:
                            if spes := get_spe_from_cmd(self.gids):
                                self.summary[0][version] = ",".join(spes)
                    continue
        self.failed_casepath = [{"case_path": path} for path in failed]
        if ops:
            self.op_details = []
            self.kernel_details = []
            for op, op_obj in ops.items():
                self.op_details.append(
                    {
                        "op_name": op,
                        "op_cases": op_obj.op_cases,
                        "tot_hardware_time(ms)": op_obj.tot_hardware_time,
                    }
                )
                op_kernels = op_obj.kernels if op_obj.kernels else {}
                for kernel, ker_obj in op_kernels.items():
                    self.kernel_details.append(
                        {
                            "op_name": op,
                            "kernel_details": kernel,
                            "kernel_count": ker_obj.kernel_count,
                        }
                    )

        # if len(kernel_details) > 0:
        #     for op in kernel_details.keys():
        #         op_data = {}
        #         op_data["op_name"] = op
        #         tot_count = 0
        #         for kernel in kernel_details[op].keys():
        #             if kernel == "hardware_time(ms)" or kernel == "op_count":
        #                 continue
        #             cur_data = {}
        #             cur_data["op_name"] = op
        #             cur_data["kernel_details"] = kernel
        #             cur_data["kernel_count"] = kernel_details[op][kernel][0]
        #             sum1[2].append(cur_data)
        #         op_data["op_cases"] = kernel_details[op]["op_count"]
        #         op_data["tot_hardware_time(ms)"] = kernel_details[op][
        #             "hardware_time(ms)"
        #         ]
        #         sum1[3].append(op_data)
        # for key, val in op_name.items():
        #     cur_data = {}
        #     cur_data["op_name"] = key
        #     cur_data["op_cases"] = val[0]
        #     cur_data["tot_hardware_time(ms)"] = val[1]
        #     sum1.append(cur_data)
        # return sum1

    def setTestInfo(self, test_info):
        pass

    # read result form dir
    def readResult(self, log_path):
        pass

    def to_excel(self, file_path="./result.xlsx"):
        data = pd.DataFrame(self.format())
        data.to_excel(file_path)

    def __str__(self):
        return self.result.__str__()
