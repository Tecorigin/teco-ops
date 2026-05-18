import os
import sys
import json
import copy
import subprocess
import time
import re


def fix_kernel_str(kernels):
    if "[" in kernels:
        kernels = kernels[1:-1]
    kernels = [
        kernel.strip()
        for kernel in kernels.split("'")
        if kernel.strip() and "teco" in kernel
    ]
    for i, kernel in enumerate(kernels):
        idxr = kernel.rfind("IDF")
        idxl = kernel.find("teco")
        kernel = kernel[idxl:idxr] if idxr != -1 else kernel[idxl:]
        kernels[i] = kernel
    return kernels


def grepLogError(file_path, lines):
    results = []
    result = {}
    result["log"] = file_path
    # case_path = os.popen(f'grep "case_path" {file_path}').read()
    tot = 0
    idxr = file_path.rfind("/")
    idxl = file_path[:idxr].rfind("/")
    result["op_name"] = file_path[idxl + 1 : idxr]
    # result["record_id"] = count[0]
    # count[0] += 1
    unknown_error_flag = False
    error_pattern = r"\berror\b"
    for i in range(len(lines)):
        line = lines[i].strip()
        if "case_path" in line:
            idx = line.find("/")
            result["case_path"] = line[idx:]
            # print(result["case_path"])
            continue
        if "StreamSynchronize" in line:
            result["result"] = "ERROR_SdaaStreamSynchronize"
            break
        if "Only found result computed on" in line:
            result["result"] = "ERROR_OnlyFoundResultOn"
            break
        if "device id" in line:
            result["result"] = "ERROR_DeviceID"
            break
        if "out of memory" in line:
            result["result"] = "ERROR_MemOutOfRange"
            break
        if "Total 0 cases of 0 op(s)." in line:
            result["result"] = "ERROR_OpName"
            break
        if "File not found" in line:
            result["result"] = "ERROR_PrototxtNotFound"
            break
        if "core dumped" in line:
            result["result"] = "ERROR_CoreDump"
            break
        if "Exception type" in line and "actsPrintExceptionInfo" in line:
            result["result"] = line.split(":")[-1].strip()
            break
        if re.search(error_pattern, line, re.IGNORECASE):
            unknown_error_flag = True
    # if len(case_path) > 0:
    #     case_path = case_path.strip()
    #     idx = case_path.find('/')
    #     case_path = case_path[idx:]
    #     result["case_path"] = case_path

    # sdaaStreamSynchronize_cmd = f'grep "sdaaStreamSynchronize" {file_path}'
    # sdaaStreamSynchronize_err = subprocess.run(sdaaStreamSynchronize_cmd,
    #                                            shell=True,
    #                                            capture_output=True,
    #                                            text=True)
    # if sdaaStreamSynchronize_err.returncode == 0:
    #     result["result"] = "ERROR_SdaaStreamSynchronize"

    # only_found_result_on_cmd = f'grep "Only found result computed on" {file_path}'
    # only_found_result_on_cmd_err = subprocess.run(only_found_result_on_cmd,
    #                                               shell=True,
    #                                               capture_output=True,
    #                                               text=True)
    # if only_found_result_on_cmd_err.returncode == 0:
    #     result["result"] = "ERROR_OnlyFoundResultOn"

    # grep_gid_cmd = f'grep "device id" {file_path}'
    # grep_gid_cmd_error = subprocess.run(grep_gid_cmd,
    #                                     shell=True,
    #                                     capture_output=True,
    #                                     text=True)
    # if grep_gid_cmd_error.returncode == 0:
    #     result["result"] = "ERROR_DeviceID"

    # grep_deviceFree_cmd = f'grep "deviceFree" {file_path}'
    # grep_deviceFree_error = subprocess.run(grep_deviceFree_cmd,
    #                                        shell=True,
    #                                        capture_output=True,
    #                                        text=True)
    # if grep_deviceFree_error.returncode == 0:
    #     result["result"] = "ERROR_MemOutOfRange"

    # grep_opName_cmd = f'grep "Running 0 tests from 0 test suites" {file_path}'
    # grep_opName_cmd_error = subprocess.run(grep_opName_cmd,
    #                                        shell=True,
    #                                        capture_output=True,
    #                                        text=True)
    # if grep_opName_cmd_error.returncode == 0:
    #     result["result"] = "ERROR_OpName"
    if "result" not in result:
        if unknown_error_flag:
            result["result"] = "UNKOWN_ERROR"
        else:
            result["result"] = "ERROR_Timeout"
    results.append(result)
    return results


def updateDictKeys(dict_nlayer):
    res = []
    for k, v in dict_nlayer.items():
        for k1, v1 in v.items():
            key = f"{k}_{k1}"
            # print(key)
            val = []
            for k2, v2 in v1.items():
                val.append(v2)
            res.append((key, val))
    return res


def grepFailedLog(lines):
    # with open(log_path, 'r') as f:
    #     lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i].strip()
        if "File not found" in line:
            return "ERROR_PrototxtNotFound"
        if "Error parsing text-format testpt.Node" in line:
            return "ERROR_Parsing"
        if "num is wrong" in line:
            return "ERROR_TensorNum"
    # grep_nan_cmd = f'grep "Only found result computed on teco output\[[0-9]\+\] is NaN, failed" {log_path}'
    # grep_nan_cmd_error = subprocess.run(grep_nan_cmd,
    #                                     shell=True,
    #                                     capture_output=True,
    #                                     text=True)
    # if grep_nan_cmd_error.returncode == 0:
    #     return "ERROR_ExistsNaN"

    # grep_inf_cmd = f'grep "Only found result computed on teco dx\[[0-9]\+\] is Inf, failed" {log_path}'
    # grep_inf_cmd_error = subprocess.run(grep_inf_cmd,
    #                                     shell=True,
    #                                     capture_output=True,
    #                                     text=True)
    # if grep_inf_cmd_error.returncode == 0:
    #     return "ERROR_ExistsInf"

    # grep_FileExists_cmd = f'grep "File not found" {log_path}'
    # grep_FileExists_cmd_error = subprocess.run(grep_FileExists_cmd,
    #                                            shell=True,
    #                                            capture_output=True,
    #                                            text=True)
    # if grep_FileExists_cmd_error.returncode == 0:
    #     return "ERROR_PrototxtNotFound"

    return ""


def grep_host_mem(lines):
    for i in range((len(lines)) - 1, -1, -1):
        line = lines[i].strip()
        if "memory leak has occurred" in line:
            return "ERROR_HostMemLeak"
    return "Success"


def grepMseRela(cur_res, st):
    try:
        if cur_res["MSE_RELA_TECO"][-1] == 0.0:
            return "PASS_S1"
        if cur_res["MSE_RELA_TECO"][-1] <= st:
            return "PASS_S2"
        return "ERROR_FAULT"
    except Exception as e:
        return "ERROR_FAULT"


def grepDeviLog(cur_res, st2, st3, st4, st5, dtype=float):
    try:
        if "DIFF3_MAX_TECO" not in cur_res or "DIFF3_MEAN_TECO" not in cur_res:
            if "MSE_RELA_TECO" in cur_res:
                return grepMseRela(cur_res, 0.025)
            else:
                return "not support"
        if cur_res["DIFF3_MAX_TECO"][-1] == 0.0:
            return "PASS_S1"
        assert st2 < st3
        assert st4 < st5
        # st3 = cur_res["DIFF3_MAX_THRESHOLD"][0] if "DIFF3_MAX_THRESHOLD" in cur_res else 10.0
        # st3_MEAN = (
        #     cur_res["DIFF3_MEAN_THRESHOLD"][0] if "DIFF3_MEAN_THRESHOLD" in cur_res else 10.0
        # )
        # print(st3, st3_MEAN)
        deviation_max = cur_res["DIFF3_MAX_THRESHOLD"][-1]
        deviation_mean = cur_res["DIFF3_MEAN_THRESHOLD"][-1]
        if deviation_max <= st2 and deviation_mean <= st2:
            return "PASS_S2"
        if deviation_max <= st3 and deviation_mean <= st3:
            return "PASS_S3"
        if cur_res["DIFF3_MAX_TECO"][-1] < st4:
            return "PASS_S4"
        if cur_res["DIFF3_MAX_TECO"][-1] < st5:
            return "PASS_S5"
        if cur_res["DIFF3_MEAN_TECO"][-1] < st5:
            return "PASS_S6"
        return "ERROR_FAULT"
    except Exception as e:
        return "ERROR_FAULT"


def dealResultFromStr(out, lines=[]):
    results = []
    # print(len(json.loads(out)))
    for record in json.loads(out):
        if "diffs" not in record:
            results.append(record)
            continue

        diffs = record["diffs"]

        output_name = set()
        for mode in diffs.keys():
            output_name.update(diffs[mode].keys())
        output_name = list(output_name)

        output_result = [{"output": op, "diff": {}} for op in output_name]

        for mode in diffs.keys():
            for op in diffs[mode].keys():
                output_result[output_name.index(op)]["diff"].update(
                    {mode: diffs[mode][op]}
                )

        for tmp in output_result:
            case = copy.deepcopy(record)
            case.pop("diffs")
            case.update(tmp)
            for key, val in updateDictKeys(tmp["diff"]):
                case[key] = val
            case[f"output"] = tmp["output"]
            results.append(case)

    for cur_res in results:
        if hardware_times := cur_res.get("hardware_times(ms)", []):
            hardware_times.sort()
            min_hwtime, max_hwtime = hardware_times[0], hardware_times[-1]
            hwtime_gap = max_hwtime - min_hwtime
            try:
                hwtime_gap_percent = hwtime_gap / cur_res["hardware_time(ms)"] * 100
            except Exception:
                hwtime_gap_percent = "NAN"
            cur_res["minmax_hardware_time(ms)"] = [min_hwtime, max_hwtime]
            cur_res["minmax_hardware_time_gap(ms)"] = hwtime_gap
            cur_res["minmax_hardware_time_gap/avg(%)"] = hwtime_gap_percent
        else:
            cur_res["minmax_hardware_time(ms)"] = []
            cur_res["minmax_hardware_time_gap(ms)"] = -1
            cur_res["minmax_hardware_time_gap/avg(%)"] = -1

        # if cur_output := cur_res.get("output", None):
        #     cur_res["result_hash"] = cur_res["result_hash"][cur_output]

        # print(cur_res)
        kernel_details = cur_res["kernel_details"]
        tmp_kernel = []
        for i in range(len(kernel_details)):
            cur_kernel = kernel_details[i][0]
            cur_kernel = cur_kernel[: cur_kernel.find("(")]
            tmp_kernel.extend(fix_kernel_str(cur_kernel))
        cur_res["kernel_details"] = tmp_kernel
        if isinstance(cur_res["launch_time(ms)"], str):
            tmp_time = cur_res["launch_time(ms)"].split("/")
            if len(tmp_time) == 2:
                # print(tmp_time)
                if float(tmp_time[1]) != 0:
                    cur_res["launch_time(ms)"] = float(tmp_time[0]) / float(tmp_time[1])
        cur_res["result"] = cur_res["status"]
        if cur_res["result"] == "TEST_STATUS_EXECUTE_ERROR":
            tmp_error = grepFailedLog(lines)
            if len(tmp_error) > 0:
                cur_res["result"] = tmp_error
        elif cur_res["result"] == "TEST_STATUS_RESULT_ERROR":
            standard = grepDeviLog(cur_res, 2.0, 10.0, 1e-5, 1e-3)
            cur_res["result"] = standard
        elif cur_res["result"] == "TEST_STATUS_SUCCESS":
            standard = grepDeviLog(cur_res, 2.0, 10.0, 1e-5, 1e-3)
            if standard not in ["PASS_S1", "PASS_S2"]:
                standard = "PASS_S2"  # if tecotest result is pass, at least PASS_S2
            cur_res["result"] = standard
        cur_res["host_result"] = grep_host_mem(lines)

    return results


def dealLogResult1(json_dict, lines, log):
    # try:
    #     with open(log, 'r') as f:
    #         lines = f.readlines()
    # except UnicodeDecodeError as e:
    #     print(f"log:{log} can not be utf-8 decoded")
    #     return [{"log":log, "result":"ERROR_LogDecode"}]
    # for i in range(len(lines)):
    #     line = lines[i].strip()
    #     if "test result json" in line and i + 1 < len(lines):
    #         record = lines[i + 1].strip()
    #         idx = record.find('{')
    #         string = '[{"log":"' + log + '",' + record[idx + 1:] + ']'
    #         return dealResultFromStr(string, log)
    if len(json_dict) > 0:
        record = json_dict.strip()
        idx = record.find("{")
        string = '[{"log":"' + log + '",' + record[idx + 1 :] + "]"
        # count[0] += 1
        # print(string)
        return dealResultFromStr(string, log)
    return grepLogError(log, lines)


def dealLogResult(cmd, log):
    try:
        with open(log, "r") as f:
            lines = f.readlines()
    except UnicodeDecodeError as e:
        print(f"log:{log} can not be utf-8 decoded")
        return [{"log": log, "result": "ERROR_LogDecode"}]
    # count ,index = 0, -1
    result = []
    kernels, test_json = [], []
    idx, cnt = -1, 0
    # kernels_name = {}
    for i in range(len(lines)):
        line = lines[i].strip()
        # if "TestOpFixture.kernel" in line:
        #     if "RUN" in line:
        #         cur_kernel = lines[i][lines[i].find("/"):].strip()
        #     else:
        #         cur_kernel = line[line.find("/"):line.find('(')].strip() if "OK" in line else line[line.find("/"):line.find(',')].strip()
        #     print(cur_kernel)
        #     kernels_name[cur_kernel] = 1 if cur_kernel not in kernels_natme.keys() else kernels_name[cur_kernel]+1
        #     if kernels_name[cur_kernel] == 1:
        #         kernels.append([i])
        #     elif kernels_name[cur_kernel] == 2:
        #         kernels[-1].append(i)
        if "RUN" in line:
            cnt += 1
            if cnt > 1:
                kernels.append([idx, i])
            idx = i
        if "test result json" in line and i + 1 < len(lines):
            test_json.append(i + 1)
    # 没有考虑到边界问题，比如找不到匹配项，对于该种情况应该返回全部日志
    if idx != -1:
        kernels.append([idx, len(lines)])
    # print(len(kernels))
    # print(len(test_json))
    # assert(len(kernels) >= len(test_json))
    i, j = 0, 0
    while i < len(kernels) and j < len(test_json):
        # if len(kernels[i]) > 1 and test_json[j] > kernels[i][0] and test_json[j] < kernels[i][1]:
        #     result.extend(dealLogResult1(lines[test_json[j]], lines[kernels[i][0]:kernels[i][1]+1], log))
        # i += 1
        # j += 1
        # if len(kernels[i]) > 1:
        if test_json[j] > kernels[i][0] and test_json[j] < kernels[i][1]:
            result.extend(
                dealLogResult1(
                    lines[test_json[j]], lines[kernels[i][0] : kernels[i][1] + 1], log
                )
            )
            i += 1
            j += 1
        elif test_json[j] < kernels[i][0]:
            # print(kernels[i], test_json[j])
            j += 1
        elif test_json[j] > kernels[i][1]:
            # print(kernels[i], test_json[j])
            result.extend(
                dealLogResult1("", lines[kernels[i][0] : kernels[i][1] + 1], log)
            )
            i += 1
        # else:
        #     data = lines[kernels[i][0]:kernels[i+1][0]] if i+1 < len(kernels) else lines[kernels[i][0]:]
        #     result.extend(grepLogError(log, data))
        #     i += 1
    if i < len(kernels):
        for k in range(i, len(kernels)):
            data = (
                lines[kernels[k][0] : kernels[k + 1][0]]
                if k + 1 < len(kernels)
                else lines[kernels[k][0] :]
            )
            result.extend(grepLogError(log, data))
    if len(kernels) == 0:
        result.extend(grepLogError(log, lines))
    return result

    # record = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    # if record.returncode == 0:
    #     record = record.stdout.strip()
    #     idx = record.find('{')
    #     string = '[{"log":"' + log + '",' + record[idx + 1:] + ']'
    #     return dealResultFromStr(string, log)
    # else:
    # return grepLogError(log)


# support file/dir
def dealResultFromLog(log):
    result = []
    if os.path.isfile(log):
        # print("isfile")
        log_cmd = f'grep "al_type" {log}'
        result.extend(dealLogResult(log_cmd, log))
        return result

    # log is dir
    # for record in os.popen(
    #'grep -r "al_type" {}'.format(log)).read().strip().split('\n'):

    cmd = f'find {log} -name "*.log"'
    for cur_log_path in os.popen(cmd).read().strip().split("\n"):
        cur_cmd = f'grep "al_type" {cur_log_path}'
        cur_result = dealLogResult(cur_cmd, cur_log_path)
        result.extend(cur_result)
    return result


def merge_log(log_path, save_name):
    all_log_name = save_name + ".log"
    all_log_name = os.path.join(log_path, all_log_name)
    # if os.path.exists(all_log_name):
    #     os.remove(all_log_name)
    with open(all_log_name, "w") as fout:
        for op_name in os.listdir(log_path):
            op_path = os.path.join(log_path, op_name)
            if os.path.isfile(op_path):
                continue
            for log_name in os.listdir(op_path):
                log = os.path.join(op_path, log_name)
                if os.path.isfile(log) and log.endswith(".log"):
                    with open(log, "r") as fin:
                        try:
                            fout.write("----------------------------\n")
                            fout.write(f"log: {log}\n")
                            fout.write(fin.read())
                        except Exception as e:
                            fout.write(str(e) + "\n")
                            fout.write(log)
                        fout.write("\n")


# to dict, list, dir/file or str
def getResult(input):
    if isinstance(input, dict):
        return [input]

    if isinstance(input, list):
        result = []
        for i in input:
            result.extend(getResult(i))
        return result

    if isinstance(input, str):
        if os.path.exists(input):
            # print(f"{input} exists")
            return dealResultFromLog(input)
        else:
            print(f"{input} not exists")
            return dealResultFromStr(input)
