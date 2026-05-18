import os, sys, json
import logging
import pandas as pd
from case_parser import ProtoParser

USE_DB = False
if USE_DB:
    from model_db import Task, TaskCase
else:
    from model import Task, TaskCase

"""
1. 只有结果是 PASS_S* 的case才参与精度和性能的比较, 异常数据没有影响
2. 异常数据, 导致原本是float或int类型的数据, 变成NAN或空字符串, 在接收json数据并进行解析时会出错, 故而这里需要先预处理成-1
"""


def read_sheet(excel_path, sheet_name):
    if not os.path.exists(excel_path):
        return []
    df_sheet = pd.read_excel(excel_path, sheet_name).fillna("")  # nan -> ""
    data = []
    for index, row in df_sheet.iterrows():
        data_dict = {col: val for col, val in row.items()}
        data.append(data_dict)
    return data


def readFromExcel(path):
    if os.path.isfile(path):
        file = pd.ExcelFile(path)
        sheet_names = file.sheet_names
        accu_data = read_sheet(path, "accu")
        perf_data = read_sheet(path, "perf")
        version_data = read_sheet(path, "summary")
    else:
        logging.error(f"{path} not exists")
    return accu_data, perf_data, version_data[0]


def wrap_perf_data(data):
    wrap_data = {}
    for d in data:
        if "PASS" not in d["result"]:
            continue
        case_path = d["case_path"]
        wrap_data[case_path] = d
    return wrap_data


def wrap_accu_data(data):
    wrap_data = {}
    for d in data:
        case_path = d["case_path"]
        if case_path not in wrap_data:
            wrap_data[case_path] = d
        tensor_name = d["output"]
        tensor_hash = ""
        if "result_hash" in d:
            tensor_hash = d["result_hash"]

        diff3_max, diff3_mean = -1, -1
        if d["DIFF3_MAX_TECO"] != "NAN":
            diff3_max = eval(d["DIFF3_MAX_TECO"])[-1]
        if d["DIFF3_MEAN_TECO"] != "NAN":
            diff3_mean = eval(d["DIFF3_MEAN_TECO"])[-1]
        if "precision" in wrap_data[case_path]:
            # print(wrap_data[case_path]["precision"])
            # print(type(wrap_data[case_path]["precision"]))
            wrap_data[case_path]["precision"][tensor_name] = {
                "DIFF3_MAX": diff3_max,
                "DIFF3_MEAN": diff3_mean,
                "hash": tensor_hash,
            }

        else:
            wrap_data[case_path]["precision"] = {
                tensor_name: {
                    "DIFF3_MAX": diff3_max,
                    "DIFF3_MEAN": diff3_mean,
                    "hash": tensor_hash,
                }
            }
        diff3_max_gpu, diff3_mean_gpu = -1, -1
        if d["DIFF3_MAX_GPU"] != "NAN":
            diff3_max_gpu = eval(d["DIFF3_MAX_GPU"])[-1]
        if d["DIFF3_MEAN_GPU"] != "NAN":
            diff3_mean_gpu = eval(d["DIFF3_MEAN_GPU"])[-1]
        if "gpu_precision" in wrap_data[case_path]:
            wrap_data[case_path]["gpu_precision"][tensor_name] = {
                "DIFF3_MAX": diff3_max_gpu,
                "DIFF3_MEAN": diff3_mean,
            }

        else:
            wrap_data[case_path]["gpu_precision"] = {
                tensor_name: {"DIFF3_MAX": diff3_max_gpu, "DIFF3_MEAN": diff3_mean_gpu}
            }

    return wrap_data


def deal_version_data(version_data, test_type, os_env="unknown"):
    dealed_version_data = {}
    keys = [
        "os",
        "spe clock (MHz)",
        "sdaadriver",
        "sdaart",
        "tecoal",
        "tecoblas",
        "tecocustom",
        "tecotest",
    ]
    for key in keys:
        dealed_version_data[key] = "unknown"
    for key in version_data:
        if key in keys:
            dealed_version_data[key] = str(version_data[key]).split(" ")[0]
    dealed_version_data["os"] = os_env
    dealed_version_data["test_type"] = test_type
    dealed_version_data["spe_clock"] = dealed_version_data["spe clock (MHz)"]
    return dealed_version_data


# some digit maybe NAN or str ,need convert to digit
# some data format
def standardize_data(data):
    keys = [
        "interface_time(ms)",
        "hardware_time(ms)",
        "launch_time(ms)",
        "io_bandwidth(GB/s)",
        "theory_ios(Bytes)",
        "compute_force(TFLOPS)",
        "theory_ops(Ops)",
        "minmax_hardware_time(ms)",
        "minmax_hardware_time_gap(ms)",
        "minmax_hardware_time_gap/avg(%)",
    ]
    for key in keys:
        if data[key] == "NAN" or data[key] == "":
            data[key] = -1
        new_key = key[0 : key.find("(")]
        data[new_key] = data[key]
        del data[key]
    try:
        min_hw_time, max_hw_time = eval(data["minmax_hardware_time"])
    except Exception:
        min_hw_time, max_hw_time = -1, -1
    data["min_hardware_time"] = min_hw_time
    data["max_hardware_time"] = max_hw_time
    data["minmax_hardware_time_gap"] = data["minmax_hardware_time_gap"]
    data["relative_minmax_hardware_time_gap"] = data["minmax_hardware_time_gap/avg"]
    data["count"] = data["times"]
    del data["times"]

    # # in order to keep same as CaseParser(ProtoParser)
    # for key in ["shape", "layout", "dtype"]:
    #     if len(data[key]) >= 2:
    #         data[key] = data[key][1:-1]
    #         data[key] = data[key].replace("'", "").replace(" ", "")

    pp = ProtoParser(data["case_path"])
    op_name, dtype, shape, layout, prototxt, case_path, hash = pp.run()
    data["dtype"] = dtype
    data["shape"] = shape
    data["layout"] = layout
    data["hash"] = hash

    return data


def wrap_data(wrapped_perf_data, wrapped_accu_data, dealed_version_data):
    detail_data = []
    for case_path in wrapped_perf_data:
        data = {}
        for key in wrapped_perf_data[case_path]:
            data[key] = wrapped_perf_data[case_path][key]
        data["precision"] = wrapped_accu_data[case_path]["precision"]
        data["gpu_precision"] = wrapped_accu_data[case_path]["gpu_precision"]
        data["param"] = wrapped_accu_data[case_path]["param"]
        if "compute_force(op/s)" in data:
            data["compute_force(TFLOPS)"] = data["compute_force(op/s)"]
            del data["compute_force(op/s)"]

        detail_data.append(standardize_data(data))

    result = {"env_data": dealed_version_data, "case_data": detail_data}
    return result


class ResultParser:

    def __init__(self, filename, test_type, os_env):
        self.filename = filename
        self.test_type = test_type
        self.os_env = os_env

    def parse(self):
        accu_data, perf_data, version_data = readFromExcel(self.filename)
        wrapped_accu_data = wrap_accu_data(accu_data)
        wrapped_perf_data = wrap_perf_data(perf_data)
        dealed_version_data = deal_version_data(
            version_data, self.test_type, self.os_env
        )
        result = wrap_data(wrapped_perf_data, wrapped_accu_data, dealed_version_data)
        return result

    def run(self):
        result = self.parse()
        env_data = result["env_data"]
        case_data = result["case_data"]
        print(env_data)
        print(case_data[0])

        task = Task()
        for key, value in env_data.items():
            if hasattr(task, key):
                setattr(task, key, value)
        print("end env items")
        task_cases = []
        for data in case_data:
            task_case = TaskCase()
            for key, value in data.items():
                if hasattr(task_case, key):
                    setattr(task_case, key, value)
            task_cases.append(task_case)
        print("end task append")
        return task, task_cases


class BaselineParser:
    def __init__(self, file_name):
        self.file_name = file_name
        self.baseline_data_row = {}
        self.baseline_data_format = []

    def parse_baseline(self):
        accu_data = pd.read_excel(self.file_name, "accu")
        for _, row in accu_data.iterrows():
            casepath = row["case_path"]
            cur_case = self.baseline_data_row.get(casepath, {"case_path": casepath})
            cur_case_precision = cur_case.get("precision", {})
            output = row.get("output", "")
            cur_hash = row.get("result_hash", "")
            diff3_max, diff3_mean = -1, -1
            if row["DIFF3_MAX_TECO"] != "NAN":
                diff3_max = eval(row["DIFF3_MAX_TECO"])[-1]
            if row["DIFF3_MEAN_TECO"] != "NAN":
                diff3_mean = eval(row["DIFF3_MEAN_TECO"])[-1]
            cur_output_percision = {
                output: {
                    "DIFF3_MAX": diff3_max,
                    "DIFF3_MEAN": diff3_mean,
                    "hash": cur_hash,
                }
            }
            cur_case_precision[output] = cur_output_percision
            cur_case["precision"] = cur_case_precision
            self.baseline_data_row[casepath] = cur_case
        perf_data = pd.read_excel(self.file_name, "perf")
        for _, row in perf_data.iterrows():
            casepath = row["case_path"]
            cur_case = self.baseline_data_row.get(casepath, {"case_path": casepath})
            cur_case_performance = {}
            hardware_time = row.get("hardware_time(ms)", -1)
            if hardware_time == "NAN":
                hardware_time = -1
            try:
                min_ht, max_ht = eval(row["minmax_hardware_time(ms)"])
            except Exception as e:
                min_ht, max_ht = -1, -1
            interface_time = row.get("interface_time(ms)", -1)
            if interface_time == "NAN":
                interface_time = -1
            kernel_details = row.get("kernel_details", "")
            casemiss_details = row.get("cache_miss_details", "")
            cur_case_performance["hardware_time"] = hardware_time
            cur_case_performance["min_hardware_time"] = min_ht
            cur_case_performance["max_hardware_time"] = max_ht
            cur_case_performance["interface_time"] = interface_time
            cur_case_performance["cache_miss_details"] = casemiss_details
            cur_case_performance["kernel_details"] = kernel_details
            self.baseline_data_row[casepath]["performance"] = cur_case_performance

    def format(self):
        for k, v in self.baseline_data_row.items():
            self.baseline_data_format.append(v)

    def run(self):
        self.parse_baseline()
        self.format()


if __name__ == "__main__":
    # only model case
    filename = "data/bs16.xlsx"
    outfile = "data/bs16.json"
    rp = ResultParser(filename, test_type=1, os_env="ubuntu22.04")
    task, task_case = rp.run()
    task = task.to_dict()
    task_case = [tc.to_dict() for tc in task_case]
    res = {
        "task_data": task,
        "case_data": task_case,
    }
    with open(outfile, "w") as f:
        f.write(json.dumps(res, indent=4))

    filename = "data/bs32_150.xlsx"
    outfile = "data/bs32_150.json"
    rp = ResultParser(filename, test_type=0, os_env="ubuntu22.04")
    task, task_case = rp.run()
    task = task.to_dict()
    task_case = [tc.to_dict() for tc in task_case]
    res = {
        "task_data": task,
        "case_data": task_case,
    }
    with open(outfile, "w") as f:
        f.write(json.dumps(res, indent=4))

    filename = "data/bs32_140.xlsx"
    outfile = "data/bs32_140.json"
    rp = ResultParser(filename, test_type=0, os_env="centos7")
    task, task_case = rp.run()
    task = task.to_dict()
    task_case = [tc.to_dict() for tc in task_case]
    res = {
        "task_data": task,
        "case_data": task_case,
    }
    with open(outfile, "w") as f:
        f.write(json.dumps(res, indent=4))

    # filename = "data/bs32_150_exception.xlsx"
    # outfile = "data/bs32_150_exception.json"
    # rp = ResultParser(filename, test_type=0, os_env="centos7")
    # task, task_case = rp.run()
    # task = task.to_dict()
    # task_case = [tc.to_dict() for tc in task_case]
    # res = {
    #     "task_data": task,
    #     "case_data": task_case,
    # }
    # with open(outfile, 'w') as f:
    #     f.write(json.dumps(res, indent=4))

    # only not model case
    filename = "data/result.xlsx"
    outfile = "data/result.json"
    rp = ResultParser(filename, test_type=0, os_env="centos7")
    task, task_case = rp.run()
    task = task.to_dict()
    task_case = [tc.to_dict() for tc in task_case]
    res = {
        "task_data": task,
        "case_data": task_case,
    }
    with open(outfile, "w") as f:
        f.write(json.dumps(res, indent=4))

    # all case
    filename = "data/result.xlsx"
    rp = ResultParser(filename, test_type=0, os_env="centos7")
    task, task_case = rp.run()
    task = task.to_dict()
    task_case_all = [tc.to_dict() for tc in task_case]

    filename = "data/bs32_150.xlsx"
    rp = ResultParser(filename, test_type=0, os_env="centos7")
    task, task_case = rp.run()
    task = task.to_dict()
    task_case_all += [tc.to_dict() for tc in task_case]

    res = {
        "task_data": task,
        "case_data": task_case_all,
    }
    outfile = "data/all.json"
    with open(outfile, "w") as f:
        f.write(json.dumps(res, indent=4))
