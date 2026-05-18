import json
import time
import datetime
import requests
import pandas as pd


class GetDbBaseline:
    def __init__(
        self,
    ):
        self.baseline_perf = []
        self.baseline_accu = []

    def get_baseline(
        self,
    ):
        url = "http://tecode.tecorigin.net/api/testingCase/tecoal/listCase"
        data = {}
        res = requests.post(url=url, json=data).json()
        if res["code"] == 1000:
            print("get cases success")
            data_list = res["data"]["list"]
            for case in data_list:
                if case.get("baseline_data", ""):
                    self.get_case_baseline_data(
                        case, self.baseline_perf, self.baseline_accu
                    )
        else:

            print("failed cause ", res["msg"])

    def get_case_baseline_data(self, case, baseline_data_perf, baseline_data_accu):
        cur_baseline_data = eval(case["baseline_data"])
        cur_perf_data, cur_accu_data = (
            cur_baseline_data["performance"],
            cur_baseline_data["precision"],
        )
        cur_dict = {
            "case_path": case["case_path"],
            "op_name": case["op_name"],
            "shape": case["shape"],
            "dtype": case["dtype"],
            "layout": case["layout"],
        }
        # cur_prototxt = eval(case["prototxt"])
        # if cur_prototxt.get("dnn_param", ""):
        #     cur_dict["param"] = case["prototxt"].get("dnn_param", "")
        # elif cur_prototxt.get("blas_param", ""):
        #     cur_dict["param"] = case["prototxt"].get("blas_param", "")
        # elif cur_prototxt.get("custom_param", ""):
        #     cur_dict["param"] = case["prototxt"].get("custom_param", "")

        # get perf data
        perf_dict = {k: v for k, v in cur_dict.items()}
        perf_dict["hardware_time(ms)"] = cur_perf_data["hardware_time"]
        perf_dict["minmax_hardware_time(ms)"] = [
            cur_perf_data["min_hardware_time"],
            cur_perf_data["max_hardware_time"],
        ]
        perf_dict["interface_time(ms)"] = cur_perf_data.get("interface_time", -1)
        perf_dict["cache_miss_details"] = cur_perf_data["cache_miss_details"]
        baseline_data_perf.append(perf_dict)

        for output in cur_accu_data:
            cur_accu_dict = {k: v for k, v in cur_dict.items()}
            cur_accu_dict["output"] = output
            # 兼容tecotest中的diff
            cur_accu_dict["DIFF3_MAX_TECO"] = [
                -1,
                -1,
                -1,
                cur_accu_data[output]["DIFF3_MAX"],
            ]
            cur_accu_dict["DIFF3_MEAN_TECO"] = [
                -1,
                -1,
                -1,
                cur_accu_data[output]["DIFF3_MEAN"],
            ]
            # cur_accu_dict["result"] = cur_accu_data[output].get("result", "")
            cur_accu_dict["result_hash"] = cur_accu_data[output].get("hash", "")
            baseline_data_accu.append(cur_accu_dict)

    def to_excel(self, save_name):
        if not save_name:
            save_name = "baseline-" + time.strftime(
                "%Y_%m_%d_%H_%M_%S", time.localtime()
            )
        if not save_name.endswith(".xlsx"):
            save_name += ".xlsx"
        with pd.ExcelWriter(f"./{save_name}", engine="openpyxl") as writer:
            if self.baseline_accu:
                baseline_accu = pd.DataFrame(self.baseline_accu)
                baseline_accu.to_excel(writer, sheet_name="accu", index=False)
            if self.baseline_perf:
                baseline_perf = pd.DataFrame(self.baseline_perf)
                baseline_perf.to_excel(writer, sheet_name="perf", index=False)


class GetDbResult:
    def __init__(
        self,
        op_name="all",
        test_type=None,
        model_name=None,
        model_version=None,
        shape=None,
        tecoal=None,
        tecoblas=None,
        tecocustom=None,
        spe_clock=None,
        os=None,
        date=None,
    ):
        self.op_name = op_name if op_name != "all" else None
        self.test_type = test_type
        self.model_name = model_name
        self.model_version = model_version
        self.shape = shape
        self.tecoal = tecoal
        self.tecoblas = tecoblas
        self.tecocustom = tecocustom
        self.os = os
        self.date = date
        self.spe_clock = spe_clock
        self.result_accu = []
        self.resul_perf = []
        self.task_id = []

    def to_dict(self):
        return {
            attr: getattr(self, attr)
            for attr in dir(self)
            if not callable(getattr(self, attr))
            and not attr.startswith("__")
            and getattr(self, attr)
        }

    def get_task_id(
        self,
    ):
        del_keys = ["shape", "date", "op_name"]
        data = {k: v for k, v in self.to_dict().items() if k not in del_keys}
        url = "http://tecode.tecorigin.net/api/testingCase/tecoal/listTask"
        res = requests.post(url=url, json=data).json()
        # print(res["data"].keys())
        # print(res["data"]["list"])
        task_list = res["data"]["list"]
        if res["code"] == 1000:
            for task in task_list:
                if self.date:
                    test_date = task["UpdatedAt"]
                    qdt = datetime.datetime.strptime(self.date, "%Y-%m-%dT%H:%M:%S%z")
                    tdt = datetime.datetime.strptime(test_date, "%Y-%m-%dT%H:%M:%S%z")
                    if (qdt - tdt) <= datetime.timedelta(days=7):
                        self.task_id.append(task["ID"])

                else:
                    self.task_id.append(task["ID"])

        else:

            print("failed cause ", res["msg"])

    def get_task_case_info(
        self,
    ):
        url = "http://tecode.tecorigin.net/api/testingCase/tecoal/getTaskCaseInfo"
        data = {}
        if self.op_name:
            data["op_name"] = self.op_name
        if self.shape:
            data["shape"] = self.shape
        model_cases = {}
        if self.model_name:
            if not self.model_version:
                print("please check model_version")
            for case in get_model_cases(
                model_name=self.model_name, model_version=self.model_version
            ):
                model_cases[case] = 1

        marked = {}
        for td in self.task_id:
            data["task_id"] = td
            res = requests.post(url=url, json=data).json()
            if res["code"] == 1000:
                data_list = res["data"]["list"]
                for d in data_list:
                    case_path = d["case_path"]
                    if not marked.get(case_path, 0):
                        if model_cases and not model_cases.get(case_path, ""):
                            continue
                        self.parse_case_info(d, self.result_accu, self.resul_perf)
                        marked[case_path] = 1

            else:
                print("failed cause ", res["msg"])

    def parse_case_info(self, case, accu_data, perf_data):
        cur_perf_data, cur_accu_data = (
            case,
            eval(case["precision"]),
        )
        cur_dict = {
            "case_path": case["case_path"],
            "op_name": case["op_name"],
            "shape": case["shape"],
            "dtype": case["dtype"],
            "layout": case["layout"],
        }
        perf_dict = {k: v for k, v in cur_dict.items()}
        perf_dict["hardware_time(ms)"] = cur_perf_data["hardware_time"]
        perf_dict["minmax_hardware_time(ms)"] = [
            cur_perf_data["min_hardware_time"],
            cur_perf_data["max_hardware_time"],
        ]
        perf_dict["interface_time(ms)"] = cur_perf_data.get("interface_time", -1)
        perf_dict["cache_miss_details"] = cur_perf_data["cache_miss_details"]
        perf_data.append(perf_dict)

        for output in cur_accu_data:
            cur_accu_dict = {k: v for k, v in cur_dict.items()}
            cur_accu_dict["output"] = output
            # 兼容tecotest中的diff
            cur_accu_dict["DIFF3_MAX_TECO"] = [
                -1,
                -1,
                -1,
                cur_accu_data[output]["DIFF3_MAX"],
            ]
            cur_accu_dict["DIFF3_MEAN_TECO"] = [
                -1,
                -1,
                -1,
                cur_accu_data[output]["DIFF3_MEAN"],
            ]
            # cur_accu_dict["result"] = cur_accu_data[output].get("result", "")
            cur_accu_dict["result_hash"] = cur_accu_data[output].get("hash", "")
            accu_data.append(cur_accu_dict)

    def to_excel(self, save_name):
        if not save_name:
            save_name = "baseline-" + time.strftime(
                "%Y_%m_%d_%H_%M_%S", time.localtime()
            )
        if not save_name.endswith(".xlsx"):
            save_name += ".xlsx"
        with pd.ExcelWriter(f"./{save_name}", engine="openpyxl") as writer:
            if self.result_accu:
                result_accu = pd.DataFrame(self.result_accu)
                result_accu.to_excel(writer, sheet_name="accu", index=False)
            if self.resul_perf:
                result_perf = pd.DataFrame(self.resul_perf)
                result_perf.to_excel(writer, sheet_name="perf", index=False)


class GetDbCases:
    def __init__(
        self,
        op_name="all",
        level=-1,
        model_name="all",
        model_version=None,
        dtype=None,
        shape=None,
        layout=None,
        case_path=None,
        hash=None,
        version=None,
        status=None,
    ):

        self.op_name = op_name if op_name != "all" else None
        self.dtype = dtype
        self.shape = shape
        self.layout = layout
        self.case_path = case_path
        self.hash = hash
        self.version = version
        self.level = level if level != -1 else None
        self.model_name = model_name if model_name != "all" else None
        self.model_version = model_version
        self.status = status
        self.baseline_data_accu = []
        self.baseline_data_perf = []

    def to_dict(self):
        return {
            attr: getattr(self, attr)
            for attr in dir(self)
            if not callable(getattr(self, attr))
            and not attr.startswith("__")
            and getattr(self, attr)
        }

    def get_cases(
        self,
        url="http://tecode.tecorigin.net/api/testingCase/tecoal/listCase",
        data={},
    ):
        if not data:
            data = self.to_dict()
        # print(url, data)
        res = requests.post(url=url, json=data).json()
        levels = {}
        if res["code"] == 1000:
            print("get cases success")
            data_list = res["data"]["list"]
            print(len(data_list))
            cases_list = []
            for case in data_list:
                levels[case["level"]] = levels.get(case["level"], 0) + 1
                cases_list.append(case["case_path"])
            yield from cases_list
        else:

            print("failed cause ", res["msg"])
            return []

    def get_model_cases(self):
        url = "http://tecode.tecorigin.net/api/testingCase/tecoal/getModelOpCase"
        data = self.to_dict()
        if not data.get("model_name", None) and (
            not data.get("level") or data.get("level") != 1
        ):
            data["level"] = 1
            return self.get_cases(data=data) if self.get_cases(data=data) else []
        else:
            return self.get_cases(url, data) if self.get_cases(url, data) else []


def get_model_cases(model_name="", model_version=""):
    gdbc = GetDbCases(model_name=model_name, model_version=model_version)
    cases_list = gdbc.get_model_cases()
    if cases_list:
        yield from cases_list
    else:
        return []


def get_cases(
    op_name="",
    level=-1,
    dtype=None,
    shape=None,
    layout=None,
    case_path=None,
    hash=None,
    version=None,
    status=None,
):
    # print(level, op_name)
    gdbc = GetDbCases(
        op_name=op_name,
        level=level,
        dtype=dtype,
        shape=shape,
        layout=layout,
        case_path=case_path,
        hash=hash,
        version=version,
        status=status,
    )
    cases_list = gdbc.get_cases()
    if cases_list:
        yield from cases_list
    else:
        return []


def get_weekly_cases():
    weekly_cases = get_cases()
    if weekly_cases:
        yield from weekly_cases
    else:
        return []


def get_daily_cases(op_name=""):
    daily_cases = []
    levels = [i for i in range(1, 5)]
    # print(op_name)
    for cur_level in levels:
        # print(op_name, cur_level)
        cases_list = get_cases(op_name=op_name, level=cur_level)
        if cases_list:
            daily_cases.extend(cases_list)
    if daily_cases:
        yield from daily_cases
    else:
        return []


def get_pr_cases(op_name):
    cases_list = get_daily_cases(op_name)
    if cases_list:
        yield from cases_list
    else:
        return []


def get_baseline(save_name):
    gdbase = GetDbBaseline()
    gdbase.get_baseline()
    gdbase.to_excel(save_name)


def get_result(
    save_name=None,
    op_name="all",
    test_type=None,
    date=None,
    model_name=None,
    model_version=None,
    shape=None,
    tecoal=None,
    tecoblas=None,
    tecocustom=None,
    spe_clock=None,
    os=None,
):
    # gdr = GetDbResult(date="2024-04-16T19:00:00+08:00")
    gdr = GetDbResult(
        op_name=op_name,
        date=date,
        test_type=None,
        model_name=model_name,
        model_version=model_version,
        shape=shape,
        tecoal=tecoal,
        tecoblas=tecoblas,
        tecocustom=tecocustom,
        spe_clock=spe_clock,
        os=os,
    )
    gdr.get_task_id()
    gdr.get_task_case_info()
    gdr.to_excel(save_name=save_name)


get_result(save_name="querry", date="2024-04-16T19:00:00+08:00")
