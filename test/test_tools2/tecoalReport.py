# coding: utf-8
from tecoalResult import *
import pandas as pd
import time
from tecoalCase import *
import logging


sys.path.append(os.path.split(os.path.realpath(__file__))[0] + "/sample")


def isbug(accu_result):
    bugs = ["ERROR_FAULT"]
    if accu_result in bugs:
        return True
    return False


def accu_cmp(cur_max, bench_max, cur_mean, bench_mean):
    if (cur_max == -1) or (bench_max == -1) or (cur_mean == -1) or (bench_mean == -1):
        return "invalid"
    if (cur_max > bench_max) or (cur_mean > bench_mean):
        return "fallback"
    elif (cur_max == bench_max) and (cur_mean == bench_mean):
        return "flat"
    else:
        return "rise"


class CaseDetails:
    case_path = ""
    op_name = ""
    shape = []
    dtype = ""
    result = ""
    kernel = ""
    hardware_time = ""
    io_bandwidth = ""
    compute_force = ""
    diff3_max = ""
    diff3_mean = ""

    def __init__(
        self,
        case_path,
        op_name="",
        shape=[],
        dtype="",
        result="",
        kernel="",
        diff3_max="",
        diff3_mean="",
    ):
        if not case_path:
            return
        self.case_path = case_path
        if op_name:
            self.op_name = op_name
        if shape:
            self.shape = shape
        if dtype:
            self.dtype = dtype
        if result:
            self.result = result
        if kernel:
            self.kernel = kernel
        if diff3_max:
            try:
                self.diff3_max = float(diff3_max.split(",")[-1][:-1])
            except Exception as e:
                self.diff3_max = -1
        if diff3_mean:
            try:
                self.diff3_mean = float(diff3_mean.split(",")[-1][:-1])
            except Exception as e:
                self.diff3_mean = -1

    def update_result(self, result):
        if result and "PASS" not in result:
            self.result = result

    def update_perf(self, hardware_time, io_bandwidth, compute_force):
        self.hardware_time = hardware_time
        self.io_bandwidth = io_bandwidth
        self.compute_force = compute_force


class OpTp:
    op_name = ""
    failed_num = 0
    insufficent_num = 0
    passed_num = 0
    op_count = 0
    max_rise = -1
    max_rise_path = ""
    max_fallback_path = ""
    max_fallback = 0
    min_io_bandwidth = 1e6
    max_io_bandwidth = -1
    min_compute_force = 1e6
    max_compute_force = -1

    def __init__(self, op_name) -> None:
        self.op_name = op_name
        self.op_count = 0

    def update_count(self, result):
        self.op_count += 1
        if "PASS" not in result:
            if "ERROR_FAULT" not in result:
                self.insufficent_num += 1
            self.failed_num += 1
        else:
            self.passed_num += 1

    def update_io_compute(self, io_bandwidth, compute_force):
        if self.min_io_bandwidth > io_bandwidth:
            self.min_io_bandwidth = io_bandwidth
        if self.max_io_bandwidth < io_bandwidth:
            self.max_io_bandwidth = io_bandwidth
        if self.min_compute_force > compute_force:
            self.min_compute_force = compute_force
        if self.max_compute_force < compute_force:
            self.max_compute_force = compute_force


class TecoalReportData:
    json_data = {}
    json_file = ""
    test_data = {}
    accu_key = [
        "case_path",
        "op_name",
        "shape",
        "dtype",
        "result",
        "kernel_details",
        "DIFF3_MAX_TECO",
        "DIFF3_MEAN_TECO",
    ]
    perf_key = ["hardware_time(ms)", "io_bandwidth(GB/s)", "compute_force(op/s)"]
    test_file_path = ""
    last_day_path = ""
    last_version_path = ""
    result = None
    last_day_result = None
    last_version_result = None
    fix_op = {}
    new_bug_op = {}
    unfix_op = {}
    cur_op_details = {}
    last_day_op_details = {}
    add_op_details = {}
    tot_add_details = {}
    cmp_with_version = {}
    cmp_with_day = {}
    cur_case_dicts, last_day_case_dicts, last_version_case_dicts = {}, {}, {}
    op_cases_change = ""
    new_cases_num = 0
    delete_cases_num = 0
    (
        accu_rise_day_op,
        accu_rise_version_op,
        accu_fallback_day_op,
        accu_fallback_version_op,
    ) = ([], [], [], [])

    def __init__(
        self, test_file_path, last_day_path="", last_version_path="", json_file=""
    ):
        self.test_file_path = test_file_path
        # self.test_data = read_sheet(test_file_path, 0)[0]
        self.result = TecoalResult(test_file_path)
        if self.result and self.result.summary:
            self.test_data = self.result.summary[0]
        if last_day_path:
            self.last_day_path = last_day_path
            self.last_day_result = TecoalResult(last_day_path)
        if last_version_path:
            self.last_version_path = last_version_path
            self.last_version_result = TecoalResult(last_version_path)
        if json_file:
            self.json_file = json_file
        # self.accu_data = read_sheet(test_file_path, "accu")
        # self.perf_data = read_sheet(test_file_path, "perf")

    def init_json(self):
        pass

    # cur_result:TecoalResult obj
    # return {case_path:CaseDetails}

    def getCaseDetails(self, cur_result, op_details=None):
        case_dicts = {}
        if not cur_result.result:
            logging.warning("No test data here")
            return case_dicts
        no_needed_perf = False
        for accu_res in cur_result.result:
            case_path = accu_res["case_path"]
            accu_data = [accu_res[key] for key in self.accu_key]
            cur_case_obj = CaseDetails(*accu_data)
            op_name = accu_res["op_name"]
            if case_path not in case_dicts:
                case_dicts[case_path] = cur_case_obj
                # if op_details is not None:
                # op_details[op_name] = op_details.get(op_name, 0) + 1
            else:
                case_dicts[case_path].update_result(cur_case_obj.result)
            if (cur_no_needed := accu_res.get("hardware_time(ms)", None)) is not None:
                no_needed_perf = True
                cur_perf_data = [accu_res[key] for key in self.perf_key]
                case_dicts[case_path].update_perf(*cur_perf_data)
        if not no_needed_perf:
            for perf_res in cur_result.perf_result:
                case_path = perf_res["case_path"]
                cur_perf_data = [perf_res[key] for key in self.perf_key]
                case_dicts[case_path].update_perf(*cur_perf_data)

        return case_dicts

    def update_json_bug(
        self,
    ):
        bug_dicts = [self.fix_op, self.new_bug_op, self.unfix_op]
        keys = ["fix", "new_bug", "unfix"]
        for key, bug_dict in zip(keys, bug_dicts):
            self.json_data[f"{key}_num"] = sum(
                [count for op, count in bug_dict.items()]
            )
            self.json_data[f"{key}_op"] = ",".join([op for op in bug_dict])

    def getCmpSummary(self, cmp_details):
        if cmp_details:

            def getCurSummary(cmp_details):
                cur_res = []
                for key in cmp_details:
                    if key == "invalid":
                        continue
                    cur_res.append(
                        sum([op_obj.op_count for _, op_obj in cmp_details[key].items()])
                    )
                return cur_res

            cur_res = getCurSummary(cmp_details)
            res = [sum(cur_res), *cur_res]
            res = "|".join([str(val) for val in res])
            return res
        else:
            return ""

    def getCmpDetails(self, cmp_details, rise=True, last_day=True):
        if cmp_details:
            res = ""
            if rise:
                rise_ops_details = cmp_details["rise"]
                i = 0
                rise_ops_details = sorted(
                    rise_ops_details.items(), key=lambda x: x[1].max_rise, reverse=True
                )
                for op, op_obj in rise_ops_details:
                    res += f"|{i+1}|{op}|"
                    max_rise, max_rise_path = (
                        round(op_obj.max_rise, 2),
                        op_obj.max_rise_path,
                    )
                    cur_obj = self.cur_case_dicts[max_rise_path]
                    if last_day:
                        pre_obj = self.last_day_case_dicts[max_rise_path]
                    else:
                        pre_obj = self.last_version_case_dicts[max_rise_path]
                    pre_hardware_time, pre_kernel = (
                        round(pre_obj.hardware_time, 4),
                        pre_obj.kernel,
                    )
                    cur_hardware_time, cur_kernel = (
                        round(cur_obj.hardware_time, 4),
                        cur_obj.kernel,
                    )
                    res += f"{max_rise}|{pre_hardware_time}|{cur_hardware_time}|{pre_kernel}|{cur_kernel}|"
                    res += "\n"
                    i += 1

            else:
                fallback_ops_details = cmp_details["fallback"]
                i = 0
                fallback_ops_details = sorted(
                    fallback_ops_details.items(),
                    key=lambda x: x[1].max_fallback,
                )
                for op, op_obj in fallback_ops_details:
                    res += f"|{i+1}|{op}|"
                    max_fallback, max_fallback_path = (
                        round(op_obj.max_fallback, 2),
                        op_obj.max_fallback_path,
                    )
                    cur_obj = self.cur_case_dicts[max_fallback_path]
                    if last_day:
                        pre_obj = self.last_day_case_dicts[max_fallback_path]
                    else:
                        pre_obj = self.last_version_case_dicts[max_fallback_path]
                    pre_hardware_time, pre_kernel = (
                        round(pre_obj.hardware_time, 4),
                        pre_obj.kernel,
                    )
                    cur_hardware_time, cur_kernel = (
                        round(cur_obj.hardware_time, 4),
                        cur_obj.kernel,
                    )
                    res += f"{max_fallback}|{pre_hardware_time}|{cur_hardware_time}|{pre_kernel}|{cur_kernel}|"
                    res += "\n"
                    i += 1
            return res
        else:
            return ""

    def getCmpData(self, cur_case_dicts, last_case_dicts, last_day=False):
        # update cmp information with last verison/day
        # if last lay: update new_bug, fix_bug, unfix_bug, add_op_details, cmp_with_day
        # else: update cmp_with_version
        fix_op, new_bug_op, unfix_op = {}, {}, {}
        cmp_details = {"invalid": {}, "flat": {}, "rise": {}, "fallback": {}}
        new_cases = {}
        delete_cases = {}
        intersection = 0
        all_ops = {}
        accu_ops = {"rise": {}, "fallback": {}, "flat": {}, "invalid": {}}
        for cur_path, cur_obj in cur_case_dicts.items():
            op_name = cur_obj.op_name
            if (last_obj := last_case_dicts.get(cur_path, None)) is not None:
                intersection += 1
                # intersection_dicts[cur_path] = {cur_obj, last_obj}
                if isbug(cur_obj.result):
                    if isbug(last_obj.result):
                        unfix_op[op_name] = unfix_op.get(op_name, 0) + 1
                    else:
                        new_bug_op[op_name] = new_bug_op.get(op_name, 0) + 1
                else:
                    if isbug(last_obj.result):
                        fix_op[op_name] = fix_op.get(op_name, 0) + 1
                    else:
                        if "PASS" in cur_obj.result:
                            if "PASS" in last_obj.result:
                                accu_cmp_res = accu_cmp(
                                    cur_obj.diff3_max,
                                    last_obj.diff3_max,
                                    cur_obj.diff3_mean,
                                    last_obj.diff3_mean,
                                )
                                accu_ops[accu_cmp_res][op_name] = (
                                    accu_ops[accu_cmp_res].get(op_name, 0) + 1
                                )
                                cur_hardware_time, last_hardware_time = (
                                    cur_obj.hardware_time,
                                    last_obj.hardware_time,
                                )
                                if isvalid(cur_hardware_time) and isvalid(
                                    last_hardware_time
                                ):
                                    diff1 = (
                                        (last_hardware_time - cur_hardware_time)
                                        / last_hardware_time
                                        * 100
                                    )
                                    if diff1 < -5:
                                        cur_op_obj = cmp_details["fallback"].get(
                                            op_name, OpTp(op_name)
                                        )
                                        cur_op_obj.op_count += 1
                                        if cur_op_obj.max_fallback > diff1:
                                            cur_op_obj.max_fallback = diff1
                                            cur_op_obj.max_fallback_path = cur_path
                                        cmp_details["fallback"][op_name] = cur_op_obj
                                    elif diff1 > 5:
                                        cur_op_obj = cmp_details["rise"].get(
                                            op_name, OpTp(op_name)
                                        )
                                        cur_op_obj.op_count += 1
                                        if cur_op_obj.max_rise < diff1:
                                            cur_op_obj.max_rise = diff1
                                            cur_op_obj.max_rise_path = cur_path
                                        cmp_details["rise"][op_name] = cur_op_obj
                                    else:
                                        cur_op_obj = cmp_details["flat"].get(
                                            op_name, OpTp(op_name)
                                        )
                                        cur_op_obj.op_count += 1
                                        cmp_details["flat"][op_name] = cur_op_obj

                                else:
                                    cur_op_obj = cmp_details["invalid"].get(
                                        op_name, OpTp(op_name)
                                    )
                                    cur_op_obj.op_count += 1
                                    cmp_details["invalid"][op_name] = cur_op_obj
                            else:
                                cur_op_obj = cmp_details["invalid"].get(
                                    op_name, OpTp(op_name)
                                )
                                cur_op_obj.op_count += 1
                                cmp_details["invalid"][op_name] = cur_op_obj
            else:
                # if isbug(cur_obj.result):
                #     new_bug_op[op_name] = new_bug_op.get(op_name, 0) + 1

                cur_op_td = new_cases.get(op_name, OpTp(op_name))
                cur_op_td.update_count(cur_obj.result)
                if not isinstance(cur_obj.compute_force, str) and not isinstance(
                    cur_obj.io_bandwidth, str
                ):
                    cur_op_td.update_io_compute(
                        cur_obj.io_bandwidth, cur_obj.compute_force
                    )
                all_ops[op_name] = all_ops.get(op_name, 0) + 1
                new_cases[op_name] = cur_op_td
        # print(accu_ops)
        if last_day:
            self.unfix_op = unfix_op
            self.fix_op = fix_op
            self.new_bug_op = new_bug_op
            self.add_op_details = new_cases
            self.cmp_with_day = cmp_details
            self.new_cases_num = len(self.cur_case_dicts) - intersection
            self.delete_cases_num = len(self.last_day_case_dicts) - intersection
            self.accu_rise_day_op = list(accu_ops["rise"].keys())
            self.accu_fallback_day_op = list(accu_ops["fallback"].keys())
            for last_path, last_obj in last_case_dicts.items():
                op_name = last_obj.op_name
                if last_path not in cur_case_dicts:
                    last_op_td = delete_cases.get(op_name, OpTp(op_name))
                    last_op_td.update_count(last_obj.result)
                    delete_cases[op_name] = last_op_td
                    all_ops[op_name] = all_ops.get(op_name, 0) + 1
            for op in all_ops:
                self.cur_op_details[op] = [0, 0]  # (add, delete)
                if op_obj := new_cases.get(op, None):
                    self.cur_op_details[op][0] = op_obj.op_count
                if op_obj := delete_cases.get(op, None):
                    self.cur_op_details[op][1] = -op_obj.op_count

        else:
            self.cmp_with_version = cmp_details
            self.accu_fallback_version_op = list(accu_ops["fallback"].keys())
            self.accu_rise_version_op = list(accu_ops["rise"].keys())

    def form(self, data, keys):
        cur_result = [
            dict(zip(keys, [record.get(key, "NAN") for key in keys])) for record in data
        ]
        return cur_result

    def update_add_op_details_table(
        self,
    ):
        keys = ["op_count", "passed_num", "failed_num", "insufficent_num"]
        add_op_details_table = ""
        if self.add_op_details:
            self.tot_add_details = {"tot_ops": len(self.add_op_details)}
            for op, op_obj in self.add_op_details.items():
                op_obj_dict = op_obj.__dict__
                add_op_details_table += f"||{op}|"
                for key in keys:
                    val = op_obj_dict.get(key, 0)
                    self.tot_add_details[key] = self.tot_add_details.get(key, 0) + val
                    add_op_details_table += f"{val}|"
                max_io_bandwidth, min_io_bandwidth = round(
                    op_obj_dict.get("max_io_bandwidth", -1), 4
                ), round(op_obj_dict.get("min_io_bandwidth", -1), 4)
                add_op_details_table += f"{max_io_bandwidth}/{min_io_bandwidth}|"
                max_compute_force, min_compute_force = round(
                    op_obj_dict.get("max_compute_force", -1), 4
                ), round(op_obj_dict.get("min_compute_force", -1), 4)
                add_op_details_table += f"{max_compute_force}/{min_compute_force}|"
                add_op_details_table += "\n"
        return add_op_details_table[:-1]

    def update_json(self):
        if self.result:
            self.cur_case_dicts = self.getCaseDetails(self.result, self.cur_op_details)
        if self.last_day_result:
            self.last_day_case_dicts = self.getCaseDetails(
                self.last_day_result, self.last_day_op_details
            )
        if self.last_version_result:
            self.last_version_case_dicts = self.getCaseDetails(self.last_version_result)

        if self.cur_case_dicts:
            if self.last_day_case_dicts:
                self.getCmpData(
                    self.cur_case_dicts, self.last_day_case_dicts, last_day=True
                )

            if self.last_version_case_dicts:
                self.getCmpData(
                    self.cur_case_dicts, self.last_version_case_dicts, last_day=False
                )

        if self.cur_op_details:
            op_cases_change = {}
            for op, val in self.cur_op_details.items():
                op_add, op_delete = val
                if op_add or op_delete:
                    op_cases_change[op] = (op_add, op_delete)
            self.op_cases_change = "\n".join(
                [
                    f"| {key} | {val[0]}|{val[1]} |"
                    for key, val in op_cases_change.items()
                ]
            )

        self.json_data = self.test_data
        self.update_json_bug()
        self.json_data["delete_cases_num"] = self.delete_cases_num
        self.json_data["new_cases_num"] = self.new_cases_num
        self.json_data["op_cases_change"] = self.op_cases_change
        self.json_data["xlsx_path"] = self.test_file_path
        self.json_data["total_op_num"] = len(getOpList())
        self.json_data["date"] = time.strftime("%Y-%m-%d")
        self.json_data["pass_per"] = round(
            self.json_data["tot_passed"] / self.json_data["tot_cases"] * 100.0, 2
        )
        self.json_data["add_op_details"] = self.update_add_op_details_table()
        self.json_data["tot_add_details"] = "|".join(
            str(val) for _, val in self.tot_add_details.items()
        )

        self.json_data["accu_fallback_day_op"] = ", ".join(
            map(
                lambda x: x + "\n"
                if (self.accu_fallback_day_op.index(x) + 1 % 5) == 0
                else x,
                self.accu_fallback_day_op,
            )
        )
        self.json_data["accu_fallback_version_op"] = ", ".join(
            map(
                lambda x: x + "\n"
                if (self.accu_fallback_version_op.index(x) + 1) % 5 == 0
                else x,
                self.accu_fallback_version_op,
            )
        )
        self.json_data["accu_rise_day_op"] = ", ".join(
            map(
                lambda x: x + "\n"
                if (self.accu_rise_day_op.index(x) + 1) % 5 == 0
                else x,
                self.accu_rise_day_op,
            )
        )
        self.json_data["accu_rise_version_op"] = ", ".join(
            map(
                lambda x: x + "\n"
                if (self.accu_rise_version_op.index(x) + 1) % 5 == 0
                else x,
                self.accu_rise_version_op,
            )
        )
        if self.cmp_with_day:
            self.json_data["cmp_with_day_summary"] = (
                "|" + self.getCmpSummary(self.cmp_with_day) + "|"
            )
            self.json_data["cmp_with_day_rise"] = self.getCmpDetails(
                self.cmp_with_day, rise=True, last_day=True
            )
            self.json_data["cmp_with_day_fallback"] = self.getCmpDetails(
                self.cmp_with_day, rise=False, last_day=True
            )
        if self.cmp_with_version:
            self.json_data["cmp_with_version_summary"] = (
                "|" + self.getCmpSummary(self.cmp_with_version) + "|"
            )
            self.json_data["cmp_with_version_rise"] = self.getCmpDetails(
                self.cmp_with_version, rise=True, last_day=False
            )
            self.json_data["cmp_with_version_fallback"] = self.getCmpDetails(
                self.cmp_with_version, rise=False, last_day=False
            )

    def to_json(self, json_file=""):
        if json_file:
            self.json_file = json_file
        else:
            if not self.json_file:
                self.json_file = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        with open(f"{self.json_file}.json", "w", encoding="utf-8") as f:
            json.dump(self.json_data, f, ensure_ascii=False, indent=4)


class TecoalReport:
    title = "Test Report"

    templates = ""
    json_data = ""
    md_path = "./sample/daily_test_template.md"

    def init_json_data(self, cur_file_path, last_day_file_path, last_version_file_path):
        tpd = TecoalReportData(
            cur_file_path, last_day_file_path, last_version_file_path
        )
        tpd.update_json()
        if isinstance(tpd.json_data, str):
            print(1)
        self.json_data = tpd.json_data

    def __init__(self, cur_file_path, last_day_file_path, last_version_file_path):
        pa_path = os.path.split(os.path.realpath(__file__))[0]
        self.md_path = os.path.join(pa_path, self.md_path)
        if not os.path.isfile(self.md_path):
            logging.warning(f"{self.md_path} not exists")
            return
        with open(self.md_path, "r", encoding="utf-8") as f:
            self.templates = f.read()
        self.init_json_data(cur_file_path, last_day_file_path, last_version_file_path)

    def add_table_content(self, table_data, cur_keys):
        rows_data = ""
        for row_data in table_data:
            row_data2md = "|"
            for key in cur_keys:
                row_data2md += f" {row_data[key]} |"
            rows_data += row_data2md + "\n"
        return row_data

    def add_table(self, table_data, color="black", table_head=None):
        ##support list with dict
        if not table_data:
            return ""
        # add table key
        # print(table_data)
        cur_keys = table_head if table_head else table_data[0].keys()
        # print(cur_keys)
        table_key = "|"
        table_format = "|"
        for key in cur_keys:
            table_key += (
                f" <span style='color: {color}; font-weight: bold;'>**{key}**</span>|"
            )
            table_format += f"-----|"
        rows_data = self.add_table_content(table_data, cur_keys)
        return table_key + "\n" + table_format + "\n" + rows_data

    def transpose_table(self, table_content, table_key=None):
        rows = table_content.split("\n")
        rows = [row.strip() for row in rows if row.strip()]
        transposed_table = []
        for i in range(len(rows[0].split("|"))):
            transposed_row = []
            for row in rows:
                cells = row.split("|")
                cell = cells[i].strip()
                if cell == "-----":
                    continue
                transposed_row.append(cell)
            transposed_table.append("|" + "|".join(transposed_row) + "|")
        transposed_table.pop()
        transposed_table[0] = (
            "|-----|-----|" if not table_key else "|" + "-----|" * len(table_key)
        )
        if table_key:
            table_head = "|"
            for key in table_key:
                table_head += f"{key}|"
        else:
            table_head = "||version|"
        transposed_table.insert(0, table_head)
        transposed_table = "\n".join(transposed_table) + "\n"
        return transposed_table

    def update_templates(self, to_html=False, data=None):
        if not data:
            data = self.json_data
        for key, val in data.items():
            cur_key = "{" + key + "}"
            self.templates = self.templates.replace(cur_key, str(val))
        if to_html:
            self.templates += "\n"
            self.templates += """<!-- Markdeep: --><style class="fallback">body{visibility:hidden;white-space:pre;font-family:monospace}</style><script src="markdeep.min.js" charset="utf-8"></script><script src="https://casual-effects.com/markdeep/latest/markdeep.min.js" charset="utf-8"></script><script>window.alreadyProcessedMarkdeep||(document.body.style.visibility="visible")</script>"""
