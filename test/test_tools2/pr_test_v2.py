import time, os
import sys
from tecoalResult import *
import argparse

MAX_CASE_NUM = 1000000
GROUP_PER_CASES_NUM = 5
GROUP_NUM = 3


def grep_file(file_path):
    outfiles = []
    if not os.path.exists(file_path):
        return outfiles
    files = os.listdir(file_path)
    for f in files:
        f = os.path.join(file_path, f)
        if os.path.isfile(f) and f.endswith(".log"):
            outfiles.append(f)
        elif os.path.isdir(f):
            outfiles += grep_file(f)
    return outfiles


def concat_file(infiles, outfile):
    if os.path.exists(outfile):
        os.remove(outfile)
    with open(outfile, "a") as fout:
        for infile in infiles:
            with open(infile, "r") as fin:
                try:
                    fout.write(fin.read())
                except Exception as e:
                    fout.write(str(e) + "\n")
                    fout.write(infile)
                fout.write("\n")


def split_cases(cases):
    cases_num = len(cases)
    splited_cases_list = []
    if cases_num >= GROUP_PER_CASES_NUM * GROUP_NUM:
        group_per_cases_num = (len(cases) + GROUP_NUM - 1) // GROUP_NUM
        for i in range(GROUP_NUM - 1):
            split_cases = cases[group_per_cases_num * i : group_per_cases_num * (i + 1)]
            splited_cases_list.append(split_cases)
        i = GROUP_NUM - 1
        split_cases = cases[group_per_cases_num * i :]
        splited_cases_list.append(split_cases)
    else:
        for i in range(GROUP_NUM):
            start = i * GROUP_PER_CASES_NUM
            end = (i + 1) * GROUP_PER_CASES_NUM
            if end >= cases_num:
                split_cases = cases[start:]
                splited_cases_list.append(split_cases)
                break
            else:
                split_cases = cases[start:end]
                splited_cases_list.append(split_cases)
    return splited_cases_list


def run_cmd(cmd):
    print(cmd)
    os.system(cmd)


class PrTest:
    def __init__(self, cases_list, test_ops, bench_xlsx):
        self.cases_list = cases_list
        self.test_ops = test_ops
        self.cases_num = 0
        self.work_path = "./"
        self.log_outflie = "test.log"
        self.failed_case_outfile = "failed_cases.tar.gz"
        self.perf_insufficient_case_outfile = "perf_insufficient.tar.gz"
        self.accu_insufficient_case_outfile = "accu_insufficient.tar.gz"
        self.status = 0  # 0:pass 1:failed
        self.bench_xlsx = bench_xlsx
        print(f"bench xlsx: {self.bench_xlsx}")
        # time.sleep(10)

    def get_test_ops(self):
        cmd = "cd ../../../../; git diff-tree -r --no-commit-id --name-only --diff-filter=ACMRT HEAD"
        files = os.popen(cmd).read().strip().split("\n")

        self.test_ops = []
        for f in files:
            words = f.split("/")
            if len(words) >= 3 and words[1] in ["ops", "slave"]:
                op_name = f.split("/")[2]
                conv_related_ops = ["conv_forward",
                                    "fused_conv_batchnorm_activation_add_forward",
                                    "fused_conv_bias_activation_add_forward",
                                    "fused_conv_bias_activation_forward",
                                    "fused_conv_bias_add_activation_forward"]
                if "gemm" in op_name:
                    self.test_ops += [
                        "gemm",
                        "gemm_batched",
                        "gemm_batched_v2",
                        "gemm_stride_batched",
                        "gemmex",
                        "gemmex_batched",
                        "gemmex_stride_batched",
                        "add_batched_gemm",
                    ]
                elif op_name in conv_related_ops:
                    self.test_ops += conv_related_ops
                elif op_name == "broadcast":
                    self.test_ops += [
                        "tensor_add",
                        "tensor_mul",
                        "tensor_sub",
                        "tensor_div",
                        "tensor_equal_greater",
                        "tensor_greater",
                        "tensor_less",
                        "tensor_equal",
                        "where_tensor",
                        "expand",
                        "maximum",
                        "minimum",
                        "bitwise_and_tensor",
                        "logical_and_tensor",
                    ]
                elif op_name == "bitwise":
                    self.test_ops += [
                        "bitwise_and_tensor",
                        "bitwise_not_tensor",
                        "bitwise_or_tensor",
                        "bitwise_xor_tensor",
                    ]
                elif op_name == "logical":
                    self.test_ops += [
                        "logical_and_tensor",
                        "logical_not_tensor",
                        "logical_or_tensor",
                        "logical_xor_tensor",
                    ]
                else:
                    self.test_ops.append(op_name)

        self.test_ops = list(set(self.test_ops))
        print("test_op =", self.test_ops)

    def run_op_test(self):
        if len(self.test_ops) == 0:
            return

        with open(self.cases_list, "r") as f:
            lines = f.readlines()
        all_cases = [x.strip() for x in lines]

        cases = []
        for op in self.test_ops:
            op_cases = []
            for case in all_cases:
                if "/" + op + "/" in case:
                    op_cases.append(case)
            USE_MAX_NUM = 1
            try:
                USE_MAX_NUM = int(os.environ["USE_MAX_NUM"])
            except:
                pass
            if USE_MAX_NUM > 0:
                if len(op_cases) > MAX_CASE_NUM:  # limit case num
                    op_cases = op_cases[0:MAX_CASE_NUM]
            cases += op_cases
        self.cases_num = len(cases)
        print("case_num:", self.cases_num)
        if self.cases_num == 0:
            return

        test_ops = ";".join(self.test_ops)
        cases_list = f"./cases_list.txt"
        with open(cases_list, "w") as f:
            f.write("\n".join(cases))
        cases_list_abs = os.path.abspath(cases_list)
        cmd = 'python3 ../tools/unit_test_v2.py --gid {} --test_name "{}" --cases_list {} --time_out_threshold {} --rand_n 3000'.format(
            "0,1,2", test_ops, cases_list, 300
        )
        print(cmd)
        os.system(cmd)

    def deal_log(self):
        logfiles = grep_file("log/")
        concat_file(logfiles, self.log_outflie)

    def deal_fail_case(self):
        # get all cases
        files = os.listdir(self.work_path)
        xlsx_files = []
        for f in files:
            if f.endswith(".xlsx"):
                xlsx_files.append(f)

        if len(xlsx_files) == 0:
            if self.cases_num != 0:
                self.status = 1
                print("[error]: no xlsx file generated!!!")
            return

        result_obj = TecoalResult()
        for f in xlsx_files:
            cur_result_obj = TecoalResult(f)
            result_obj.extend(cur_result_obj)

        # get failed cases
        failed_cases = {}
        accu_insufficient_cases = {}
        perf_insufficient_cases = {}

        passed_res = ["PASS_S1", "PASS_S2", "PASS_S3", "PASS_S4", "PASS_S5", "PASS_S6"]
        for res in result_obj.result:
            if res["result"] not in passed_res:
                self.status = 1
                op = res["op_name"]
                case_path = res["case_path"]
                # failed_cases.append([op, case_path])
                failed_cases[op] = failed_cases.get(op, {})
                failed_cases[op][case_path] = failed_cases[op].get(case_path, 0) + 1

        try:
            if self.bench_xlsx and os.path.exists(self.bench_xlsx):
                print(self.bench_xlsx)
                diff_cmd = f"python3 ../tools/diff_log.py --cur_path={xlsx_files[0]} --pre_path={self.bench_xlsx} --threshold=5 --is_ignore --is_ci --ignore_list=/data/Hpe_share/tecoal_long-term_stability/testsuite/ignore_case_list.txt --output='cmp'"
                print(diff_cmd)
                os.system(diff_cmd)
                cmp_path = "./cmp.xlsx"
                if os.path.exists(cmp_path):
                    diff_accu_result = read_sheet(cmp_path, "diff_accu")
                    diff_perf_result = read_sheet(cmp_path, "diff_perf")
                    for cur_accu_result in diff_accu_result:
                        if (
                            "accu_performance" in cur_accu_result
                            and cur_accu_result["accu_performance"] == "fall_back"
                        ):
                            op = cur_accu_result["op_name"]
                            case_path = cur_accu_result["case_path"]
                            accu_insufficient_cases[op] = accu_insufficient_cases.get(
                                op, {}
                            )
                            accu_insufficient_cases[op][case_path] = (
                                accu_insufficient_cases[op].get(case_path, 0) + 1
                            )
                            self.status = 1
                    # opInfo {op:[tot_count, fall_count(5%-20%)]}
                    opInfo = {}
                    for cur_perf_result in diff_perf_result:
                        op = cur_perf_result["op_name"]
                        cur_op_info = opInfo.get(op, [0, 0])
                        cur_op_info[0] += 1
                        if (
                            "performance" in cur_perf_result
                            and cur_perf_result["performance"] == "fall_back"
                        ):
                            rate = cur_perf_result["rate(%)"]
                            case_path = cur_perf_result["case_path"]
                            perf_insufficient_cases[op] = perf_insufficient_cases.get(
                                op, {}
                            )
                            perf_insufficient_cases[op][case_path] = (
                                perf_insufficient_cases[op].get(case_path, 0) + 1
                            )
                            if rate < -20:
                                self.status = 1
                            else:  # (-5,-20)
                                cur_op_info[1] += 1
                        opInfo[op] = cur_op_info
                    for op, val in opInfo.items():
                        tot_cnt, f_cnt = val
                        rate = (f_cnt / tot_cnt) * 100.0
                        if rate > 0.5:
                            self.status = 1
                            break
        except Exception as e:
            self.status = 1

        if failed_cases:
            # tar failed cases
            self.tar_files(failed_cases, self.failed_case_outfile)
            self.gen_failed_cases_txt(failed_cases, self.failed_case_outfile)

        if accu_insufficient_cases:
            self.tar_files(accu_insufficient_cases, self.accu_insufficient_case_outfile)
            self.gen_failed_cases_txt(
                accu_insufficient_cases, self.accu_insufficient_case_outfile
            )

        if perf_insufficient_cases:
            self.tar_files(perf_insufficient_cases, self.perf_insufficient_case_outfile)
            self.gen_failed_cases_txt(
                perf_insufficient_cases, self.perf_insufficient_case_outfile
            )

    def gen_failed_cases_txt(self, cases_dict, cases_outfile):
        cases_outpath = cases_outfile.split(".tar.gz")[0] + ".log"
        fw = open(cases_outpath, "w", encoding="utf-8")
        for op in cases_dict.keys():
            fw.write("op name: " + op + "\n")
            fw.write("case_path: " + "\n")
            for path in cases_dict[op].keys():
                fw.write(path)
                fw.write("\n")
            fw.write("\n")

    def tar_files(self, cases_dict, cases_outfile):
        cases_outpath = cases_outfile.split(".tar.gz")[0]
        if os.path.exists(cases_outpath):
            cmd = "rm -rf {}".format(cases_outpath)
            os.system(cmd)
        os.mkdir(cases_outpath)

        for op, cases in cases_dict.items():
            cur_cases = list(cases.keys())
            op_path = os.path.join(cases_outpath, op)
            if not os.path.exists(op_path):
                os.mkdir(op_path)
            for case_path in cur_cases:
                cmd = "cp {} {}/".format(case_path, op_path)
                os.system(cmd)

        cmd = "tar -zcf {} {}".format(cases_outfile, cases_outpath)
        print(cmd)
        os.system(cmd)

    def run(self):
        if len(self.test_ops) == 0:
            self.get_test_ops()
        self.run_op_test()
        self.deal_log()
        self.deal_fail_case()

    def deal_result(self):
        local_time = time.localtime(time.time())
        ctime = time.strftime("%Y-%m-%d_%H-%M-%S", local_time)
        out_path = "../log_{}".format(ctime)
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        cmd = "cp *.log *.xlsx *.tar.gz {}".format(out_path)
        print(cmd)
        os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases_list", dest="cases_list", type=str, help="cases list")
    parser.add_argument("--test_name", dest="test_name", type=str, help="test name")
    parser.add_argument("--bench_xlsx", dest="bench_xlsx", type=str, help="bench xlsx")
    args = parser.parse_args()

    if not args.cases_list:
        print("please input cases_list")
        sys.exit(1)
    cases_list = args.cases_list
    bench_xlsx = ""
    print(cases_list)

    if args.test_name:
        test_ops = args.test_name.split(",")
    else:
        test_ops = []
    if args.bench_xlsx:
        bench_xlsx = args.bench_xlsx
    if not os.path.exists(cases_list):
        print("[error]: cases_list {} not exist!!!".format(cases_list))
        sys.exit(-1)

    local_time = time.localtime(time.time())
    ctime = time.strftime("%Y-%m-%d_%H-%M-%S", local_time)
    print("start time is {}".format(ctime))

    pr_test = PrTest(cases_list, test_ops, bench_xlsx)
    pr_test.run()
    pr_test.deal_result()

    local_time = time.localtime(time.time())
    ctime = time.strftime("%Y-%m-%d_%H-%M-%S", local_time)
    print("end time is {}".format(ctime))

    if pr_test.status == 0:
        print("TEST PASSED!!!")
        sys.exit(0)
    else:
        print("TEST FAILED!!!")
        sys.exit(-1)
