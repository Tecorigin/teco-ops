import os
import time
from parse_result import parse_result

class CaseHelper:
    def __init__(self, cases_dir, al_type, version, reduce_num, renew):
        self.cases_dir = cases_dir
        self.al_type = al_type
        self.version = version
        self.reduce_num = reduce_num
        self.renew = renew

    def compare_version(self, version, dst_version):
        if dst_version == "develop":
            return -1
        v1 = version.split('.')
        v2 = dst_version.split('.')
        if len(v1) != len(v2):
            return 1
        for i in range(len(v1)):
            if int(v1[i]) > int(v2[i]):
                return 1
            elif int(v1[i]) < int(v2[i]):
                return -1
        return 0

    def get_versions(self, al_case_path, dst_version):
        versions = []
        if not os.path.exists(al_case_path):
            print("al_case_path {} not found".format(al_case_path))
            return versions
        files = os.listdir(al_case_path)
        for f in files:
            if os.path.isdir(os.path.join(al_case_path, f)):
                if self.compare_version(f, dst_version) > 0:
                    continue
                versions.append(f)
        return versions

    def grep_op_cases(self, op_path):
        cases = []
        files = os.listdir(op_path)
        for f in files:
            f = os.path.join(op_path, f)
            if os.path.isfile(f) and f.endswith(".prototxt"):
                cases.append(f)
            elif os.path.isdir(f):
                cases += self.grep_op_cases(f)
        return cases

    def get_op_cases(self, op_path):
        filted_cases = []
        if not os.path.exists(op_path):
            print("op_path {} not found".format(op_path))
            return filted_cases
        cases = self.grep_op_cases(op_path)
        for i in range(len(cases)):
            if i % self.reduce_num == 0:
                filted_cases.append(cases[i])  
        return filted_cases          

    def get_version_cases(self, version_path):
        cases = []
        if not os.path.exists(version_path):
            print("version_path {} not found".format(version_path))
            return cases
        version_case_file = os.path.join(version_path, "reduced_{}_cases_list.txt".format(self.reduce_num))
        if self.renew:
            if os.path.exists(version_case_file):
                os.remove(version_case_file)
                print("renew testcases, remove old version_case_file {}".format(version_case_file))
        if os.path.exists(version_case_file):
            with open(version_case_file, "r") as f:
                cases = f.readlines()
            cases = [x.strip() for x in cases]
            return cases
        else:
            ops = os.listdir(version_path)
            print(ops)
            for op in ops:
                op_path = os.path.join(version_path, op)
                if os.path.isdir(op_path):
                    cases += self.get_op_cases(op_path)

        if not os.path.exists(version_case_file):
            with open(version_case_file, "w") as f:
                for i in range(len(cases)):
                    f.write(cases[i] + "\n")
        return cases

    def get_all_cases(self):
        cases = []
        al_case_path = os.path.join(self.cases_dir, self.al_type)
        versions = self.get_versions(al_case_path, self.version)
        print("versions:", versions)
        for version in versions:
            cases += self.get_version_cases(os.path.join(al_case_path, version))

        return cases


class Result:
    def __init__(self, al_type=None, op_name=None):
        local_time = time.localtime(time.time())
        ctime = time.strftime("%Y-%m-%d_%H-%M-%S", local_time)
        if al_type is not None and op_name is not None:
            out_path = "../{}_{}_daily_{}".format(al_type, op_name, ctime)
        elif al_type is not None:
            out_path = "../{}_daily_{}".format(al_type, ctime)
        elif op_name is not None:
            out_path = "../{}_{}".format(op_name, ctime)
        else:
            out_path = "../daily_{}".format(ctime)
        self.xml_result_file = "{}.xml".format(out_path)
        self.xls_result_file = "{}.xlsx".format(out_path)
        self.log_file = "{}.log".format(out_path)
        self.cases_list_file= "{}.txt".format(out_path)

class Env:
    def __init__(self, al_type):
        self.cases_dir = "/data/hpe_data/testcase/daily_test"
        self.cases_info_dir="/data/hpe_data/case_info/{}".format(al_type)
        if not os.path.exists(self.cases_info_dir):
            os.makedirs(self.cases_info_dir)
        # self.diffs_info_dir="/data/hpe_data/diff_info"

    def setup(self):
        os.environ['MKL_SERVICE_FORCE_INTEL'] = "1"
        os.environ["MKL_THREADING_LAYER"] = "GNU"
        os.environ['BLAS_DEVICE_WORKSPACE_SIZE'] = "512"
        os.environ['DNN_RESERVE_DATA'] = "1"
        os.environ['DNN_CASE_INFO_PATH'] = self.cases_info_dir
        # os.environ['DNN_DIFF_INFO_PATH'] = self.diffs_info_dir


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gid', dest='gid', type=str, help='gid')
parser.add_argument('--al_type', dest='al_type', type=str, help='al_type')
parser.add_argument('--version', dest='version', type=str, help='version')
parser.add_argument('--op_name', dest='op_name', type=str, help='op_name')
parser.add_argument('--reduce_num', dest='reduce_num', type=int, default=1, help='reduce_num')
parser.add_argument('--renew', dest='renew', type=bool, default=False, help='renew')
args = parser.parse_args()


if __name__ == "__main__":
    if args.gid is None or args.al_type is None or args.version is None:
        print("gid, al_type, version is required")
        exit(1)
    print("gid:{}, al_type:{}, version:{}, op_name:{}, reduce_num:{}, renew:{}".format(
        args.gid, args.al_type, args.version, args.op_name, args.reduce_num, args.renew))
    
    env = Env(args.al_type)
    env.setup()
    result = Result(args.al_type, args.op_name)

    case_helper = CaseHelper(env.cases_dir, args.al_type, args.version, args.reduce_num, args.renew)
    cases = case_helper.get_all_cases()
    with open(result.cases_list_file, "w") as f:
        for case in cases:
            f.write(case + "\n")
    print("cases number is {}".format(len(cases)))

    if args.op_name is not None:
        gtest_filter = "{}_{}/*".format(args.al_type, args.op_name)
    else:
        gtest_filter = "{}_*".format(args.al_type)
    cmd = "cd ../build; ./demo --gid={} --gtest_filter={} --cases_list={} --gtest_output=xml:{} --warm_repeat=3 --perf_repeat=5 > {}".format(args.gid, 
            gtest_filter, result.cases_list_file, result.xml_result_file, result.log_file)
    print(cmd)
    os.system(cmd)

    parse_result(result.xml_result_file, result.xls_result_file, env.cases_info_dir)
