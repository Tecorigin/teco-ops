import os
import hashlib
from prototxt_parser.prototxt_parser_main import parse
import argparse
import subprocess

def read_from_txt(file_path):
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]
        return lines
    except Exception as e:
        print(e)
        return []

def read_from_dir(dirs):
    try :
        cmd = f"find {dirs} -name '*.prototxt' -type f"
        read_process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if read_process.returncode == 0:
            return read_process.stdout.split()
        else:
            print("no prototxt found")
            return []
    except Exception as e:
        print(e)
        return []

def read(filename):
    content = ''
    with open(filename, 'r') as f:
        content = f.read()
    return content

def get_hash(string):
    m = hashlib.md5()
    m.update(string.encode('utf-8'))
    return m.hexdigest()

parser = argparse.ArgumentParser()
parser.add_argument('--cases_dir',
                    dest='cases_dir',
                    type=str,
                    help='cases dir')
parser.add_argument('--cases_list',
                    dest='cases_list',
                    type=str,
                    help='cases list')
parser.add_argument('--output',
                    dest='output',
                    type=str,
                    help='cases list')
parser.add_argument('--save_prototxt',
                    dest='save_prototxt',
                    action="store_true",
                    help='whether to save prototxt')
args = parser.parse_args()
if __name__ == "__main__":
    path, test_case = "", []
    if args.cases_dir is not None:
        path = args.cases_dir
        test_case = read_from_dir(path)

    if args.cases_list is not None:
        path = args.cases_list
        test_case = read_from_txt(path) 
    cases_num_before = len(test_case)
    if cases_num_before == 0:
        print("no cases")
    
    cases_hash = []
    all_cases = []
    save_prototxt = args.save_prototxt
    for case_path in test_case:
        content = read(case_path)
        case_hash = get_hash(content)
        if case_hash not in cases_hash:
            cases_hash.append(case_hash)
            all_cases.append(case_path)
    cases_num_after = len(all_cases)
    print(f"cases_before_num:{cases_num_before}, cases_after_num:{cases_num_after}")
    save_name = "./cases_list.txt" if args.output is None else args.output
    if not save_name.endswith(".txt"):
        save_name += ".txt"
    if save_prototxt:
        save_path = args.output if args.output is not None else "after_deduplicate"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
    with open(save_name, 'w') as f:
        for case in all_cases:
            f.write(case)
            f.write("\n")
            if save_prototxt:
                op_name = case[:case.rfind('/')]
                op_name = op_name[op_name.rfind("/")+1:]
                op_path = os.path.join(save_path, op_name)
                if not os.path.exists(op_path):
                    os.mkdir(op_path)
                os.system(f"cp {case} {op_path}")

    # data = {}
    # ops = os.listdir(path)
    # cases_num_before = 0
    # for op in ops:
    #     if op.endswith(".py"):
    #         continue
    #     op_path = os.path.join(path, op)
    #     files = os.listdir(op_path)
    #     testcase = []
    #     for f in files:
    #         if f.endswith(".prototxt"):
    #             testcase.append(os.path.join(op_path, f))
    #     print(op, len(testcase))
    #     cases_num_before += len(testcase)
    #     data[op] = testcase

    # print("cases_num_before =", cases_num_before)
    # # print(data)

    # op_cases_hash = {}
    # all_cases = []
    # for op, cases in data.items():
    #     print(op)
    #     if op not in op_cases_hash:
    #         op_cases_hash[op] = []
    #     for case in cases:
    #         case = os.path.abspath(case)
    #         content = read(case)
    #         case_hash = get_hash(content)
    #         if case_hash not in op_cases_hash[op]:
    #             op_cases_hash[op].append(case_hash)
    #             all_cases.append(case)
    #         # else:
    #         # print(case)
    #         # if case_hash == "d20273a8acf5bb0a993a8edc2b856630":
    #         # print(case)
    #     print(op, len(op_cases_hash[op]))
    # print(op_cases_hash.keys())
    # print(len(all_cases))
    # with open("cases_list.txt", "w") as f:
    #     for case in all_cases:
    #         f.write("%s\n" % case)
