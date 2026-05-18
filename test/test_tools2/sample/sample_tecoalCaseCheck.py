import sys
import os
import argparse
from getTecoalNode import getTecoalNode


def find_prototxt_path(path, prototxt_list):
    dir_or_files = os.listdir(path)
    for dir_file_path in dir_or_files:
        new_path = os.path.join(path, dir_file_path)
        if os.path.isdir(new_path):
            find_prototxt_path(new_path, prototxt_list)
        else:
            if new_path.endswith(".prototxt"):
                prototxt_list.append(new_path)


def get_cases_for_dir(path):
    cases = []
    if not os.path.exists(path):
        print("cases dir file {} not found".format(path))
        return cases
    if os.path.isfile(path):
        print("CASES DIR MUST BE A DIR")
        return cases
    if os.path.isdir(path):
        find_prototxt_path(path, cases)
        return cases


def check_save(cases):
    for i, case in enumerate(cases):
        if i % 1000 == 0:
            print(f"{i}/{len(cases)}")
        node = getTecoalNode(case)
        if not node.check():
            print(case)
            node.saveTo(case)


parser = argparse.ArgumentParser()
parser.add_argument("--cases_list",
                    dest="cases_list",
                    type=str,
                    help="cases_list")
parser.add_argument("--cases_dir",
                    dest="cases_dir",
                    type=str,
                    help="cases_dir")

parser.add_argument("--case_path",
                    dest="case_path",
                    type=str,
                    help="case_path")

args = parser.parse_args()

if __name__ == "__main__":
    cases = []
    if args.cases_dir is not None:
        if not os.path.exists(args.cases_dir):
            print(f"{args.cases_dir} not found!!!")
            exit(-1)
        find_prototxt_path(args.cases_dir, cases)
    elif args.cases_list is not None:
        if not os.path.exists(args.cases_list):
            print(f"{args.cases_list} not found!!!")
            exit(-1)
        with open(args.cases_list, 'r') as f:
            cases = f.readlines()
        cases = [case.strip() for case in cases]
    elif args.case_path is not None:
        if not os.path.exists(args.case_path):
            print(f"{args.case_path} not found!!!")
            exit(-1)
        cases.append(args.case_path)
    print(f"case num is {len(cases)}")

    check_save(cases)
