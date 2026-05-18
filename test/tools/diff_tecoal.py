import os
import sys
import argparse
import time
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--pre_path", dest="pre_path", type=str, help="pre_version")
parser.add_argument("--cur_path", dest="cur_path", type=str, help="cur_version")
parser.add_argument("--ops", dest="ops", type=str, help="op_name")
parser.add_argument("--output", dest="output", type=str, help="output_name")

args = parser.parse_args()


def find_files(file_path, ops=[]):
    files = []
    cmd = f"find {file_path} -type f"
    process = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    res = process.stdout.strip().split("\n")
    for file in res:
        if ops:
            for op in ops:
                if op in file or file.endswith(".txt"):
                    files.append(file)
                    break
        else:
            files.append(file)
    yield from files


def get_teco2case(files, teco2case):
    for file in files:
        with open(file, "r") as f:
            lines = f.read().strip().split(".prototxt")
            # print(lines)
            for line in lines:
                if ":" not in line:
                    continue
                # print(line)
                tecohash, casepath = line.strip().split(":")
                casepath = (
                    casepath + ".prototxt"
                    if not casepath.endswith(".prototxt")
                    else casepath
                )
                tecohash, casepath = tecohash.strip(), casepath.strip()
                teco2case[tecohash] = casepath


def diff_file(cur_file, pre_file):
    cmd = f"diff -b {cur_file} {pre_file}"
    process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    result = process.stdout.strip()
    if result:
        return False
    else:
        return True


def diff_dirs(cur_dir, pre_dir, result={}, ops=[]):
    # cur_teco_files = [
    #     file for file in sorted(os.listdir(cur_dir)) if not file.endswith(".txt")
    # ]
    # pre_teco_files = [
    #     file for file in sorted(os.listdir(pre_dir)) if not file.endswith(".txt")
    # ]
    cur_teco_files, pre_teco_files, txt_files = [], [], []
    for file in find_files(cur_dir, ops):
        if file.endswith(".txt"):
            txt_files.append(file)
        else:
            cur_teco_files.append(file)

    for file in find_files(pre_dir, ops):
        if not file.endswith(".txt"):
            pre_teco_files.append(file)

    if not txt_files:
        print(f"cur dir not contain bin2casepath.txt, please input correct dir")
        sys.exit(-1)
        # print(cur_dir)
        # parent_dir = os.path.split(cur_dir)[0]
        # for file in os.listdir(parent_dir):
        #     file = os.path.join(parent_dir, file)
        #     if file.endswith(".txt"):
        #         txt_files.append(file)

    cur_teco_files = sorted(cur_teco_files)
    pre_teco_files = sorted(pre_teco_files)
    teco2case = {}
    get_teco2case(txt_files, teco2case)
    # print(teco2case)
    i, j = 0, 0
    e1, e2 = len(cur_teco_files), len(pre_teco_files)
    # diff, not_matched = 0, 0
    not_matched = 0
    while i < e1 and j < e2:
        cur_teco_file = os.path.split(cur_teco_files[i])[1]
        pre_teco_file = os.path.split(pre_teco_files[j])[1]
        if cur_teco_file == pre_teco_file:
            cur_res = diff_file(cur_teco_files[i], pre_teco_files[j])
            if cur_res:
                result["same"] = result.get("same", set())
                result["same"].add(teco2case[cur_teco_files[i]])
                # print(teco2case)
                # print(cur_teco_file)
            else:
                result["diff"] = result.get("diff", set())
                result["diff"].add(teco2case[cur_teco_files[i]])
            i, j = i + 1, j + 1
        elif cur_teco_file < pre_teco_file:
            result["not_matched"] = result.get("not_matched", set())
            result["not_matched"].add(teco2case[cur_teco_files[i]])
            i += 1
        else:
            j += 1
    if i < e1:
        result["not_matched"] = result.get("not_matched", set())
        # result["not_matched"].update()
        for tecofile in cur_teco_files[i:]:
            result["not_matched"].add(teco2case[tecofile])


if __name__ == "__main__":
    if args.cur_path and os.path.exists(args.cur_path):
        cur_dir = os.path.abspath(args.cur_path)
    else:
        print("cur dir should input or exist")
        sys.exit(-1)
    if args.pre_path and os.path.exists(args.pre_path):
        pre_dir = os.path.abspath(args.pre_path)
    else:
        print("pre dir should input or exist")
        sys.exit(-1)
    ops = []
    if args.ops:
        ops = args.ops.split(";")
    output = args.output if args.output else "diff_files"
    if not output.endswith(".txt"):
        output += ".txt"
    result = {}
    diff_dirs(cur_dir, pre_dir, result, ops)
    tot_len = 0
    for key in result:
        tot_len += len(result[key])
    print(f"num of cases is {tot_len}")
    if result.get("diff", []):
        len_diff = len(result["diff"])
        print(f"num of diff cases is {len_diff}")
        with open(output, "w") as f:
            f.write("\n".join(result["diff"]))
    if result.get("same", []):
        len_same = len(result["same"])
        print(f"num of same cases is {len_same}")
