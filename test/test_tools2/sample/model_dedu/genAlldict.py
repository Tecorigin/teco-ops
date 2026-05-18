import subprocess
import shutil
import stat
import asyncio
import argparse
import os
from prototxt_parser.prototxt_parser_main import parse
from tqdm import tqdm
from node_utils import getTecoalNode
from dedu_utils import *
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--cases_dir", dest="cases_dir", type=str, help="cases_dir")
parser.add_argument("--save_dir", dest="save_dir", type=str, help="save_dir")
parser.add_argument("--model_name", dest="model_name", type=str, help="model_name")
args = parser.parse_args()


def genAllHashCase(cases_dir, save_dir):
    pt_files = list(find_prototxt_files(cases_dir))
    print(len(pt_files))
    hash_cases = {}
    pt_content = {}
    gen_cases_hash_dict(pt_files, pt_content)
    for pt in pt_files:
        case = pt_content.get(pt, None)
        if not case:
            continue
        model_case = ModelCases(case)
        op_name = model_case.op_name
        case_hash = model_case.to_str()
        hash_cases[case_hash] = pt
    with open(f"{save_dir}/hash2case.json", "w") as f:
        hash_cases_str = json.dumps(hash_cases, indent=4)
        f.write(hash_cases_str)


if __name__ == "__main__":
    cases_dir = args.cases_dir
    save_dir = args.save_dir
    model_name = args.model_name if args.model_name else "all_models"
    if not cases_dir or not os.path.exists(cases_dir):
        print(f"{cases_dir} not exist")
        sys.exit(-1)
    if not save_dir or not os.path.exists(save_dir):
        print(f"{save_dir} not exist")
        sys.exit(-1)
    genAllHashCase(cases_dir, save_dir)
