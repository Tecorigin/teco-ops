from dedu_utils import *
import pandas as pd
import sys
import os
import hashlib
import json
import argparse
import time
import numpy as np
import subprocess
import shutil
import stat
import asyncio
from prototxt_parser.prototxt_parser_main import parse
from tqdm import tqdm
from node_utils import getTecoalNode

parser = argparse.ArgumentParser()
parser.add_argument("--cases_dir", dest="cases_dir", type=str, help="cases_dir")
parser.add_argument("--model_name", dest="model_name", type=str, help="model_name")
args = parser.parse_args()



def single_model_dedu(cases_dir, model_name):
    pt_files = list(find_prototxt_files(cases_dir))
    pt_content = {}
    gen_cases_hash_dict(pt_files, pt_content)
    # print(len(pt_content))
    all_cases = {}
    dedu_cases = []
    for pt in pt_files:
        case = pt_content.get(pt, None)
        if not case:
            continue
        model_case = ModelCases(case)
        case_hash = model_case.to_str()
        if val:=all_cases.get(case_hash, []):
            val.append(pt)
            all_cases[case_hash] = val
        else:
            dedu_cases.append(pt)
            all_cases[case_hash] = [pt]
    if dedu_cases:
        with open(f"./{model_name}_dedu.txt", 'w') as f:
            f.write("\n".join(dedu_cases))
    
if __name__ == "__main__":
    cases_dir = args.cases_dir
    model_name = args.model_name if args.model_name else "all_models"
    if not cases_dir or not os.path.exists(cases_dir):
        print(f"{cases_dir} not exist")
        sys.exit(-1)
    single_model_dedu(cases_dir, model_name)

