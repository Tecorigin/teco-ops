#encoding utf-8
import argparse
import sys
import time
import logging
import os
import json
import pandas as pd
import json
parser = argparse.ArgumentParser()
parser.add_argument("--commit_message", dest="commit_message", type=str, help="commit_message")
parser.add_argument("--commit_email", dest="commit_email", type=str, help="commit_email")
parser.add_argument("--pr_url", dest="pr_url", type=str, help="pr_url")

args = parser.parse_args()

# title_dict = {'1':"unique/half", '2':"unique/float", '3':"masked_fill/bool", '4':"arg_max/int64",
# '5':"unary_ops/int64", '6': 'unary_ops/float', '7':"scale_tensor/float", '8':"scatter_out/float"}
with open("../tools/title_info.json", 'r') as f:
    title_dict = json.load(f)
fp, _ = os.path.split(os.path.realpath(__file__))
cases_dir = "/eco/teco-al"
info_dir = f"/eco/hpe/user_info/algo_map.json"
def gen_cmd(s, pr_url, email_addr):
    s = s.split(":")[0]
    title_num = s[s.find("[")+1:s.rfind("]")]
    if title_dict.get(title_num, -1) == -1:
        logging.warning(f"title_num not valid, please check commit message")
        sys.exit(-1)
    algo_num = s[s.find("(")+1:s.rfind(")")]
    op_dtype = title_dict[title_num]
    with open(info_dir, 'r') as f:
        info_list = json.load(f)
    info_dict = {}
    for cur_info in info_list:
        info_dict[cur_info["email"]] = cur_info
    if "algo" not in algo_num and not algo_num.isdigit(): 
        per_info = info_dict.get(email_addr, {})
        if not per_info:
            logging.warning(f"{email_addr} not exists, please check")
            sys.exit(-1)
        algo_num = per_info.get("algo", -1)
        if algo_num == -1 :
            logging.warning(f"{email_addr} {algo_num} not valid, please check")
            sys.exit(-1)
    if algo_num.isdigit():
        algo_num = f"algo{algo_num}"
    case_path = f"{cases_dir}/{op_dtype}/{algo_num}"
    pr_num = pr_url[pr_url.rfind("pr_"):]
    sys_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    if os.path.isdir(case_path):
        cmd = f"python3 ../tools/unit_test_v2.py --cases_dir={case_path} --warm_repeat=5 --perf_repeat=10 --output={pr_num}_{sys_time} --gid=all"
        return cmd
    else:
        logging.warning(f"{case_path} not exists, please check commit message")
        sys.exit(-1)
    

if __name__ == "__main__":
    cs = args.commit_message
    # print(cs)
    # print(args.pr_url)
    pr_url = args.pr_url
    email_addr = args.commit_email
    if not cs or not pr_url or not email_addr:
        logging.warning("please check input")
        sys.exit(-1)
    cmd = gen_cmd(cs, pr_url, email_addr)
    print(cmd)
