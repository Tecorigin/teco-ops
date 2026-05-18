from get_data_utils import *
import os, json, sys
import argparse
import requests
from log import logger

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", dest="model_name", type=str, help="model name")
parser.add_argument(
    "--model_version", dest="model_version", type=str, help="model version"
)
parser.add_argument("--op_name", dest="op_name", type=str, help="op name")
parser.add_argument("--version", dest="version", type=str, help="version")
parser.add_argument("--level", dest="level", type=int, help="level")
parser.add_argument(
    "--pr_list", dest="pr_list", action="store_true", help="whether to get pr list"
)
parser.add_argument(
    "--daily_list",
    dest="daily_list",
    action="store_true",
    help="whether to get daily list",
)
parser.add_argument(
    "--weekly_list",
    dest="weekly_list",
    action="store_true",
    help="whether to get weekly list",
)
args = parser.parse_args()
if __name__ == "__main__":
    # 获取model list, model_name跟model_version不能为空
    if args.model_name:
        if not args.model_version:
            logger.error("parmas error")
        model_names = args.model_name.split(";")
        model_version = args.model_version
        for model_name in model_names:
            cases_list = get_model_cases(
                model_name=model_name, model_version=model_version
            )
            if cases_list:
                with open(f"./{model_name}.txt", "w") as f:
                    f.write("\n".join(list(cases_list)))
    elif args.pr_list:
        ops = args.op_name.split(";")
        for op in ops:
            cases_list = get_pr_cases(op_name=op)
            if cases_list:
                with open(f"./pr_list_{op}.txt", "w") as f:
                    f.write("\n".join(list(cases_list)))
    elif args.daily_list:
        cases_list = get_daily_cases()
        if cases_list:
            with open(f"./daily_list.txt", "w") as f:
                f.write("\n".join(list(cases_list)))
    elif args.weekly_list:
        cases_list = get_weekly_cases()
        if cases_list:
            with open(f"./weekly_list.txt", "w") as f:
                f.write("\n".join(list(cases_list)))
    else:
        op_name = args.op_name
        version = args.version
        level = args.level
        cases_list = get_cases(op_name=op_name, level=level, version=version)
        if cases_list:
            with open(f"./queery_list.txt", "w") as f:
                f.write("\n".join(list(cases_list)))
