import os, json, sys
import argparse
import requests
from log import logger
from result_parser import BaselineParser

parser = argparse.ArgumentParser()
parser.add_argument("--result_file", dest="result_file", type=str, help="result file")
parser.add_argument("--task_id", dest="task_id", type=int, help="task_id")
args = parser.parse_args()


def save_baseline_by_task(task_id):
    data = {"task_id": task_id}
    url = "http://tecode.tecorigin.net/api/testingCase/tecoal/updateCaseBaselineByTask"
    res = requests.post(
        url=url,
        json=data,
    ).json()
    print(res)


def save_case_baseline(data):
    url = "http://tecode.tecorigin.net/api/testingCase/tecoal/updateCaseBaseline"
    res = requests.post(
        url=url,
        json=data,
    ).json()
    print(res)


if __name__ == "__main__":
    if args.task_id:
        save_baseline_by_task(task_id=args.task_id)

    else:
        file_name = args.result_file
        if not file_name or not os.path.exists(file_name):
            logger.error(f"please check result_file {file_name}")
        bp = BaselineParser(file_name)
        bp.run()
        save_case_baseline(bp.baseline_data_row)
