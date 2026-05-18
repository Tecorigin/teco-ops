import os, json, sys
import argparse
import requests
from log import logger
from result_parser import ResultParser


parser = argparse.ArgumentParser()
parser.add_argument("--result_file", dest="result_file", type=str, help="result file")
parser.add_argument("--test_type", dest="test_type", type=int, help="test_type")
parser.add_argument(
    "--os_env", dest="os_env", type=str, help="os env:centos7, ubuntu22.04"
)

args = parser.parse_args()


def save_task(data):

    res = requests.post(
        url="http://tecode.tecorigin.net/api/testingCase/tecoal/saveTask", json=data
    )
    print(res)

    # res = requests.post(url="http://tecode.tecorigin.net/api/testingCase/tecoal/saveTask", json=data).json()
    # print(res)


if __name__ == "__main__":
    test_type = args.test_type
    os_env = args.os_env
    result_file = args.result_file
    logger.info(f"{test_type=}, {os_env=}, {result_file=}")
    if test_type is None or os_env is None or result_file is None:
        sys.exit(-1)

    if not os.path.exists(result_file):
        logger.error(f"{result_file=} not exists")
        sys.exit(-1)

    rp = ResultParser(result_file, test_type, os_env)
    task, task_case = rp.run()
    print("start task to dict")
    task = task.to_dict()
    print("end task to dict")
    task_case = [tc.to_dict() for tc in task_case]
    print(task_case[0])
    res = {
        "task_data": task,
        "case_data": task_case,
    }
    with open("save_task.json", "w") as f:
        f.write(json.dumps(res, indent=4))
    print("start save")
    save_task(res)
    print("end save")
