import os, json, sys
import argparse
import requests
from log import logger
from case_parser import CaseParser, ModelParser


parser = argparse.ArgumentParser()
parser.add_argument("--cases_list",
                    dest="cases_list",
                    type=str,
                    help="cases list")
parser.add_argument("--cases_json",
                    dest="cases_json",
                    type=str,
                    help="cases json")
parser.add_argument("--version", dest="version", type=str, help="version")
parser.add_argument("--level", dest="level", type=int, help="level")
parser.add_argument("--bug_id", dest="bug_id", type=int, help="bug id")
parser.add_argument("--model_name",
                    dest="model_name",
                    type=str,
                    help="model name")
parser.add_argument("--image",
                    dest="image",
                    type=str,
                    help="image")

args = parser.parse_args()


def save_cases(data):
    res = requests.post(url="http://tecode.tecorigin.net/api/testingCase/tecoal/saveCase", json=data).json()
    print(res)

def save_model_case(data):
    res = requests.post(url="http://tecode.tecorigin.net/api/testingCase/tecoal/saveModelCase", json=data).json()
    print("save_model_case:", res)

if __name__ == "__main__":
    # model case
    if args.model_name is not None:
        model_name = args.model_name
        version = args.version
        image = args.image
        cases_json = args.cases_json
        logger.info(f"{version=}, {model_name=}, {image=}, {cases_json=}")
        if version is None or image is None or cases_json is None:
            logger.error("params error")
            sys.exit(-1)

        if not os.path.exists(cases_json):
            logger.error(f"{cases_json=} not exists")
            sys.exit(-1)

        mp = ModelParser(cases_json, model_name, image, version)
        model, cases = mp.run()
        model = model.to_dict()
        cases = [case.to_dict() for case in cases]
        res = {
            "model_data": model,
            "case_data": cases,
        }
        save_model_case(res)
    # bug case
    elif args.bug_id is not None and args.bug_id != -1:
        bug_id = args.bug_id
        version = args.version
        cases_list = args.cases_list
        level = 2
        if version is None or cases_list is None:
            logger.error("params error")
            sys.exit(-1)
        if not os.path.exists(cases_list):
            logger.error(f"{cases_list=} not exists")
            sys.exit(-1)

        logger.info(f"{version=}, {bug_id=}, {cases_list=}")
        cp = CaseParser(cases_list, version, level, bug_id)
        cases = cp.run()
        cases = [case.to_dict() for case in cases]
        # save_cases(cases)
        with open("111.json", 'w') as f:
            f.write(json.dumps(cases, indent=4))
    else:
        version = args.version
        level = args.level
        cases_list = args.cases_list
        logger.info(f"{version=}, {level=}, {cases_list=}")
        if version is None or cases_list is None or level is None or level==2 or level==1:
            logger.error("params error")
            sys.exit(-1)

        if not os.path.exists(cases_list):
            logger.error(f"{cases_list=} not exists")
            sys.exit(-1)

        cp = CaseParser(cases_list, version, level)
        cases = cp.run()
        cases = [case.to_dict() for case in cases]
        # save_cases(cases)
        with open("222.json", 'w') as f:
            f.write(json.dumps(cases, indent=4))
