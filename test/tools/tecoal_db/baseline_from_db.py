from get_data_utils import get_baseline

import os, json, sys
import argparse
import requests
from log import logger

parser = argparse.ArgumentParser()
parser.add_argument("--save_name", dest="save_name", type=str, help="save name")
if __name__ == "__main__":
    save_name = args.save_name
    get_baseline(save_name)
