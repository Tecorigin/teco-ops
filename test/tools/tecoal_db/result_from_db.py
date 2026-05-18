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
parser.add_argument("--test_type", dest="test_type", type=str, help="test type")
parser.add_argument("--shape", dest="shape", type=str, help="shape")
parser.add_argument("--spe_clock", dest="spe_clock", type=str, help="spe_clock")
parser.add_argument("--os", dest="os", type=str, help="os")
parser.add_argument("--date", dest="date", type=str, help="date")
parser.add_argument("--tecoal", dest="tecoal", type=str, help="tecoal")
parser.add_argument("--tecoblas", dest="tecoblas", type=str, help="tecoblas")
parser.add_argument("--tecocustom", dest="tecocustom", type=str, help="tecocustom")
parser.add_argument("--save_name", dest="save_name", type=str, help="save name")
args = parser.parse_args()

if __name__ == "__main__":
    model_name = args.model_name
    model_version = args.model_version
    op_name = args.op_name
    shape = args.shape
    spe_clock = args.spe_clock
    cur_os = args.os
    date = args.date
    tecoal = args.tecoal
    tecoblas = args.tecoblas
    tecocustom = args.tecocustom
    save_name = args.save_name
    test_type = args.test_type
    get_result(
        save_name=save_name,
        op_name=op_name,
        test_type=test_type,
        date=date,
        model_name=model_name,
        model_version=model_version,
        shape=shape,
        tecoal=tecoal,
        tecoblas=tecoblas,
        tecocustom=tecocustom,
        spe_clock=spe_clock,
        os=cur_os,
    )
