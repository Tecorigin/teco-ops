from tecoalReport import *
import pandas as pd
import argparse
import time
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cur_file_path", dest="cur_file_path", type=str, help="cur file case path"
    )
    parser.add_argument(
        "--last_day_file_path",
        dest="last_day_file_path",
        type=str,
        help="yesterday file case path",
    )
    parser.add_argument(
        "--last_version_file_path",
        dest="last_version_file_path",
        type=str,
        help="last version file case path",
    )
    parser.add_argument("--output", dest="output", type=str, help="output name")

    args = parser.parse_args()
    if args.cur_file_path:
        cur_file_path = args.cur_file_path
    else:
        logging.warning("please input cur test result path")

    if args.last_day_file_path:
        last_day_file_path = args.last_day_file_path
    else:
        logging.warning("please input yesterday test result path")

    if args.last_version_file_path:
        last_version_file_path = args.last_version_file_path
    else:
        logging.warning("please input last version test result path")
    if args.output:
        output_name = args.output
    else:
        output_name = time

    tp = TecoalReport(cur_file_path, last_day_file_path, last_version_file_path)
    tp.update_templates(to_html=True)
    with open(f"{output_name}.html", "w", encoding="utf-8") as f:
        f.write(tp.templates)
