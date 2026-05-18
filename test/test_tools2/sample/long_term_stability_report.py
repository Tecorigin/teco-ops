import argparse
import os
import json
from tecoalReport import *

def dir_path(l_path):
    dir_list = []
    for f in os.listdir(l_path):
        if os.path.isdir(os.path.join(l_path, f)) and f != "tmp":
            dir_list.append(f)
        else:
            pass
    return dir_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--log",
        dest="log",
        default="/data/Hpe_share/tecoal_long-term_stability/tecoal_stability_accuracy/log",
        type=str,
        help="Please input test log dir",
    )
    parser.add_argument(
        "-rf",
        "--result_file_name",
        dest="result_file_name",
        default="test_perf.xlsx",
        type=str,
        help="Please input result file name",
    )
    parser.add_argument(
        "-rr",
        "--release_result_path",
        dest="release_result_path",
        default="/data/Hpe_share/tecoal_long-term_stability/tecoal_stability_accuracy/release_result/release_result.xlsx",
        type=str,
        help="Please input result file name",
    )
    parser.add_argument(
        "-o",
        "--output_file_name",
        dest="output_file_name",
        default="fancy_report",
        type=str,
        help="Please input output file name",
    )
    
    args = parser.parse_args()
    print(args)
    log_dir = args.log
    release_result_path = args.release_result_path
    result_file_name = args.result_file_name
    output_file_name = args.output_file_name
    dir_list = sorted(dir_path(log_dir), reverse=True)
    if len(dir_list) >= 1:
        current_dir_name = os.path.join(log_dir, dir_list[0])
    if len(dir_list) >= 2:
        for i, item in enumerate(dir_list):
            log_list = []
            for files in os.listdir(os.path.join(log_dir, item)):
                if files.endswith(".xlsx"):
                    log_list.append(files)
            if i >= 1 and len(log_list) < 3:
                print("this dir lose some log:", os.path.join(log_dir, item))
            if i >= 1 and len(log_list) >= 3:
                previous_dir_name = os.path.join(log_dir, item)
                break
    
    cur_file_path = current_dir_name + "/" + result_file_name
    pre_file_path = previous_dir_name + "/" + result_file_name
    print(cur_file_path)
    print(pre_file_path)
    tp = TecoalReport(cur_file_path, pre_file_path, release_result_path)
    json_data = json.dumps(tp.json_data)
    f2 = open(f"{output_file_name}.json", 'w')
    f2.write(json_data)
    f2.close()
    
    tp.update_templates(to_html=True)
    with open(f"{output_file_name}.html", "w", encoding="utf-8") as f:
        f.write(tp.templates)
