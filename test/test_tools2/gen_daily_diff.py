import argparse
import os
import sys

def dir_path(l_path):
    dir_list = []
    for f in os.listdir(l_path):
        if os.path.isdir(os.path.join(l_path, f)) and f != "tmp":
            dir_list.append(f)
        else:
            pass
    return dir_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Log Dir.")
    parser.add_argument(
        "-l",
        "--log",
        dest="log",
        default="/data/Hpe_share/tecoal_long-term_stability/tecoal_stability_accuracy/log/",
        type=str,
        help="Please input test log dir",
    )
    parser.add_argument(
        "-fn",
        "--file_name",
        dest="file_name",
        default="test_perf.xlsx",
        type=str,
        help="Please input test file name",
    )

    args = parser.parse_args()
    log_dir = args.log
    file_name = args.file_name

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
    print(previous_dir_name)
    print(current_dir_name)

    script_path = os.path.abspath(sys.argv[0])
    folder_path = script_path.replace("gen_daily_diff.py","")

    diff_cmd = f"python3 {folder_path}../tools/diff_log.py --cur_path={current_dir_name + '/'+ file_name} --pre_path={previous_dir_name + '/'+ file_name} --threshold=5 --is_ignore --output='cmp_result'"
    print(diff_cmd)
    os.system(diff_cmd)