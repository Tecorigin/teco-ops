import argparse
import os
import sys
from GitRepository import GitRepository

def count_prototxt_files(folder_path):
    count = 0
    try:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.prototxt'):
                    count += 1
    except FileNotFoundError:
        print("Folder path cannnot be found!!!")
        sys.exit(1)
    except Exception as e:
        print("Exception happened:", e)
        sys.exit(1)
    return count

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tecoal_path", 
                        dest="tecoal_path", 
                        type=str, 
                        default="/data/dnn/zhuangxin/tecoal/",
                        help="tecoal path")
    args = parser.parse_args()
    tecoal_path = args.tecoal_path

    tecoal = GitRepository(local_path = tecoal_path, repo_url = "ssh://gerrit@10.10.30.1:29418/tecoal")
    latest_commit = tecoal.latest_commit()

    if "[feat]" in latest_commit or "[fix]" in latest_commit or "[perf]" in latest_commit:
        if "cases_list" not in latest_commit:
            print("Need to add cases_list in commit msg!!")
            sys.exit(1)
        elif "tecotest_branch" not in latest_commit:
            print("Need to add tecotest_branch in commit msg!!")
            sys.exit(1)
        else:
            latest_commit_detail_list = latest_commit.split("\n")
            print(latest_commit_detail_list)
            for i in latest_commit_detail_list:
                if "cases_list" in i:
                    cases_list = i.split(":")[-1].strip()
            if cases_list.endswith(".txt"):
                print("Case list type is txt")
                try:
                    # 打开文件进行读取
                    with open(cases_list, 'r') as file:
                        # 逐行读取文件内容并计算行数
                        line_count = sum(1 for line in file)
                        print("Case num is:", line_count)
                        if "[feat]" in latest_commit and line_count < 300:
                            print("The number of cases must be greater than 300!!")
                            sys.exit(1)
                except FileNotFoundError:
                    print("Cases_list cannot be found!!")
                    sys.exit(1)
                except IOError:
                    print("File cannot be read succesfully")
                    sys.exit(1)
                except Exception as e:
                    print("Exception happened:", e)
                    sys.exit(1)
            else:
                print("Case type is folder")
                try:
                    file_count = count_prototxt_files(cases_list)
                    print("The number of files which ends with .prototxt is:", file_count)
                    if "[feat]" in latest_commit and file_count < 300:
                        print("The number of cases must be greater than 300!!")
                        sys.exit(1)
                except Exception as e:
                    print("Exception happened:", e)
        print("check all paseed!!")
    else:   
        print("No need to run self test!!!!")
        sys.exit(0)
