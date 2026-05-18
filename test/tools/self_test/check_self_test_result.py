import argparse
import os
import sys
import pandas as pd


def read_sheet(excel_path, sheet_name):
    if not os.path.exists(excel_path):
        return []
    df_sheet = pd.read_excel(excel_path, sheet_name)
    data = []
    for index, row in df_sheet.iterrows():
        data_dict = {col: val for col, val in row.items()}
        data.append(data_dict)
    # self.perf_result = data
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_path", 
        dest="result_path", 
        default="/data/dnn/zhuangxin/tecoal/test_result/self_test.xlsx",
        type=str, 
        help="result path")
    parser.add_argument(
        "--log_path", 
        dest="log_path", 
        default="/data/dnn/zhuangxin/tecoal/test/frame_work/tecotest/tools/",
        type=str, 
        help="log path")
    args = parser.parse_args()

    result_path = args.result_path
    log_path = args.log_path
    fail_log_path = result_path.split(".xlsx")[0] + "_fail.log"

    test_status = 0

    fw1 = open(fail_log_path,'w',encoding='utf-8')
    pass_result = ["PASS_S2", "PASS_S1"]
    if os.path.exists(result_path):
        result_accu = read_sheet(result_path, "accu")
        # print(result_accu)
        for result in result_accu:
            if result["result"] not in pass_result:
                test_status = 1
                dest_path = log_path + result["log"] 
                with open(dest_path, 'r') as in_file:
                    fw1.write(in_file.read())
                    fw1.write('\n')
        if test_status != 0:
            print("Test Failed!!!!")
        else:
            print("Test Passed!!!")
    else:
        print("self test result cannot be found in " + result_path)
        test_status = 1
    sys.exit(test_status)
