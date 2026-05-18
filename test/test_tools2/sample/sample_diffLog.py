from tecoalResult import TecoalResult
import pandas as pd
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--pre_path", dest="pre_path", type=str, help="pre_version")
parser.add_argument("--cur_path", dest="cur_path", type=str, help="cur_version")

args = parser.parse_args()

if __name__ == "__main__":
    pre_path = args.pre_path
    cur_path = args.cur_path
    cur_data = TecoalResult(cur_path)
    pre_data = TecoalResult(pre_path)

    cur_data.diff_perf(pre_data)
    cur_data.perf_format.append("performance")
    cur_data = cur_data.getLogResult("perf")
    cur_data = pd.DataFrame(cur_data)
    save_name = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    cur_data.to_excel(f"./{save_name}_cmp.xlsx")
