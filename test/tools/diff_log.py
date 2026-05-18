from tecoalResult import TecoalResult
import pandas as pd
import sys

# import matplotlib.pyplot as plt
import argparse
import time
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--pre_path", dest="pre_path", type=str, help="pre_version")
parser.add_argument("--cur_path", dest="cur_path", type=str, help="cur_version")
parser.add_argument("--ignore_list", dest="ignore_list", type=str, help="ignore case")
parser.add_argument("--output", dest="output", type=str, help="output_name")
parser.add_argument(
    "--threshold",
    dest="threshold",
    default=5,
    type=int,
    help="Please input perf threshold",
)
parser.add_argument(
    "--is_ignore",
    dest="is_ignore",
    action="store_true",
    help="Please decide whether to ignore perf result that belows 10us",
)
parser.add_argument(
    "--is_ci",
    dest="is_ci",
    action="store_true",
    help="nothing to say",
)
args = parser.parse_args()

# def my_label(v, data):
#     # print(v)
#     abso = v * sum(data) / 100
#     return "{:.1f}%\n({:d})".format(v, int(abso))

# def sub_draw(labels1, x, ax1, fig, fg_name, colors=None):
#     # print(x)
#     sum1 = sum(x)
#     # autopct = []
#     # for v in x:
#     #     pct = v / sum1
#     #     autopct.append("{:.1f}%\n({:d})".format(pct, int(v)))
#     explode = [0.1] * len(x)
#     if colors is not None:
#         ax1.pie(
#             x,
#             labels=labels1,
#             autopct=lambda v: my_label(v, x),
#             colors=colors,
#             explode=explode,
#             radius=1.5,
#         )
#     else:
#         ax1.pie(
#             x,
#             labels=labels1,
#             autopct=lambda v: my_label(v, x),
#             explode=explode,
#             radius=1.2,
#         )
#     # for i, val in enumerate(x):
#     #     cur_label = f'{labels1[i]}: {int(x[i])}'
#     #     ax1.annotate(f'{labels1[i]}: {int(x[i])}', xy=(2, 5), xytext=(i*0.5+2,i*0.5+2), textcoords='data',arrowprops=dict(arrowstyle="->"),
#     #             fontsize=7, va='center')
#     fig.savefig(fg_name)
#     plt.close(fig)

# def draw(data):
#     fig = plt.figure()
#     ##pie
#     labels1 = ["rise", "flat", "fall_back"]
#     x = []
#     for label in labels1:
#         x.append(float(data[0][label][0]))
#     fig1, ax1 = plt.subplots()
#     colors = ["#90EE90", "#8FA7E8", "#FF4500"]
#     fg_name = f"./summary.png"
#     sub_draw(labels1, x, ax1, fig1, fg_name, colors)

#     fall_op_name = []
#     fall_op_count = []
#     fall_op_time = []

#     rise_op_name = []
#     rise_op_count = []
#     rise_op_time = []

#     for i in range(len(data[1])):
#         op = data[1][i]["op_name"]
#         if data[1][i]["tot_rise_count"] != "":
#             rise_op_name.append(op)
#             rise_op_count.append(data[1][i]["tot_rise_count"])
#             rise_op_time.append(data[1][i]["tot_rise_hardware_time"])
#         if data[1][i]["tot_fallback_count"] != "":
#             fall_op_name.append(op)
#             fall_op_count.append(data[1][i]["tot_fallback_count"])
#             fall_op_time.append(data[1][i]["tot_fallback_hardware_time"])

#     fig2, ax2 = plt.subplots()
#     fg_name = "./rise details.png"
#     sub_draw(rise_op_name, rise_op_count, ax2, fig2, fg_name)

#     fig3, ax3 = plt.subplots()
#     fg_name = "./fallback_details.png"
#     sub_draw(fall_op_name, fall_op_count, ax3, fig3, fg_name)
#     plt.show()

if __name__ == "__main__":
    save_name = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + "cmp"
    pre_path = args.pre_path
    cur_path = args.cur_path
    threshold = args.threshold
    is_ignore = args.is_ignore
    is_ci = args.is_ci
    ignore_list = args.ignore_list
    ignore_cases = {}
    if ignore_list:
        with open(ignore_list, "r") as f:
            for line in f.readlines():
                line = line.strip()
                ignore_cases[line] = 1
    cur_data = TecoalResult(cur_path)
    pre_data = TecoalResult(pre_path)
    diff_result = cur_data.diff_perf(pre_data, threshold=threshold, is_ignore=is_ignore)
    cur_data.diff_accu(pre_data, is_ci=is_ci)
    cur_data.accu_format = [
        "case_path",
        "op_name",
        "output",
        "shape",
        "kernel_details",
        "kernel_details(base_line)",
        "accu_performance",
        "diff_err",
        "diff_err(base_line)",
        "kernel_diff",
    ]
    cur_data.perf_format = [
        "case_path",
        "op_name",
        "shape",
        "kernel_details",
        "kernel_details(base_line)",
        "interface_time(ms)",
        "interface_time(base_line)",
        "hardware_time(ms)",
        "hardware_time(base_line)",
        "min_hardwaretime_diff(ms)",
        "max_hardwaretime_diff(ms)",
        "hardwaretime_iou",
        "cache_miss_details",
        "cache_miss_details(base_line)",
        "cache_miss_time(ns)",
        "cache_miss_time(ns)(base_line)",
        "performance",
        "rate(%)",
        "kernel_diff",
    ]
    diff_status_perf = True
    diff_status_accu = True
    cur_data_accu = cur_data.getLogResult("accu")
    for accu_res in cur_data_accu:
        val = accu_res.get("accu_performance", "")
        if val and val == "fall_back":
            diff_status_accu = False
            break
    cur_data_perf = cur_data.getLogResult("perf")
    for perf_res in cur_data_perf:
        case_path = perf_res["case_path"]
        if ignore_cases and ignore_cases.get(case_path, 0):
            perf_res["performance"] = "ignore"
    cur_data_perf = pd.DataFrame(cur_data_perf)
    cur_data_accu = pd.DataFrame(cur_data_accu)
    if args.output is not None:
        save_name = args.output
    if not save_name.endswith(".xlsx"):
        save_name += ".xlsx"
    if len(diff_result[4]) > 0:
        diff_status_perf = False
    sum0 = pd.DataFrame([diff_result[0]])
    sum1 = pd.DataFrame(diff_result[1])
    sum2 = pd.DataFrame(diff_result[2])
    sum3 = pd.DataFrame(diff_result[3]) if len(diff_result[3]) > 0 else []
    sum4 = pd.DataFrame(diff_result[4]) if len(diff_result[4]) > 0 else []
    with pd.ExcelWriter(f"./{save_name}", engine="openpyxl") as writer:
        sum0.to_excel(writer, sheet_name="summary", index=False)
        sum1.to_excel(writer, sheet_name="op_summary", index=False)
        sum2.to_excel(writer, sheet_name="op_details", index=False)
        if isinstance(sum3, pd.DataFrame):
            sum3.to_excel(writer, sheet_name="rise_casepath", index=False)
        if isinstance(sum4, pd.DataFrame):
            sum4.to_excel(writer, sheet_name="fallback_casepath", index=False)
        cur_data_perf.to_excel(writer, sheet_name="diff_perf", index=False)
        cur_data_accu.to_excel(writer, sheet_name="diff_accu", index=False)
    if diff_status_perf and diff_status_accu:
        print("Not exists fallback")
        sys.exit(0)
    else:
        if not diff_status_perf:
            print("Exists perf fallback")
        if not diff_status_accu:
            print("Exists accu fallback")
        sys.exit(-1)
