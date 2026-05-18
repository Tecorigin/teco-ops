from tecoalResult import TecoalResult
import pandas as pd
import argparse
import time
import os


def merge_excel(excel_dirs, save_name=None):
    output_data = TecoalResult()
    for excel_file in os.listdir(excel_dirs):
        if excel_file.endswith(".xlsx"):
            excel_file_path = os.path.join(excel_dirs, excel_file)
            cur_data = TecoalResult(excel_file_path)
            output_data.extend(cur_data)
    accu_data = pd.DataFrame(output_data.getLogResult("accu"))
    perf_data = pd.DataFrame(output_data.getLogResult("perf"))
    summary = pd.DataFrame(output_data.summary) if output_data.summary else []
    failed_cases = (
        pd.DataFrame(output_data.failed_casepath) if output_data.failed_casepath else []
    )
    op_details = pd.DataFrame(output_data.op_details) if output_data.op_details else []
    kernel_details = (
        pd.DataFrame(output_data.kernel_details) if output_data.kernel_details else []
    )

    if not save_name:
        save_name = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    with pd.ExcelWriter(f"{save_name}.xlsx", engine="openpyxl") as writer:
        if isinstance(summary, pd.DataFrame):
            summary.to_excel(writer, sheet_name="summary", index=False)
        if isinstance(failed_cases, pd.DataFrame):
            failed_cases.to_excel(writer, sheet_name="failed_casepath", index=False)
        if isinstance(kernel_details, pd.DataFrame):
            kernel_details.to_excel(writer, sheet_name="kernel_details", index=False)
        if isinstance(op_details, pd.DataFrame):
            op_details.to_excel(writer, sheet_name="op_details", index=False)
        accu_data.to_excel(writer, sheet_name="accu", index=False)
        perf_data.to_excel(writer, sheet_name="perf", index=False)

    with open(f"./{save_name}_accu.json", "w") as f:
        f.write(accu_data.to_json(orient="records"))
    with open(f"./{save_name}_perf.json", "w") as f:
        f.write(perf_data.to_json(orient="records"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", dest="folder_path", type=str, help="case path")
    parser.add_argument("--output", dest="output", type=str, help="output name")

    args = parser.parse_args()

    folder_path = args.folder_path
    output = args.output
    merge_excel(folder_path, output)
