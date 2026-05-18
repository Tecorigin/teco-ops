import os
import json
import numpy as np
import pandas as pd
import math
from collections import defaultdict
from datetime import datetime


# 函数1：读取每个PR的JSON数据，获取unique_path、avg_hardware_time、accu_score存为新JSON格式
def process_pr_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)

    # 提取1-100个测例路径的硬件时间
    hardware_times = []
    example_paths = []

    for key, value in data.items():
        if key.endswith(".prototxt"):
            hardware_time = value["hardware_time(ms)"]
            if isinstance(hardware_time, float):
                hardware_times.append(hardware_time)
            else:
                hardware_times.append(0.0)  # 如果不是FLOAT类型，默认赋值为0
            example_paths.append(key)

    # 计算硬件时间的平均值
    avg_hardware_time = np.mean(hardware_times)

    # 提取唯一的不含文件名的路径地址
    unique_path = os.path.dirname(example_paths[0])

    # 提取accu_score
    accu_score = data["accu_score"]

    return {
        "unique_path": unique_path,
        "avg_hardware_time(ms)": avg_hardware_time,
        "accu_score": accu_score,
    }


# 函数2：基于函数1的输出，计算并存储每个PR的详细信息
def compute_scores(pr_json_data):
    pr_info = {}
    question_id_times = defaultdict(list)
    accu_not = {}
    for key, value in pr_json_data.items():
        unique_path = value["unique_path"]

        # 去掉前缀 '/eco/teco-al/'
        path = unique_path.replace("/eco/teco-al/", "", 1)

        # 通过最后一个 '/' 分割字符串
        parts = path.rsplit("/", 1)
        if len(parts) == 2:
            question_id, algo_id = parts
        else:
            question_id = parts[0]
            algo_id = ""  # 如果没有 '/algoN'，则 algo_id 为空

        # 记录信息
        pr_info[key] = {
            "question_id": question_id,
            "algo_id": algo_id,
            "avg_hardware_time(ms)": value["avg_hardware_time(ms)"],
            "accu_score": float(value["accu_score"]),  # 转换为浮点数
        }

        # 仅当 accu_score = "6" 时，将 avg_hardware_time(ms) 添加到 question_id_times 中
        if value["accu_score"] == "6.00000":
            question_id_times[question_id].append(value["avg_hardware_time(ms)"])
        else:
            accu_not[key] = 1
    # 计算每个 question_id 的最短 avg_hardware_time(ms)
    min_times = {qid: min(times) for qid, times in question_id_times.items()}

    # 为每个 pr 计算 perf_score
    for key, info in pr_info.items():
        question_id = info["question_id"]
        current_time = info["avg_hardware_time(ms)"]
        min_time = min_times.get(
            question_id, float("inf")
        )  # 如果没有符合条件的记录，默认设置为无穷大
        #精度不通过则性能分为0
        if accu_not.get(key, -1) == 1:
            pr_info[key]["perf_score"] = 0
            continue

        perf_score = 0  # Initialize perf_score with a default value

        # 计算 perf_score
        if question_id in [
            "index_put/int32",
            "unary_ops/int32",
            "arg_max/int32",
            "masked_select/bfloat16",
            "logical_not_tensor/bfloat16",
        ]:  
            perf_score = min_time / current_time if current_time > 0 else 0
            perf_score = 4 * perf_score

        elif question_id == "arg_max/half":
            if current_time >= 81.1873:
                pr_info[key]["perf_score"] = 0
                continue
            best_perf_rate = (81.1873 - min_time) / 81.1873
            current_perf_rate = (81.1873 - current_time) / 81.1873
            perf_score = current_perf_rate / best_perf_rate
            perf_score = 4 * perf_score

        elif question_id == "activation_backward/silu/half":
            if current_time >= 1.2993:
                pr_info[key]["perf_score"] = 0
                continue
            best_perf_rate = (1.2993 - min_time) / 1.2993
            current_perf_rate = (1.2993 - current_time) / 1.2993
            perf_score = current_perf_rate / best_perf_rate
            perf_score = 4 * perf_score

        elif question_id == "conv_forward/rs_not_1":
            if current_time >= 18.6761:
                pr_info[key]["perf_score"] = 0
                continue
            best_perf_rate = (18.6761 - min_time) / 18.6761
            current_perf_rate = (18.6761 - current_time) / 18.6761
            perf_score = current_perf_rate / best_perf_rate
            perf_score = 14 * perf_score

        elif question_id == "activation_forward/silu/half":
            if current_time >= 0.8614:
                pr_info[key]["perf_score"] = 0
                continue
            best_perf_rate = (0.8614 - min_time) / 0.8614
            current_perf_rate = (0.8614 - current_time) / 0.8614
            perf_score = current_perf_rate / best_perf_rate
            perf_score = 9 * perf_score

        elif question_id == "scatter_nd_add/int32":
            if current_time >= 109.2245:
                pr_info[key]["perf_score"] = 0
                continue
            best_perf_rate = (109.2245 - min_time) / 109.2245
            current_perf_rate = (109.2245 - current_time) / 109.2245
            perf_score = current_perf_rate / best_perf_rate
            perf_score = 14 * perf_score

        # 更新信息字典
        pr_info[key]["perf_score"] = perf_score

    # 计算总分，并为每个 pr 添加排名
    for key, info in pr_info.items():
        # 计算总分
        total_score = info["perf_score"] + info["accu_score"]
        info["total_score"] = total_score

    # 排序
    sorted_prs = sorted(
        pr_info.items(), key=lambda x: x[1]["total_score"], reverse=True
    )
    # 为每个 pr 添加排名字段
    for rank, (key, info) in enumerate(sorted_prs, start=1):
        info["rank"] = rank
    
    # 将排名添加回 pr_info
    pr_info = {key: info for key, info in sorted_prs}
    return pr_info


# 函数3：基于函数2的输出，计算每个用户的最高分记录并生成排名
def person_score(pr_info):
    # 用于存储用户的最高分记录
    user_best_scores = defaultdict(
        lambda: defaultdict(lambda: {"total_score": float("-inf"), "key": None})
    )

    # 用于存储去重后的最高分记录
    deduplicated_scores = {}

    # 遍历所有 pr 记录，更新每个用户在每个赛题上的最高分记录
    for key, info in pr_info.items():
        algo_id = info["algo_id"]
        question_id = info["question_id"]
        total_score = info["total_score"]
        # 更新用户在该赛题上的最高分记录
        if total_score > user_best_scores[algo_id][question_id]["total_score"]:
            user_best_scores[algo_id][question_id] = {
                "total_score": total_score,
                "key": key,
            }

    # 保留每个algo_id和question_id的最高分记录
    for algo_id, questions in user_best_scores.items():
        for question_id, record in questions.items():
            deduplicated_scores[record["key"]] = pr_info[record["key"]]

    # 用于存储用户的最终得分和完成的赛题列表
    user_scores = defaultdict(lambda: {"completed_questions": [], "total_score_sum": 0})

    # 遍历用户最佳分记录，计算每个用户的总得分
    for key, info in deduplicated_scores.items():
        algo_id = info["algo_id"]
        question_id = info["question_id"]
        total_score = info["total_score"]

        # 更新 total_score_sum
        user_scores[algo_id]["total_score_sum"] += total_score

        # 更新 completed_questions 列表，包含 question_id 及其得分
        user_scores[algo_id]["completed_questions"].append(
            {"question_id": question_id, "score": total_score}
        )

    # 排序用户，依据总得分进行排序
    sorted_users = sorted(
        user_scores.items(), key=lambda x: x[1]["total_score_sum"], reverse=True
    )

    # 生成排名信息
    ranked_users = []
    for rank, (algo_id, info) in enumerate(sorted_users, start=1):
        ranked_users.append(
            {
                "algo_id": algo_id,
                "completed_questions": info["completed_questions"],
                "total_score_sum": info["total_score_sum"],
                "rank": rank,
            }
        )

    return ranked_users


# 函数4：基于函数2的输出，计算每个赛题的最高分记录并生成排名
def question_score(pr_info):
    deduplicated_scores = {}

    # 遍历 pr_info，更新每个 algo_id 下每个 question_id 的最高得分信息
    for pr_id, info in pr_info.items():
        algo_id = info["algo_id"]
        question_id = info["question_id"]
        total_score = info["total_score"]

        # 更新最高得分信息
        cur_idx = f"{algo_id}+{question_id}"
        cur_aq_info = deduplicated_scores.get(cur_idx, {})
        current_score = cur_aq_info.get("total_score", -float("inf"))  # 初始化为负无穷大，以确保任何分数都会被更新
        if total_score >= current_score:  # 更新逻辑，确保即使 total_score 为0也会被记录
            deduplicated_scores[cur_idx] = {
                "question_id": question_id,
                "algo_id": algo_id,
                "pr_id": pr_id,
                "avg_hardware_time(ms)": info["avg_hardware_time(ms)"],
                "accu_score": info["accu_score"],
                "perf_score": info["perf_score"],
                "total_score": total_score,
            }

    deduplicated_ranks = sorted(deduplicated_scores.items(), key=lambda x: x[1]["total_score"], reverse=True)

    # 生成新的排名并添加 deduplicate_rank 字段
    deduplicate_rank = 1
    deduplicated_ranked_data = []
    for rank, (idx_info, info) in enumerate(deduplicated_ranks):
        info["rank"] = deduplicate_rank
        deduplicate_rank += 1
        deduplicated_ranked_data.append(info)

    # 根据 question_id 分组排名
    grouped_data = {}
    for info in deduplicated_ranked_data:
        question_id = info["question_id"]
        if question_id not in grouped_data:
            grouped_data[question_id] = []
        grouped_data[question_id].append(info)

    # 对每个 question_id 内的记录进行排名
    for question_id, records in grouped_data.items():
        records.sort(key=lambda x: x["total_score"], reverse=True)
        for rank, record in enumerate(records):
            record["group_rank"] = rank + 1

    # 生成 question_id 的分组排名数据
    group_ranked_data = []
    for records in grouped_data.values():
        group_ranked_data.extend(records)

    return deduplicated_ranked_data, group_ranked_data

# 生成Markdown文件内容

def dataframe_to_markdown(df):
    header = '| ' + ' | '.join(df.columns) + ' |\n'
    separator = '|---' * len(df.columns) + '|\n'
    rows = ''
    for _, row in df.iterrows():
        rows += '| ' + ' | '.join(str(value) for value in row) + ' |\n'
    return header + separator + rows

def generate_markdown(ci_time, df_users, df_questions, df_distinct_questions):
    # Remove 'rank' column from df_distinct_questions
    if 'rank' in df_distinct_questions.columns:
        df_distinct_questions = df_distinct_questions.drop(columns=['rank'])
    
    # 分组按 question_id 列
    grouped = df_distinct_questions.groupby('question_id')
    
    # 初始化 markdown 内容
    markdown = f"""
**当前PR排名信息（algo_id = user_id）**

**各题分组排名结果**

"""
    
    # 遍历每个分组
    for question_id, group in grouped:
        markdown += f"""
**{question_id} 排名结果**

{dataframe_to_markdown(group)}

"""

#     markdown +=f"""
# **单人总分排名结果**

# {dataframe_to_markdown(df_users)}

#  **单人单题排名结果** 

# {dataframe_to_markdown(df_questions)}
# """
    
    return markdown

# 主函数：依次调用函数1、2、3
def main(directory_path, output_base_path):
    pr_json_data_list = {}
    score_data_list = []

    # 遍历目录中的所有JSON文件
    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            if filename.endswith(".json"):
                file_path = os.path.join(root, filename)

                # 调用函数1
                pr_json_data = process_pr_json(file_path)
                pr_json_data_list[filename] = pr_json_data

    # 调用函数2
    score_data = compute_scores(pr_json_data_list)

    # 保存函数2的结果
    score_data_list.extend(score_data.values())

    # 获取当前时间，创建新的文件夹
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    ci_time = datetime.now().strftime("%Y%m%d")
    output_dir = os.path.join(output_base_path, current_time)
    os.makedirs(output_dir, exist_ok=True)

    # 保存函数2的输出为临时文件
    temp_output_file_path = os.path.join(output_dir, "pr_record_with_rank.json")
    with open(temp_output_file_path, "w") as temp_output_file:
        json.dump(score_data, temp_output_file, indent=4)

    # 调用函数3
    ranked_users = person_score(score_data)


    # 调用函数4结果
    ranked_questions = question_score(score_data)[0]
    ranked_distinct_questions = question_score(score_data)[1]
        
    # 将 JSON 数据加载到 pandas 的 DataFrame 中
    df_users = pd.DataFrame(ranked_users)
    df_questions = pd.DataFrame(ranked_questions)
    df_distinct_questions = pd.DataFrame(ranked_distinct_questions)

    # 创建 ExcelWriter 对象，指定输出路径
    output_excel_path = os.path.join(output_dir, "ranked_data.xlsx")
    with pd.ExcelWriter(output_excel_path) as writer:
        # 将 DataFrame 写入不同的 sheet
        df_users.to_excel(writer, sheet_name="users_rank", index=False)
        df_questions.to_excel(writer, sheet_name="questions_rank", index=False)
        df_distinct_questions.to_excel(writer, sheet_name="distinct_questions_rank", index=False)

    markdown_content = generate_markdown(ci_time, df_users, df_questions, df_distinct_questions)
    markdown_file_path = os.path.join(output_dir, f"{current_time}_ranking_info.md")
    with open(markdown_file_path, "w") as md_file:
        md_file.write(markdown_content)

    print(f"Final output saved: {output_dir}")


# 运行主函数
if __name__ == "__main__":
    directory_path = "/eco/ci_result/teco_al"
    output_base_path = "/eco/ci_result/rank_result"
    main(directory_path, output_base_path)
