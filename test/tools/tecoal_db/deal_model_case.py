import os, json
from case_parser import ModelParser

cases_list_path = "/data/Hpe_share/project/models/model_information"
data_dir = "./tmp"

'''
根据model case列表的json文件，生成model信息+case信息的入库数据文件
'''

def get_model_list_files(path):
    files = os.listdir(path)
    print(len(files))

    out = []
    for file in files:
        file = os.path.join(path, file)
        if os.path.isfile(file) and file.endswith(".json"):
            out.append(file)
    out.sort()
    return out


def parse_model_file(filename):
    # filename = "/data/Hpe_share/project/models/model_information/dump_info_tecoinfer_ppyoloeplusl_bs128_img640x640_caseCount.json"
    # filename = "/data/Hpe_share/project/models/model_information/dump_info_tecoinfer_yolov5l6_bs1_caseCount.json"
    model_name, img = "", ""
    model_name = filename.split("/")[-1].replace("dump_info_", "").replace("_caseCount.json", "")
    words = model_name.split("_img")
    if len(words) == 2:
        _, img = words
    return model_name, img

def generate_model_case_data(path):
    files = get_model_list_files(path)
    print(f"{len(files)=}")

    all_case_num = 0
    for i, file in enumerate(files):
        print(i, file)
        model_name, img = parse_model_file(file)
        if model_name != "":
            try:
                version = "1.14.0" # todo????
                outfile = file.replace(".json", "") + "_case.json"
                outfile = "{}/{}".format(data_dir, file.split("/")[-1].replace(".json", "") + "_case.json")
                mp = ModelParser(file, model_name, img, version)
                model, cases = mp.run()
                model = model.to_dict()
                cases = [case.to_dict() for case in cases]
                res = {
                    "model_data": model,
                    "case_data": cases,
                }
                with open(outfile, 'w') as f:
                    f.write(json.dumps(res, indent=4))
                print(f"num = {len(cases)}")
                all_case_num += len(cases)
            except Exception as e:
                print(e)

    print(f"{all_case_num=}")


'''
model case数据入库
'''

import requests
def save_model_case(data):
    """
    保存模型测例
    :return:
    """
    with open(data, "r") as f:
        data = json.load(f)
        res = requests.post(url="http://tecode.tecorigin.net/api/testingCase/tecoal/saveModelCase", json=data).json()
        print(res)

def list_cases(data={}):
    """
    查询测例，可分页也可不分页
    :return:
    """
    # data = {
    #     # "page_size": 15,
    #     # "page_no": 1,  # page_size 和 page_no是为了翻页， 可以不传，不传就会返回所有数据
    #     # "op_name": "conv_back",  # 模糊查询
    #     # "dtype": "aaa",  # 模糊查询
    #     # "shape": "aaa",  # 模糊查询
    #     # "layout": "aaa",  # 模糊查询
    #     # "case_path": "aaa",  # 模糊查询
    #     # "hash": "aaa",  # 模糊查询
    #     # "version": "aaa",  # 模糊查询
    #     # "level": 2,  # 精准查询
    #     # "status": 2,  # 模糊查询

    # }
    res = requests.post(url="http://tecode.tecorigin.net/api/testingCase/tecoal/listCase", json=data).json()
    print(f'num = {len(res["data"]["list"])}')
    # for r in res["data"]["list"]:
    #     print(r)
    #     break

def model_case_to_db(path):
    files = os.listdir(path)
    files.sort()
    for i, file in enumerate(files):
        print(i, file)
        file = os.path.join(path, file)
        if os.path.isfile(file) and file.endswith(".json"):
            save_model_case(file)
            list_cases()


'''
其他接口
'''

def get_model_case_count(data={}):
    """
    查询某个或所有模型的所有测例数量
    :return:
    """
    # data = {
    #     # "model_name": "r50_pt_bs14"  # 精准查询，如果不传，会返回所有模型数据。传的话只返回单个模型数据
    # }
    res = requests.post(url="http://tecode.tecorigin.net/api/testingCase/tecoal/getModelTestCaseCount",
                        json=data).json()
    num = 0
    for c in res["data"]:
        print(c["model_name"], c["count"])
        num += int(c["count"])
    print(f"{num=}")

def get_model_case(data={}):
    """
    查询某个或所有模型的所有测例
    :return:
    """
    data = {
        "model_name": "pd_ppocr_rec_bs32",
        "model_version": "1.14.0",
        # "op_name": "",
    }
    res = requests.post(url="http://tecode.tecorigin.net/api/testingCase/tecoal/getModelOpCase",
                        json=data).json()
    
    for c in res["data"]["list"]:
        print(c["case_path"])
    print(len(res["data"]["list"]))



# generate_model_case_data(cases_list_path)
# model_case_to_db(data_dir)
get_model_case_count()
# get_model_case()
