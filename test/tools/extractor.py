import json

'''
该脚本统计trace的json 文件中出现的cuda/cpu aten级别接口统计
包括每个aten级别api分支所占时间，以及总时间统计
'''

def get_cuda_ops_perf(json_file_path: str, device: str) -> dict:
    op_time = {}
    op_time["tracing"] = []
    op_time["summary"] = {}
    with open(json_file_path, 'r') as f:
        traces = json.load(f)
        if "traceEvents" in  traces.keys():
            tmp = traces["traceEvents"]
            for l in tmp:
                if 'name' in l.keys():
                    if l["name"] == "cudaLaunchKernel":
                        op_time["tracing"].append({"op_name":l["name"],"time":l["dur"]})
    sum_time = 0
    for i in op_time["tracing"]:
        sum_time += i["time"]
    op_time["summary"]["total_device_time"] = sum_time
    op_time["summary"]["device"] = device
    json.dump(op_time,open(json_file_path,"w"))
