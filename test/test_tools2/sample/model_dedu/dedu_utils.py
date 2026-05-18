import pandas as pd
import sys
import os
import hashlib
import json
import argparse
import time
import numpy as np
import subprocess
import shutil
import stat
import asyncio
from prototxt_parser.prototxt_parser_main import parse
from tqdm import tqdm
from node_utils import getTecoalNode

# parser = argparse.ArgumentParser()
# parser.add_argument("--cases_dir", dest="cases_dir", type=str, help="cases_dir")
# parser.add_argument("--model_name", dest="model_name", type=str, help="model_name")
# args = parser.parse_args()

cur_dir = os.path.split(os.path.abspath(__file__))[0]
prototxt2pb_path = f"{cur_dir}/../../prototxt2pb"


# save_dir = "/data/dnn/heqq/model_dump/deduplicate"
def genHash(prototxt_file):
    hash_object = hashlib.md5()
    hash_object.update(prototxt_file.encode())
    return hash_object.hexdigest()


async def cmd_pb_node_async(cmd, child_env, time_out):
    try:
        process = await asyncio.create_subprocess_exec(*cmd, env=child_env)
        await asyncio.wait_for(process.wait(), timeout=time_out)
    except asyncio.TimeoutError:
        print(cmd)
        os.kill(process.pid, 9)
        await process.wait()
    # finally:
    #     print(cmd)
    #     os.kill(process.pid, 9)
    #     await process.wait()


async def queue_put_ptfile(queue, pt_files):
    for file in pt_files:
        # content = read_prototxt(file)
        await queue.put(file)


async def creater_worker_ptdedu(queue, tasks, pt_files, pt_content, pbar, time_out):
    print("start put queue")
    await queue_put_ptfile(queue, pt_files)
    print("end put queue")
    for _ in range(min(8, len(pt_files))):
        woker_task = asyncio.create_task(
            worker_pt_dedu(queue, pt_content, pbar, time_out)
        )
        tasks.append(woker_task)
    await worker_ptdedu_run(queue, tasks)


async def worker_ptdedu_run(queue, workers):
    try:
        await queue.join()
        for _ in workers:
            await queue.put(None)
        await asyncio.gather(*workers, return_exceptions=True)
    except Exception as e:
        print(f"Exception as {e}")


async def worker_pt_dedu(queue, pt_pb, pbar, time_out):
    while True:
        try:
            pt_file = await queue.get()
            child_env = os.environ.copy()
            save_name = genHash(pt_file)
            pb_file = f"{cur_dir}/{save_name}.pb"
            cmd = [f"{prototxt2pb_path}", pt_file, pb_file]
            await cmd_pb_node_async(cmd, child_env, time_out)
            if os.path.exists(pb_file):
                pt_pb[pt_file] = getTecoalNode(pt_file, pb_file)
                os.remove(pb_file)
                pbar.update(1)
                queue.task_done()
        except asyncio.QueueEmpty():
            break


def check_dim_stride(dim_stride):
    for i, stride in enumerate(dim_stride):
        stride = stride if stride >= 0 else 0
        dim_stride[i] = stride


def find_prototxt_files(root_dir):
    if os.path.isdir(root_dir):
        print("is dir")
        cases_list = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith(".prototxt"):
                    cases_list.append(os.path.join(dirpath, filename))
        yield from cases_list
    elif os.path.isfile(root_dir):
        if root_dir.endswith(".prototxt"):
            return [root_dir]
        elif root_dir.endswith(".txt"):
            with open(root_dir, "r") as f:
                cases_list = [line.strip() for line in f.readlines() if line.strip()]
                yield from cases_list
        else:
            print(f"not support type of {root_dir}")
            return []

    else:
        print(f"{root_dir} not exist")
        return []


class ModelCases:
    input_tensor = []
    output_tensor = []
    interface_param = {}
    op_name = ""

    def add_tensor_data(self, tecoal_tensor):
        ##only support node.input/output
        record = ""
        shape = tecoal_tensor.shape
        # empty_list = []
        dims = shape.dims
        dim_stride = shape.dim_stride
        if not isinstance(dims, list):
            dims = list(dims)
        if not isinstance(dim_stride, list):
            dim_stride = list(dim_stride)
        dims = "_".join([str(dim) for dim in dims])
        check_dim_stride(dim_stride)
        dim_stride = "_".join([str(stride) for stride in dim_stride])
        record += dims + dim_stride
        record += "_" + str(tecoal_tensor.ttype)
        record += "_" + str(tecoal_tensor.reused)
        record += str(tecoal_tensor.inplace)
        record += "_" + str(tecoal_tensor.dtype)
        record += "_" + str(tecoal_tensor.layout) + "_"
        return record

    def __init__(self, case_dic):
        self.op_name = case_dic.op_name
        self.input_tensor = []
        self.output_tensor = []
        self.interface_param = {}
        for ipt in case_dic.node.input:
            self.input_tensor.append(self.add_tensor_data(ipt))
        for ipt in case_dic.node.output:
            self.input_tensor.append(self.add_tensor_data(ipt))
        if case_dic.node.HasField("dnn_param"):
            self.interface_param = "dnn_param {{\n{}}}".format(case_dic.node.dnn_param)
        elif case_dic.node.HasField("blas_param"):
            self.interface_param = "blas_param {{\n{}}}".format(
                case_dic.node.blas_param
            )
        elif case_dic.node.HasField("custom_param"):
            self.interface_param = "custom_param {{\n{}}}".format(
                case_dic.node.custom_param
            )
            if "custom" not in self.op_name:
                self.op_name = "custom_" + self.op_name

    def to_str(self):
        # return {
        #     "op_name": self.op_name,
        #     "input": self.input_tensor,
        #     "output": self.output_tensor,
        #     "interface": self.interface_param,
        # }
        return (
            self.op_name
            + "#".join(self.input_tensor)
            + "#".join(self.output_tensor)
            + str(self.interface_param)
        )


def getDictData(model_json_path):
    if os.path.exists(model_json_path):
        with open(model_json_path, "r") as f:
            return json.load(f)
    else:
        return {}


def gen_cases_hash_dict(pt_files, pt_content):

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    queue = asyncio.Queue()
    task = []
    pbar = tqdm(total=len(pt_files), desc="read_case_dedu", leave=True, ncols=100)
    loop.run_until_complete(
        creater_worker_ptdedu(queue, task, pt_files, pt_content, pbar, time_out=30)
    )
    loop.close()
