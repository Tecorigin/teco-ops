import logging
from getTecoalNode import getTecoalNode
import os
import sys
import random
import pandas as pd
import time
import asyncio
import psutil
from tecoalNode import TecoalNode
from tqdm import tqdm
import shutil
import hashlib

# support list, dir, txt, prototxt/pb
# !!! not check for file's existence


def getOpList():
    filepath = os.path.split(os.path.realpath(__file__))[0]
    result = os.popen(
        'find {}/../zoo/teco/ -mindepth 1 -maxdepth 1 -type d ! -name "__pycache__"'.format(
            filepath
        )
    )
    tmp = result.read().strip().split("\n")
    result.close()
    tmp = [cur[cur.rfind("/") + 1 :] for cur in tmp]
    tecoal_api_list = tmp


    return tecoal_api_list 


def getCasesList(file_path):
    # start = time.time()
    if isinstance(file_path, list):
        return file_path

    cases_list = []
    # prototxt / pb
    if file_path.endswith(".pb") or file_path.endswith(".prototxt"):
        cases_list.append(file_path)

    # txt
    if file_path.endswith(".txt"):
        if not os.path.exists(file_path):
            logging.warning("{} is not exist".format(file_path))
            return []

        with open(file_path, "r") as f:
            # cases_list.extend(f.read().strip().split("\n"))
            cases_dict = {line.strip(): 1 for line in f.readlines() if line.strip()}
            cases_list = list(cases_dict.keys())

    # dir
    if os.path.isdir(file_path):
        result = os.popen('find {} -name "*pb" -or -name "*prototxt"'.format(file_path))
        cases_list = result.read().strip().split("\n")
        result.close()
    # end = time.time()
    # print(end-start)
    return cases_list


def genHash(prototxt_file):
    hash_object = hashlib.md5()
    hash_object.update(prototxt_file.encode())
    return hash_object.hexdigest()


async def queue_put_proto(queue, prototxt_dir):
    op_list = getOpList()
    pb_dirs = os.path.abspath("./pb_dirs")
    if not os.path.exists(pb_dirs):
        os.mkdir(pb_dirs)
    for prototxt_file in prototxt_dir:
        # print(prototxt_file)
        op_path = ""
        for op_name in prototxt_file.split("/"):
            if op_name in op_list:
                op_path = os.path.join(pb_dirs, op_name)
                if not os.path.exists(op_path):
                    # op_dirs.append(op_path)
                    os.mkdir(op_path)
                break
        proto_name = genHash(prototxt_file)
        pb_name = f"temp_{proto_name}.pb"
        # print(pb_name)
        pb_file = os.path.join(op_path, pb_name)
        await queue.put([prototxt_file, pb_file])


async def cmd_get_node_async(cmd, time_out):
    try:
        process = await asyncio.create_subprocess_shell(cmd)
        await asyncio.wait_for(process.wait(), timeout=time_out)
    except asyncio.TimeoutError:
        print(f"cmd : {cmd} timed out")
        children = psutil.Process(process.pid).children(recursive=True)
        for child in children:
            child.terminate()
        for child in children:
            child.wait()
        process.kill()
        await process.terminate()


async def worker_prototxt_to_pb(time_out, queue, done_event, pb_list, pbar):
    while True:
        try:
            # print(queue.qsize())
            res = await queue.get()
            if res is None:
                break
            pt_file, pb_file = res
            # grep_demo_cmd = "ps u |grep demo|grep -v /bin/sh"
            # grep_demo_cmd_result = subprocess.uun(grep_demo_cmd, shell=True, capture_output=True, text=True)
            # if grep_demo_cmd_result.returncode == 0:
            #     output = [line for line in grep_demo_cmd_result.stdout.split("\n") if line.strip()]
            #     print("coroutine num", len(output))
            proto2pb = (
                os.path.split(os.path.realpath(__file__))[0]
                + "/../test_tools2/prototxt2pb"
            )
            cmd = f"{proto2pb} {pt_file} {pb_file}"
            # print(cmd)
            await cmd_get_node_async(cmd, time_out)
            if os.path.exists(pb_file):
                pb_list[pt_file] = getTecoalNode(pb_file)
                os.remove(pb_file)
            pbar.update(1)
            queue.task_done()
            done_event.set()
        except asyncio.QueueEmpty:
            break


async def worker_ptTopb_run(queue, workers, done_event):
    try:
        await queue.join()
        for _ in workers:
            await queue.put(None)
        await asyncio.gather(*workers, return_exceptions=True)
        pb_dirs = os.path.abspath("./pb_dirs")
        if os.path.isdir(pb_dirs):
            shutil.rmtree(pb_dirs)
        # for op_dir in op_dirs:
        #     if os.path.exists(op_dir):
        #         os.rmdir(op_dir)
        done_event.clear()
    except Exception as e:
        print(f"Exception as {e}")


async def create_worker_ptTopb(queue, tasks, done_event, pt_dir, pb_list, pbar):
    await queue_put_proto(queue, pt_dir)
    for _ in range(min(16, len(pt_dir))):
        worker_task = asyncio.create_task(
            worker_prototxt_to_pb(10, queue, done_event, pb_list, pbar)
        )
        tasks.append(worker_task)
        # print(i)
    await worker_ptTopb_run(queue, tasks, done_event)


async def check_file_exists(file_list, exists_list):
    for file in file_list:
        file = os.path.abspath(file)
        if os.path.exists(file):
            exists_list.append(file)
        else:
            logging.warning("{} is not exist".format(file))


async def create_check_file_coro(tot_file_list, exists_cases_list):
    chunk_size = len(tot_file_list) // 16 + 1
    exists_lists = [[] for _ in range(16)]
    tasks = []
    for i in range(16):
        last_pos = i * chunk_size
        if last_pos >= len(tot_file_list):
            break
        cur_pos = (
            (i + 1) * chunk_size
            if chunk_size < len(tot_file_list) - last_pos
            else len(tot_file_list)
        )
        chunk = tot_file_list[last_pos:cur_pos]
        task = asyncio.create_task(check_file_exists(chunk, exists_lists[i]))
        tasks.append(task)
        if cur_pos == len(tot_file_list):
            break
    await asyncio.gather(*tasks)
    # with open("./exists_cases_list.txt", "w") as f:
    for exists_list in exists_lists:
        if exists_list:
            exists_cases_list.extend(exists_list)
        # f.write("\n".join(exists_list))


class TecoalCase:
    nodes = []
    cases_list = []
    subprocess_flag = False
    api_list = {}

    # cases op: add, delete, query(not need)
    def addCase(self, file_path):
        if file_path is None:
            return

        if isinstance(file_path, TecoalCase):
            self.nodes.extend(file_path.nodes)
            self.cases_list(file_path.cases_list)

        cases_list = getCasesList(file_path)
        # with open("./pre_cases.txt", "w") as f:
        #     f.write("\n".join(cases_list))
        start = time.time()
        # for i, case_path in enumerate(cases_list):
        #     if not os.path.isfile(case_path):
        #         logging.warning("{} is not exist".format(case_path))
        #         continue
        #     self.cases_list.append(os.path.abspath(case_path))
        # if case_path[0] == '/':
        #     self.cases_list.append(case_path)
        # else:
        #     self.cases_list.append(os.path.abspath(case_path))
        # self.nodes.append(getTecoalNode(case_path))
        if self.subprocess_flag:
            print("start check file exists")
            asyncio.run(create_check_file_coro(cases_list, self.cases_list))
        else:
            self.cases_list = cases_list
        end = time.time()
        print(end - start)

    def append(self, case):
        self.addCase(case)

    def extend(self, cases):
        if isinstance(cases, TecoalCase):
            self.nodes.extend(cases.nodes)
            self.cases_list.extend(cases.cases_list)
        else:
            logging.error("extend input is error")

    def get_case_node(
        self,
    ):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        queue = asyncio.Queue()
        done_event = asyncio.Event()
        task = []
        pb_list = {}
        pbar = tqdm(
            total=len(self.cases_list), desc="translated_cases", leave=True, ncols=100
        )
        loop.run_until_complete(
            create_worker_ptTopb(
                queue, task, done_event, self.cases_list, pb_list, pbar
            )
        )
        pb_dirs = os.path.abspath("./pb_dirs")
        if os.path.isdir(pb_dirs):
            shutil.rmtree(pb_dirs)
        # print("start get nodes")
        # start = time.time()
        for i, case_file in enumerate(self.cases_list):
            if case_file in pb_list:
                self.nodes.append(pb_list[case_file])
            else:
                logging.warning(f"{case_file} can not be translated to pb")
                self.nodes.append(TecoalNode())
        # end = time.time()
        # print(end-start)
        loop.close()

    def initCase(self, file_path):
        self.nodes = []
        self.cases_list = []
        self.addCase(file_path)

    def deleteIndex(self, index):
        if isinstance(index, int):
            self.cases_list.pop(index)
            self.nodes.pop(index)
        index = index.copy()
        index.sort(reverse=True)
        for i in index:
            self.cases_list.pop(i)
            self.nodes.pop(i)

    def deleteCase(self, file_path):
        cases_list = getCasesList(file_path)
        index = []
        for i in range(len(self.cases_list)):
            if self.cases_list[i] in cases_list:
                index.append(i)

        self.deleteIndex(index)

    def deleteOp(self, op_name):
        index = []
        for i in range(len(self.nodes)):
            if self.nodes[i].op_name in op_name:
                index.append(i)
        self.deleteIndex(index)

    def filter(self, file_path, op_names, rand_n=None):
        cur_cases = []
        self.cases_list = []
        self.nodes = []
        # print(op_names)
        if isinstance(file_path, TecoalCase):
            cur_cases = file_path.cases_list
        else:
            cur_cases = getCasesList(file_path)
        filter_cases = []
        if rand_n is None:
            assert op_names is not None
            for file in cur_cases:
                # if not os.path.exists(file):
                #     logging.warning("{} is not exist".format(file))
                #     continue
                for op in op_names:
                    if op in file:
                        filter_cases.append(file)
                        break
        else:
            if op_names is not None:
                tmp_cases = []
                for file in cur_cases:
                    # if not os.path.exists(file):
                    #     logging.warning("{} is not exist".format(file))
                    #     continue
                    for op in op_names:
                        if op in file:
                            tmp_cases.append(file)
                            break
                # print(len(tmp_cases))
                rand_n = min(len(tmp_cases), rand_n)
                random.shuffle(tmp_cases)
                filter_cases = tmp_cases[:rand_n]
            else:
                random.shuffle(cur_cases)
                rand_n = min(rand_n, len(cur_cases))
                filter_cases.extend(cur_cases[:rand_n])
                # for file in cur_cases[:rand_n]:
                #     if os.path.exists(file):
                #         self.cases_list.append(file)
                #     else:
                #         logging.warning("{} is not exist".format(file))
        start = time.time()
        if self.subprocess_flag:
            print("start check file exists")
            asyncio.run(create_check_file_coro(filter_cases, self.cases_list))
        else:
            self.cases_list = filter_cases
        end = time.time()
        print(end - start)
        # for file in self.cases_list:
        #     self.nodes.append(getTecoalNode(file))

    def __init__(
        self,
        file_path,
        op_names=None,
        rand_n=None,
        subprocess_flag=False,
    ):
        print("start find prototxt")
        self.api_list = {key: 1 for key in getOpList()}
        self.subprocess_flag = subprocess_flag
        if op_names is None and rand_n is None:
            self.initCase(file_path)
        else:
            self.filter(file_path, op_names, rand_n)

        print("start transalte pb")
        if subprocess_flag:
            self.get_case_node()
        save_name = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        # with open(f'./{save_name}.txt', 'w') as f:
        #     for file in self.cases_list:
        #         f.write(file)
        #         f.write("\n")
        # print(os.path.realpath(f"{save_name}.txt"))

    def toDict(self):
        nodes_dict = []
        for node in self.nodes:
            nodes_dict.append(node.toDict())
        return nodes_dict

    def saveTo(self, dst_dir):
        if os.isfile(dst_dir):
            if len(self.nodes) == 1:
                self.nodes[0].saveTo(dst_dir)
            else:
                logging.warning("can not save total proto in dst_dir")

        for i in range(len(self.nodes)):
            node = self.nodes[i]
            cur_dst_dir = dst_dir + "/" + node.op_name
            dst_file = cur_dst_dir + "/" + "case_{}.prototxt".format(i)
            j = 1
            while True:
                if os.path.exists(dst_file):
                    dst_file = cur_dst_dir + "/" + "case_{}({}).prototxt".format(i, j)
                    j += 1
                else:
                    break

            node.saveTo(dst_file)

    def caseNum(self):
        return len(self.nodes)

    def saveNeedMemory(self):
        size = 0
        for node in self.nodes:
            size += node.saveNeedMemory()

        return size

    def toDataFrame(self, op_name=None, mode="simple"):
        data = []
        for i in range(self.caseNum()):
            record = self.nodes[i].toSimpleRecord()
            record["case_path"] = self.cases_list[i]
            data.append(record)

        return pd.DataFrame(data)

    def getOpCaseNum(self, op_name):
        pass

    def get_case_info(self):
        pass

    def __str__(self):
        return str(self.cases_list)
