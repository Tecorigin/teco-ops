import logging
import os
import sys
import subprocess
import json
import asyncio
import psutil
from tqdm import tqdm
from tecoalCase import *
from tecoalResult import *
from tecoalLog import merge_log
import signal


def get_case_op(case_path, api_dict):
    ops = case_path.split("/")
    for op in ops[::-1]:
        if val := api_dict.get(op, 0):
            return op

    return "unknown"


async def kill_and_wait_child(child):
    os.kill(child.pid, 9)
    await asyncio.to_thread(child.wait)


async def cmd_time_guard_async(cmd, time_out, log, case_path=None):
    retry_time = 0
    while retry_time < 10:
        try:
            child_env = os.environ.copy()
            with open(log, "w") as f:
                if case_path is not None:
                    cur_cmd = [arg.strip() for arg in cmd.split(" ") if arg.strip()]
                    cur_cmd.append(f"--case_path={case_path}")
                    # print(cmd)
                process = await asyncio.create_subprocess_exec(
                    *cur_cmd,
                    env=child_env,
                    stdout=f,
                    stderr=f,
                )
                await asyncio.wait_for(process.wait(), timeout=time_out)
        except asyncio.TimeoutError:
            print(f"cmd : {cmd} timed out")
            # children = psutil.Process(process.pid).children(recursive=True)
            # for child in children:
            #     await kill_and_wait_child(child)
            #     child.terminate()
            # for child in children:
            #     child.wait()
            # print(process.pid)
            os.kill(process.pid, signal.SIGKILL)
            await process.wait()
        try:
            with open(log, "r") as f:
                content = f.read()
            if content:
                break
            else:
                retry_time += 1
        except Exception as e:
            print(e)
            break


async def worker(cmd, time_out, queue, done_event, pbar):
    while True:
        try:
            # print(queue.qsize())
            cur_res = await queue.get()
            if cur_res is None:
                break
            log, case_path = cur_res[0], cur_res[1]
            # cmd = cmd + '--case_path=' + case_path
            # print(cmd)
            start = time.time()
            grep_demo_cmd = "ps u |grep demo|grep -v /bin/sh"
            grep_demo_cmd_result = subprocess.run(
                grep_demo_cmd, shell=True, capture_output=True, text=True
            )
            if grep_demo_cmd_result.returncode == 0:
                output = [
                    line
                    for line in grep_demo_cmd_result.stdout.split("\n")
                    if line.strip()
                ]
                print("coroutine num", len(output))
            await cmd_time_guard_async(cmd, time_out, log, case_path)
            end = time.time()
            print(f"log:{log}", end - start)
            queue.task_done()
            pbar.update(1)
            done_event.set()
        except asyncio.QueueEmpty:
            break


async def worker_run(queue, workers, done_event):
    try:
        await queue.join()
        for _ in workers:
            await queue.put(None)
        await asyncio.gather(*workers, return_exceptions=True)
        done_event.clear()
    except Exception as e:
        print(f"Exception as {e}")


async def queue_put(queue, cases, logs, log_dir):
    for i in range(len(cases.cases_list)):
        log = (
            log_dir
            + get_case_op(cases.cases_list[i], cases.api_list)
            + "/"
            + str(i)
            + ".log"
        )
        os.system("mkdir -p {}".format(os.path.dirname(log)))
        logs.append(log)
        await queue.put([log, cases.cases_list[i]])


async def cmd_run(cmds, time_out_threshold, logs):
    tasks = [
        cmd_time_guard_async(cmd, time_out_threshold, log)
        for cmd, log in zip(cmds, logs)
    ]
    await asyncio.gather(*tasks)


# time(second)
def cmd_time_guard(cmd, time_out_threshold, log):
    print(os.path.dirname(log))
    os.system("mkdir -p {}".format(os.path.dirname(log)))
    # print(os.environ)
    # tempfile_name = "./log/{}.log".format(str(abs(hash(cmd)))[0:8])
    print("log_name {}".format(log))
    print("cmd: {}".format(cmd))
    # os.system(f"{cmd} > {log}")
    try:
        f = open(log, "w")
        process = subprocess.run(
            cmd, shell=True, text=True, stdout=f, stderr=f, timeout=time_out_threshold
        )

        # time_out_flag = False
        # start = time.time()

        # while (process.poll() is None):
        #     end = time.time()
        #     if (end - start >= time_out_threshold):
        #         time_out_flag = True
        #         process.kill()
        #         break

        f.close()
    except subprocess.TimeoutExpired:
        print(f"{cmd} time out")


class TecoalTest:
    time_out_threshold = 0

    # param
    warm_repeat = -1
    perf_repeat = -1
    gtest_repeat = -1
    async_gids = False
    gids = [0]
    envs = {
        "MKL_SERVICE_FORCE_INTEL": "1",
        "MKL_THREADING_LAYER": "GNU",
        # 'TECO_ENABLE_PROFILING': '1',
        "DNN_RESERVE_DATA": "1",
        "DNN_TEST_LOG": "1",
    }
    test_stable = False
    subprocess_case = False
    use_cuda = False
    merge = False
    cases = None
    op_names = None
    rand_n = None

    def __init__(self):
        pass

    def getTestInfo(self):
        pass

    # envs is dictionary
    def updateEnvParam(self, envs):
        for key in envs.keys():
            self.envs[key] = envs[key]

    def setEnvParam(self, envs):
        self.envs = []
        self.update_env(envs)

    def setEnv(self):
        for key in self.envs.keys():
            os.environ[key] = self.envs[key]
        pass

    def demo(self, gid):
        cmd = os.path.split(os.path.realpath(__file__))[0]
        cmd += "/../build/demo" if not self.use_cuda else "/../build_cuda/demo_cuda"
        cmd += " --gid=" + str(gid)

        if self.perf_repeat != -1:
            cmd += " --perf_repeat=" + str(self.perf_repeat)

        if self.warm_repeat != -1:
            cmd += " --warm_repeat=" + str(self.warm_repeat)

        if self.gtest_repeat != -1:
            cmd += " --gtest_repeat=" + str(self.gtest_repeat)
        if self.test_stable:
            cmd += " --test_stable"
        return cmd

    def update_dic(self, result, time, record, case_path):
        for i in range(len(result)):
            if time != -1:
                result[i]["case_time"] = time
            result[i]["case_path"] = case_path
            for key, val in record.items():
                result[i][key] = val

    def getCases(self, path):
        self.cases = TecoalCase(path)

    def test(
        self,
        case_path,
        warm_repeat=10,
        perf_repeat=100,
        test_stable=False,
        time_out_threshold=300,
    ):
        demo_path = os.path.split(os.path.realpath(__file__))[0] + "/../build/demo"
        if not os.path.exists(demo_path):
            print(f"{demo_path} not exist")
            sys.exit(-1)
        # self.setEnv()
        # print(self.rand_n)
        if self.warm_repeat == -1:
            self.warm_repeat = warm_repeat
        if self.perf_repeat == -1:
            self.perf_repeat = perf_repeat
        if self.time_out_threshold == 0:
            self.time_out_threshold = time_out_threshold
        if case_path is None:
            logging.warning("no case to test")
            return
        cases = TecoalCase(case_path, self.op_names, self.rand_n, self.subprocess_case)

        tid = os.getpid()
        test_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        log_dir = "./log/{}_{}/".format(test_time, tid)

        # result.test_version = self.get_test_info()
        # result.test_time = test_time
        tot_len = len(cases.cases_list)
        result = TecoalResult()
        pbar = tqdm(total=tot_len, desc="cases", leave=True, ncols=100)
        start_1 = time.time()
        result.gids = self.gids
        if len(self.gids) == 1:
            gid = self.gids[0]
            for i in range(len(cases.cases_list)):
                case_path = cases.cases_list[i]
                cmd = self.demo(gid) + " --case_path=" + case_path
                log = (
                    log_dir
                    + get_case_op(cases.cases_list[i], cases.api_list)
                    + "/"
                    + str(i)
                    + ".log"
                )
                # logging.info("[DO]: {}".format(cmd))
                start = time.time()
                cmd_time_guard(cmd, self.time_out_threshold, log)
                end = time.time()
                cur_result = getResult(log)
                if i < len(cases.nodes):
                    record = cases.nodes[i].toSimpleRecord()
                    self.update_dic(cur_result, end - start, record, case_path)
                result.addResult(cur_result)
                pbar.update(1)
                # cur_result[0]["case_time"] = end - start
                # cur_result[0].update(record)

                # log_name = log_dir + op_name + '/' + str(i) + '.log'
                # os.system('mkdir -p {}'.format(os.path.split(os.path.realpath(log_name))[0]))
                # cur_result.update(cmd_time_guard(cmd, self.time_out_threshold, log_name))
                # tmp = os.popen('grep "dtype" {}'.format(case_path))
                # cur_result['dtype'] = tmp.read()
                # tmp.close()

                # result.append(cur_result)
                pass
        else:
            if self.async_gids == False:
                for i in range(len(cases.cases_list)):
                    case_path = cases.cases_list[i]
                    print(f"{i}: {case_path}")
                    cmds, logs = [], []
                    for gid in self.gids:
                        cmd = self.demo(self.gids[i]) + f" --case_path={case_path}"
                        log = (
                            log_dir
                            + get_case_op(case_path, cases.api_list)
                            + "/"
                            + "gid_"
                            + str(gid)
                            + "_"
                            + str(i)
                            + ".log"
                        )
                        os.system("mkdir -p {}".format(os.path.dirname(log)))
                        cmds.append(cmd)
                        logs.append(log)
                        # tasks.append(cmd_time_guard_async(cmd, self.time_out_threshold, log))
                    # assert(len(tasks) > 0)
                    # asyncio.run(cmd_run(tasks))
                    start = time.time()
                    asyncio.run(cmd_run(cmds, self.time_out_threshold, logs))
                    end = time.time()
                    for log in logs:
                        case_time = (end - start) / len(self.gids)
                        cur_result = getResult(log)
                        if i < len(cases.nodes):
                            record = cases.nodes[i].toSimpleRecord()
                            self.update_dic(cur_result, end - start, record, case_path)
                        result.addResult(cur_result)
                    pbar.update(1)
            else:
                gid_num = len(self.gids)
                cases_len = len(cases.cases_list)
                # for i in range(0,len(cases.cases_list),gid_num):
                #     cmds, logs = [], []
                #     for j,gid in enumerate(self.gids):
                #         if i + j < cases_len:
                #             cmd = self.demo(gid)+ '--case_path=' + cases.cases_list[i+j]
                #             print(f'{i+j}:{cmd}')
                #             log = log_dir + cases.nodes[i+j].op_name + '/' + 'gid_' + str(
                #             gid) + '_' + str(i+j) + '.log'
                #             os.system('mkdir -p {}'.format(os.path.dirname(log)))
                #             cmds.append(cmd)
                #             logs.append(log)
                #     start = time.time()
                #     asyncio.run(cmd_run(cmds, self.time_out_threshold, logs))
                #     end = time.time()
                #     for j,log in enumerate(logs):
                #         case_time = (end - start) / gid_num
                #         cur_result = getResult(log)
                #         record = cases.nodes[i+j].toSimpleRecord()
                #         self.update_dic(cur_result, end - start, record,cases.cases_list[i+j])
                #         result.addResult(cur_result)
                #     pbar.update(len(logs))
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                queue = asyncio.Queue()
                done_event = asyncio.Event()
                semaphore = asyncio.Semaphore(1)
                logs = []
                print(len(logs))
                tasks = []
                loop.run_until_complete(
                    self.create_worker(
                        queue, tasks, done_event, cases, logs, log_dir, pbar
                    )
                )
                print(len(logs))
                assert len(logs) == len(cases.cases_list)
                for i, log in enumerate(logs):
                    cur_result = getResult(log)
                    if i < len(cases.nodes):
                        record = cases.nodes[i].toSimpleRecord()
                        self.update_dic(cur_result, -1, record, cases.cases_list[i])
                    result.addResult(cur_result)
        end_1 = time.time()
        result.test_time = end_1 - start_1
        if self.merge:
            save_name = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            merge_log(log_dir, save_name)
        return result

    def unit_test(self, case_path):
        self.setEnv()
        return self.test(case_path, warm_repeat=3, perf_repeat=5)

    def perf_test(self, case_path):
        self.setEnv()
        return self.test(case_path)

    def press_test(self, case_path):
        self.setEnv()
        return self.test(
            case_path, warm_repeat=10, perf_repeat=int(1e6), time_out_threshold=60 * 60
        )

    def integration_test(self, case_path):
        self.envs["DNN_CHECK_ALL_MEM"] = "1"
        self.setEnv()
        return self.test(case_path, warm_repeat=10, perf_repeat=100, test_stable=True)

    def __repr__(self):
        return self.cases

    async def create_worker(self, queue, tasks, done_event, cases, logs, log_dir, pbar):
        await queue_put(queue, cases, logs, log_dir)
        for i in range(len(self.gids)):
            cmd = self.demo(self.gids[i])
            worker_task = asyncio.create_task(
                worker(cmd, self.time_out_threshold, queue, done_event, pbar)
            )
            tasks.append(worker_task)
            # print(i)
        await worker_run(queue, tasks, done_event)
