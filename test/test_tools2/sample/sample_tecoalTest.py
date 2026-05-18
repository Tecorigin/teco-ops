import sys
import os
import pandas as pd

sys.path.append(os.path.split(os.path.realpath(__file__))[0] + "/../")

from getTecoalNode import getTecoalNode
from tecoalCase import TecoalCase
from tecoalTest import TecoalTest

# test =  TecoalTest()
# # test.unit_test('./cases_list.txt')
# test.unit_test('/data/dnn/tangyu/workspace/version/custom/tecoal/test/frame_work/tecotest/tangyu_tools/sample/cases_list.txt')

test = TecoalTest()
path = "/data/dnn/heqq/heqq_test/0807/test_0808.txt"
# path = '/data/hpe_data/testcase/models/benchmark/dump_info_custom/all_pass_models_list.txt'
# 指定核心
test.gids = [1, 2]
# 指定warm_repeat, perf_repeat...
test.warm_repeat = 3
test.perf_repeat = 10
# 指定并发执行不同测例
test.async_gids = True
# 指定环境变量
# test.envs["DNN_CHECK_ALL_MEM"] = 1
result = test.unit_test(path)
# accu_format = [
#     'log', 'case_path', 'op_name', 'shape','dtype','layout','param','output','DIFF3_MAX_GPU', 'DIFF3_MAX_TECO',
#     'DIFF3_MAX_THRESHOLD', 'DIFF3_MEAN_GPU', 'DIFF3_MEAN_TECO',
#     'DIFF3_MEAN_THRESHOLD', 'result','case_time'
# ]
# perf_format = [
#     'log', 'case_path', 'op_name', 'interface_time(ms)', 'hardware_time(ms)',
#     'launch_time(ms)', 'io_bandwidth(GB/s)', 'theory_ios(Bytes)',
#     'compute_force(op/s)', 'theory_ops(Ops)', 'result'
# ]
accu_data = pd.DataFrame(result.getLogResult("accu"))
accu_data.to_excel("./accu.xlsx")


perf_data = pd.DataFrame(result.getLogResult("perf"))
perf_data.to_excel("./perf.xlsx")
# test.press_test(path)
# test.perf_test(path)
# test.integration_test(path)
