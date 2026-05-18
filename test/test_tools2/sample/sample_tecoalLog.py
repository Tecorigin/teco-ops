from tecoalResult import TecoalResult
import tecoalLog
import pandas as pd
import copy

# result = TecoalResult('./tmp.log')
# result = tecoalLog.getResult('./tmp.log')
# data = pd.DataFrame(result)
# data.to_excel('./tmp.xlsx')

# result = TecoalResult('/data/dnn/heqq/heqq_test/0807/log/pooling_forward/5104_2.log')
# result = TecoalResult('/data/dnn/heqq/tecoal_1.10.0/test/frame_work/tecotest/test_tools2/log/2023_08_08_21_12_24_40580')
# result = TecoalResult('/data/dnn/heqq/heqq_test/opname_error.log')
# result = TecoalResult('/data/dnn/heqq/qqh_note/code_py/utils/log')
# result = TecoalResult('/data/dnn/heqq/heqq_test/log/activation_backward/5_2.log')
# result = TecoalResult(
#     '/data/dnn/heqq/tecoal_1.10.0/test/frame_work/tecotest/log/2023_08_09_20_59_38_42419/gemm_batched/gid_0_9092.log'
# )
# result = TecoalResult('/data/dnn/heqq/tecoal_1.10.0/test/frame_work/tecotest/test_tools2/log/2023_08_08_17_53_52_6415')
"""data = result.format(['log','case_path', 'op_name', 'user_time', 'interface_time(ms)',
        'hardware_time(ms)', 'launch_time(ms)', 'theory_ios(Bytes)',
        'io_bandwidth(GB/s)', 'kernel_details', 'result'])
"""
# result = TecoalResult('/data/dnn/heqq/tecoal_1.10.0/test/frame_work/tecotest/log/2023_08_09_20_59_38_42419/gemm/gid_0_4083.log')
# res = copy.deepcopy(result)
result = TecoalResult("/data/dnn/heqq/tecoal_1.10.0/test/frame_work/tecotest/log/2023_08_16_00_08_03_31058")
keys = ["dtype", "shape", "param", "layout"]
for key in keys:
    result.accu_format.remove(key)
accu_result = result.getLogResult("accu")
data = pd.DataFrame(accu_result)
data.to_excel("./cur.xlsx")
