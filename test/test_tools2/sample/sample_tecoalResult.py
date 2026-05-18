import sys
import os
import pandas as pd

sys.path.append(os.path.split(os.path.realpath(__file__))[0] + "/../")

from tecoalResult import TecoalResult

path = "/data/dnn/maliang/workspace/1.15.0/tecoal/test/frame_work/tecotest/build/log/2024_01_03_21_37_10_12251"
result = TecoalResult(path)

# accu_data = pd.DataFrame(result.getLogResult("accu"))
# accu_data.to_excel("./accu.xlsx")

perf_data = pd.DataFrame(result.getLogResult("perf"))
perf_data.to_excel("./perf.xlsx")
