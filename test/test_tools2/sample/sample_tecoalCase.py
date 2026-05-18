import sys
import os

from getTecoalNode import getTecoalNode
from tecoalCase import TecoalCase

# test tecoalCase
cases = TecoalCase("./cases_list.txt")
print(cases)

print("save need memory: ", cases.saveNeedMemory())

df = cases.toDataFrame()
# 统计有多少算子(如果算子名称未知，则都属于unknow)
print("total op: ", df["op_name"].nunique())

# 统计每个算子有多少测例
print("\n#####")
print(df["op_name"].value_counts())
print("#####\n")

# save in excel
os.system("mkdir -p log")
df.to_excel("./log/tmp.xlsx")
