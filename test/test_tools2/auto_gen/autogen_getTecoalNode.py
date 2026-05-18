import os

code = """
import os
import sys

from tecoalNode import TecoalNode
{}
api_list = {}

def getTecoalNode(case_path):
    if case_path is None:
        return TecoalNode()

    for op_name in case_path.split('/'):
        if(op_name in api_list):
            break
    
    if op_name is None:
        return TecoalNode()
    {}

    return TecoalNode()
"""

code_part1 = """from nodes.{}_node import Tecoal{}Node
"""

code_part2 = """
    if op_name == '{}':
        return Tecoal{}Node(case_path)
"""

filepath = os.path.split(os.path.realpath(__file__))[0]


def getNodeList():
    result = os.popen("ls {}/../nodes/*_node\.py".format(filepath))
    tmp = result.read().strip().split("\n")
    result.close()

    nodes = []
    for node in tmp:
        nodes.append(node.split("/")[-1][0:-8])

    return nodes


result = os.popen("ls {}/../nodes/".format(filepath))

tmp_code1 = ""
tmp_code2 = ""
for op_name in getNodeList():
    class_name = ""
    for tmp in op_name.split("_"):
        class_name += tmp.capitalize()

    tmp_code1 += code_part1.format(op_name, class_name)
    tmp_code2 += code_part2.format(op_name, class_name)

# code.format(str(getNodeList()), tmp_code)

out_file = "{}/getTecoalNode.py.auto".format(filepath)
with open(out_file, "w") as f:
    f.write(code.format(tmp_code1, str(getNodeList()), tmp_code2))

os.system("yapf -i {}".format(out_file))
