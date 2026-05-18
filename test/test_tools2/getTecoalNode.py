import os
import sys

from tecoalNode import TecoalNode
from nodes.arg_max_node import TecoalArgMaxNode
from nodes.masked_fill_node import TecoalMaskedFillNode
from nodes.scale_tensor_node import TecoalScaleTensorNode
from nodes.scatter_out_node import TecoalScatterOutNode
from nodes.unary_ops_node import TecoalUnaryOpsNode
from nodes.unique_node import TecoalUniqueNode

api_list = [
    'arg_max', 'masked_fill', 'scale_tensor', 'scatter_out', 'unary_ops',
    'unique'
]


def getTecoalNode(case_path):
    if case_path is None:
        return TecoalNode()

    for op_name in case_path.split('/'):
        if (op_name in api_list):
            break

    if op_name is None:
        return TecoalNode()

    if op_name == 'arg_max':
        return TecoalArgMaxNode(case_path)

    if op_name == 'masked_fill':
        return TecoalMaskedFillNode(case_path)

    if op_name == 'scale_tensor':
        return TecoalScaleTensorNode(case_path)

    if op_name == 'scatter_out':
        return TecoalScatterOutNode(case_path)

    if op_name == 'unary_ops':
        return TecoalUnaryOpsNode(case_path)

    if op_name == 'unique':
        return TecoalUniqueNode(case_path)

    return TecoalNode()
