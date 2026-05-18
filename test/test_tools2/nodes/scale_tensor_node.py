import sys
import os
import numpy as np
import json
from google.protobuf import json_format

sys.path.append(
    os.path.split(os.path.realpath(__file__))[0] + '/../../test_proto')
import optest_pb2 as optest

sys.path.append(os.path.split(os.path.realpath(__file__))[0])
sys.path.append(os.path.split(os.path.realpath(__file__))[0] + '/../')
from tecoalNode import *

class TecoalScaleTensorNode(TecoalNode):
    op_name = "scale_tensor"

    def check(self):
        return True

    def reRand(self):
        if self.node is None:
            print("node is None")
            return

        for i in range(len(self.node.input)):
            randData(self.node.input[i])
 
    def getParamInfo(self):
        if self.node.HasField('tecoal_param'):
            return 'tecoal_param {{\n{}}}'.format(self.node.tecoal_param)
