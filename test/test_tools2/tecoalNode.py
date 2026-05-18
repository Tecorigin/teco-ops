import sys
import os

sys.path.append(os.path.split(os.path.realpath(__file__))[0] + "/../test_proto")
import optest_pb2 as optest
import tensor_pb2 as optest_tensor
from google.protobuf import json_format

import numpy as np
import json
import pdb
import time
import logging


def getDataTypeStr(dtype):
    if not isinstance(dtype, int):
        return dtype
    if dtype == 0:
        return "DTYPE_FLOAT"
    if dtype == 1:
        return "DTYPE_HALF"
    if dtype == 2:
        return "DTYPE_INT8"
    if dtype == 3:
        return "DTYPE_INT16"
    if dtype == 4:
        return "DTYPE_INT32"
    if dtype == 5:
        return "DTYPE_INT64"
    if dtype == 6:
        return "DTYPE_UINT8"
    if dtype == 7:
        return "DTYPE_BOOL"
    if dtype == 8:
        return "DTYPE_UINT16"
    if dtype == 9:
        return "DTYPE_UINT32"
    if dtype == 10:
        return "DTYPE_UINT64"
    if dtype == 11:
        return "DTYPE_DOUBLE"
    if dtype == 12:
        return "DTYPE_COMPLEX_HALF"
    if dtype == 13:
        return "DTYPE_COMPLEX_FLOAT"
    if dtype == 14:
        return "DTYPE_INVALID"
    logging.warning("unknow dtype")
    return "unknow dtype"


def getLayoutStr(layout):
    if not isinstance(layout, int):
        return layout
    if layout == 0:
        return "LAYOUT_NCHW"
    if layout == 1:
        return "LAYOUT_NHWC"
    if layout == 2:
        return "LAYOUT_HWCN"
    if layout == 3:
        return "LAYOUT_NDHWC"
    if layout == 4:
        return "LAYOUT_ARRAY"
    if layout == 5:
        return "LAYOUT_NCDHW"
    if layout == 6:
        return "LAYOUT_TNC"
    if layout == 7:
        return "LAYOUT_NTC"
    if layout == 8:
        return "LAYOUT_NC"
    if layout == 9:
        return "LAYOUT_NLC"
    if layout == 10:
        return "LAYOUT_NWHC"
    if layout == 11:
        return "LAYOUT_CHWN"
    logging.warning("unknow layout")
    return "unknow layout"


def getRandomDistributionStr(distribute):
    if not isinstance(distribute, int):
        return distribute
    if distribute == 1:
        return "UNIFORM"
    if distribute == 2:
        return "GAUSSIAN"
    if distribute == 3:
        return "SAMPLE"
    if distribute == 4:
        return "BINOMIAL"
    if distribute == 5:
        return "UNIQUE"
    logging.warning("unknow distribute")
    return "unknow distribute"


def randData(tensor, distribute=1, lower_bound=-100, upper_bound=100):
    if tensor.HasField("prev_path"):
        return

    # !!!! proto.enum in python is int
    if getRandomDistributionStr(distribute) == "UNIFORM":
        if getDataTypeStr(tensor.dtype) == "DTYPE_BOOL":
            tensor.random_data.lower_bound = 2
            tensor.random_data.upper_bound = 0
            tensor.random_data.seed = int(np.random.rand() * 100000000)
            return

        tensor.random_data.lower_bound = lower_bound
        tensor.random_data.upper_bound = upper_bound
        tensor.random_data.seed = int(np.random.rand() * 100000000)
    else:
        logging.warning("randData not support now")


def saveNodeTo(node, file_path):
    if node is None:
        logging.error("node is none")
        return

    if file_path is None:
        logging.error("dst_path is None")

    dst_path = file_path
    if os.path.isdir(file_path):
        time_str = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        dst_path = file_path + "/{}.prototxt".format(time_str)

        i = 1
        while True:
            if not os.path.exists(dst_path):
                break
            dst_path = file_path + "/{}({}).prototxt".format(time_str, i)
            i += 1

    if not os.path.exists(os.path.dirname(os.path.realpath(dst_path))):
        os.system("mkdir -p {}".format(os.path.dirname(os.path.realpath(dst_path))))

    if os.path.exists(file_path):
        logging.warning("{} is exist, will rewrite".format(file_path))

    if dst_path.endswith(".prototxt"):
        with open(dst_path, "w") as f:
            f.write(node.__str__())
    elif dst_path.endswith(".pb"):
        with open(dst_path, "wb") as f:
            f.write(node.SerializeToString())
    else:
        logging.error("save dst_path fault")


def readNodeFrom(case_path):
    if case_path is None:
        logging.error("case_path is None")

    node = optest.Node()
    if case_path.endswith(".pb"):
        with open(case_path, "rb") as f:
            node.ParseFromString(f.read())
    elif case_path.endswith(".prototxt"):
        tmp_pb_file = "./temp_{}.pb".format(str(abs(hash(case_path)))[0:6])
        proto2pb = os.path.split(os.path.realpath(__file__))[0] + "/prototxt2pb"
        os.system(proto2pb + " " + case_path + " " + tmp_pb_file)
        with open(tmp_pb_file, "rb") as f:
            node.ParseFromString(f.read())
        os.system("rm " + tmp_pb_file)
    else:
        logging.error("case_path fault, support pb & prototxt")

    return node


def getTypeSize(dtype):
    if isinstance(dtype, int):
        dtype_str = getDataTypeStr(dtype)
    else:
        dtype_str = dtype

    if dtype_str in ["DTYPE_INT8", "DTYPE_BOOL", "DTYPE_UINT8"]:
        return 1

    if dtype_str in ["DTYPE_HALF", "DTYPE_INT16", "DTYPE_UINT16"]:
        return 2

    if dtype_str in ["DTYPE_FLOAT", "DTYPE_INT32", "DTYPE_UINT32"]:
        return 4

    if dtype_str in ["DTYPE_DOUBLE", "DTYPE_INT64", "DTYPE_UINT64"]:
        return 8

    logging.warning("unknow type: {}".format(dtype))
    return 8


def getTensorSize(tensor):
    size = 1
    for dim in tensor.shape.dims:
        size *= dim
        if size > 2 * 1024 * 1024 * 1024:
            logging.warning("tensor size > 2G")
    return size


def getTensorByteSize(tensor):
    if len(tensor.shape.dim_stride) == 0:
        return getTensorSize(tensor) * getTypeSize(tensor.dtype)

    if len(tensor.shape.dims) != len(tensor.shape.dim_stride):
        logging.error("dims != strides")

    dims = tensor.shape.dims
    strides = tensor.shape.dim_stride

    size = 1
    for i in range(len(dims)):
        size += (dims[i] - 1) * strides[i]

    return size * getTypeSize(tensor.dtype)


def setTensorShape(tensor, dims, dim_stride=None):
    shape = optest_tensor.Shape()
    shape.dims.extend(dims)
    if dim_stride is not None:
        shape.dim_stride.extend(dim_stride)

    tensor.shape.CopyFrom(shape)


def setTensor(tensor, num):
    if not tensor:
        logging.warning(f"tensor is None")
    if len(tensor) < num:
        for _ in range(num - len(tensor)):
            tensor.append(tensor[0])
    else:
        for _ in range(len(tensor) - num):
            tensor.pop()


class TecoalNode:
    op_name = "unknow"
    node = None

    # need
    def init_node(self):
        if self.op_name == "unknow":
            self.node = optest.Node()
            logging.warning("op_name: unknow")
        else:
            self.node = readNodeFrom(
                os.path.split(os.path.realpath(__file__))[0]
                + "/nodes/template/{}.prototxt".format(self.op_name)
            )

    def check(self):
        return True

    def readFrom(self, file_path):
        self.node = readNodeFrom(file_path)

    def saveTo(self, file_path):
        saveNodeTo(self.node, file_path)

    def __init__(self, case_path=None):
        if case_path is None:
            self.init_node()
            self.reRand()
            return

        self.readFrom(case_path)
        # self.check()

    def __str__(self):
        return self.node.__str__()

    def reRand(self, distribute=1, lower_bound=-100, upper_bound=100):
        if self.node is None:
            logging.error("node is None")
            return

        for i in range(len(self.node.input)):
            randData(self.node.input[i], distribute, lower_bound, upper_bound)

    def runNeedMemory(self):
        size = 0
        for tensor in self.node.input:
            size += getTensorByteSize(tensor)

        for tensor in self.node.output:
            size += getTensorByteSize(tensor)

        return size

    def saveNeedMemory(self):
        size = 0
        for tensor in self.node.input:
            size += getTensorByteSize(tensor)
            if tensor.HasField("reused") and tensor.reused:
                size += getTensorByteSize(tensor)  # save prev value

        for tensor in self.node.output:
            size += getTensorByteSize(tensor)

        return size

    def getOpName(self):
        return self.op_name

    def getShapeInfo(self):
        shape = []
        for tensor in self.node.input:
            shape.append([tensor.shape.dims, tensor.shape.dim_stride])

        for tensor in self.node.output:
            shape.append([tensor.shape.dims, tensor.shape.dim_stride])

        return shape

    def getLayoutInfo(self):
        layout = []
        for tensor in self.node.input:
            layout.append(getLayoutStr(tensor.layout))

        for tensor in self.node.output:
            layout.append(getLayoutStr(tensor.layout))

        return layout

    def getDtypeInfo(self):
        dtype = []
        for tensor in self.node.input:
            dtype.append(getDataTypeStr(tensor.dtype))

        for tensor in self.node.output:
            dtype.append(getDataTypeStr(tensor.dtype))

        return dtype

    def getParamInfo(self):
        if self.node.HasField("dnn_param"):
            return "dnn_param {{\n{}}}".format(self.node.dnn_param)

        if self.node.HasField("blas_param"):
            return "blas_param {{\n{}}}".format(self.node.blas_param)

        if self.node.HasField("custom_param"):
            return "custom_param {{\n{}}}".format(self.node.custom_param)

    def toDict(self):
        return json_format.MessageToDict(self.node)

    def toSimpleRecord(self):
        record = {}
        record["op_name"] = self.getOpName()
        record["shape"] = self.getShapeInfo()
        record["dtype"] = self.getDtypeInfo()
        record["layout"] = self.getLayoutInfo()
        record["param"] = self.getParamInfo()
        return record

    def toRecord(self):
        return self.toDict()
