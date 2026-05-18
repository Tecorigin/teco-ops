from gen_random import *
import numpy as np
import time

def str2array(string):
    return np.array(string.strip('[]').split(','), dtype=int)

tensor_desc_str = '''
  id: "{}"
  shape: {}
  layout: {}
  dtype: {}
  ttype: {}
  reused: {}
'''

def shape_2_str(shape, stride):
    tmp = "{"
    for i in range(len(shape)):
        tmp += "\n    dims:{}".format(shape[i])
    if stride is not None:
        for i in range(len(stride)):
            tmp += "\n    dim_stride:{}".format(stride[i])
    tmp += "\n  }"
    return tmp

def value_2_str(value, dtype):
    v_str = 'value_f'
    if dtype == 'DTYPE_FLOAT':
        v_str = 'value_f'
    if dtype == 'DTYPE_INT32':
        v_str = 'value_i'
    if dtype == 'DTYPE_INT64':
        v_str = 'value_l'
    if dtype == 'DTYPE_HALF':
        v_str = 'value_h'

    tmp = "{"
    for i in range(len(value)):
        tmp += "\n    {}:{}".format(v_str, value[i])
    tmp += "\n  }"
    tmp += "\n"
    return tmp


class TensorDesc():
    seed = np.random.seed(int(time.time()))

    name = 'null'
    shape = [0]
    stride = None
    layout = 'LAYOUT_ARRAY'
    dtype = 'DTYPE_FLOAT'
    ttype = 'TENSOR'
    reused = 'false'

    prev_value = None
    value = None
    prev_path = None
    path = None

    random_data = RandomData()

    thresholds = None
    threshold_use = None

    def __init__(self):
        while True:
            N = np.random.randint(1, 128)
            C = np.random.randint(1, 256)
            H = np.random.randint(1, 256)
            W = np.random.randint(1, 256)
            if N * C * H * W < 1024 * 1024 * 256:
                break
        self.shape = [N, C, H, W]
        self.random_data = RandomData()

    def str(self):
        tmp = "{"
        tmp += tensor_desc_str.format(self.name, shape_2_str(self.shape, self.stride),
                                      self.layout, self.dtype, self.ttype,
                                      self.reused)
        if self.prev_value is None:
            tmp += "\n  random_data:{}".format(self.random_data.str())
        else:
            tmp += "\n  prev_value:{}".format(value_2_str(self.prev_value, self.dtype))
        tmp += "}\n"
        return tmp
