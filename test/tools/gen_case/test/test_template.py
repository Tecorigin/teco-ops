import pandas as pd
from gen_tensor import *

# TODO
case='''op_name: "nearest_interp_backward"
input :{}
output :{}
dnn_param: {{
  interp_param: {{
    align_corners: {}
    scale_factor: {}
    scale_factor: {}
}}
}}
'''

def save_prototxt(file_name, a):
  file = open(file_name, "w+")
  file.write(a)
  file.close()

type_size = {'DTYPE_BOOL':1, 'DTYPE_UINT8':1, 'DTYPE_INT8':1, 'DTYPE_HALF':2, 'DTYPE_INT16':2, 'DTYPE_FLOAT': 4.0, 'DTYPE_INT32':4, 'DTYPE_INT64': 8.0}

feat = ( {'dtype':'DTYPE_HALF', 'align_corners':'ALIGN_CORNERS'},
{'dtype':'DTYPE_HALF', 'align_corners':'ALIGN_CORNERS'})

max_size = 2.0*1024 * 1024 *1024
# compute tensor max size

# TODO
def gen_nearest_interp_backward_size(dtype, in_shape, out_shape, scale_factor):
  real_out_shape = out_shape.copy()
  if(scale_factor[0]>0): real_out_shape[1] = scale_factor[0] * in_shape[1]
  if(scale_factor[1]>0): real_out_shape[2] = scale_factor[1] * in_shape[2]

  in_num = 1
  out_num = 1
  for i in range(len(in_shape)):
    in_num = in_num * in_shape[i]
    out_num = out_num * real_out_shape[i]
    if( max([in_num ,out_num]) > max_size):
      return max([in_num, out_num])
        
  in_size = in_num*type_size[dtype]
  out_size = out_num*type_size[dtype]
  return max([in_size, out_size])

# TODO
def gen_nearest_interp_backward(dtype, in_shape, out_shape, aligned, scale_factor):
  layout = 'LAYOUT_NHWC'

  input0 = TensorDesc()
  input0.name = "dy"
  input0.shape = in_shape
  input0.layout = layout
  input0.dtype = dtype

  output = TensorDesc()
  output.name = "dx"
  output.shape = out_shape
  output.layout = layout
  output.dtype = dtype

  return case.format(input0.str(), output.str(), aligned, scale_factor[0], scale_factor[1])

bound_array = [2,4,8,16,32,64,128,256,512,1024,2048,1024*1024]
case_num_dim4 = [4, 4, 4, 20, 40, 40, 40, 20, 20, 20, 20, 20]

def test_2_4():
  for i in range(len(feat)):
    # TODO
    dtype = feat[i]['dtype']
    aligned = feat[i]['align_corners']
    for j in range(len(bound_array)):
      bound = bound_array[j]
      case_num = int(max(1, int(case_num_dim4[j]/reduce_case)))
      for z in range(case_num):
        # TODO
        scale_factor = np.random.uniform(-2, 2, [2])
        tmp = np.random.randint(1, bound+1, size=[6])

        max_index = np.argmax(tmp)
        max_dim = np.max(tmp)
        if(max_dim<bound/2): tmp[max_index] = np.random.randint(bound/2,bound)

        in_shape = tmp[0:4]
        out_shape = [tmp[0], tmp[1], tmp[4], tmp[5]]

        while(True):
          # TODO
          size = gen_nearest_interp_backward_size(dtype, in_shape, out_shape, scale_factor)
          if(size<max_size): break
          idx = np.random.randint(6)
          if(idx != max_index):
            tmp[idx] = np.random.randint(1,16)
          in_shape = tmp[0:4]
          out_shape = [tmp[0], tmp[1], tmp[4], tmp[5]]

        a = gen_nearest_interp_backward(dtype, in_shape, out_shape, aligned, scale_factor)
        file_name = "./test2_4_feat" + str(i) + "_bound" + str(j) + "_case_"+ str(z) + ".prototxt"
        save_prototxt(file_name, a)

# 随机设下种子
# TODO
np.random.seed(64562)
reduce_case = 1
test_2_4()
