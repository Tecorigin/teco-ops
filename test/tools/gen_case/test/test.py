import pandas as pd
from gen_tensor import *

case='''op_name: "nllloss2d_forward"
input :{}
input :{}
input :{}
output :{}
dnn_param: {{
  nlll_param: {{
  reduction: {}
  ignore_index: {}
}}
}}
'''

df = pd.read_excel('test.xls')

row_num = len(df)
col_num = len(df.columns)

print("row: {}, col: {}".format(row_num, col_num));

for i in range(row_num):
#for i in range( len(df)):
    series = df.loc[i]
    shape = series.iloc[0]
    format_ = series.iloc[1]
    dataType = series.iloc[1]
    ignore_index = series.iloc[2]
    reduction = 0

    print("shape: ", shape)
    print("format:", format_)
    print("dataType:", dataType)
    print("ignore_index:", ignore_index)

    input0 = TensorDesc()
    input0.name = "x"
    input0.shape = str2array(shape)
    input0.layout = format_
    input0.dtype = dataType

    input1 = TensorDesc()
    input1.name = "input1"
    input1.shape = str2array(shape)
    input1.layout = format_
    input1.dtype = dataType

    input2 = TensorDesc()
    input2.name = "input2"
    input2.shape = str2array(shape)
    input2.layout = format_
    input2.dtype = dataType

    output = TensorDesc()
    output.name = "output"
    output.shape = str2array(shape)
    output.layout = format_
    output.dtype = dataType

    a = case.format(input0.str(), input1.str(), input2.str(),
    output.str(), 1, reduction, ignore_index)
    file = open("./case_" + str(i) + ".prototxt", "w+")
    file.write(a)
    file.close()
    
