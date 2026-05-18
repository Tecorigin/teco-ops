import hashlib
import json
from prototxt_parser.prototxt_parser_main import parse

USE_DB = False
if USE_DB:
    from model_db import Case, Model, ModelCase
else:
    from model import Case, Model, ModelCase

class ProtoParser:

    def __init__(self, filename):
        self.filename = filename

    def read(self):
        filtered_lines = []
        with open(self.filename, 'r') as f:
            for line in f:
                index = line.find("#")
                if index != -1:
                    line = line[0:index] + "\n"
                filtered_lines.append(line)
        self.content = "".join(filtered_lines)

    def create_record(self, params):
        op_name = params["op_name"]
        case_path = self.filename
        prototxt = self.content

        input_tensors = []
        output_tensors = []
        if "input" in params:
            input_tensors = params["input"]
        if "output" in params:
            output_tensors = params["output"]

        if type(input_tensors) != list:
            input_tensors = [input_tensors]
        if type(output_tensors) != list:
            output_tensors = [output_tensors]
        tensors = input_tensors + output_tensors

        all_shape = []
        all_layout = []
        all_dtype = []
        for tensor in tensors:
            shape = tensor["shape"]["dims"]
            if type(shape) != list:
                shape = [shape]
            shape = [str(int(i)) for i in shape]
            shape = "[" + ",".join(shape) + "]"

            stride = "[]"
            if "dim_stride" in tensor["shape"]:
                stride = tensor["shape"]["dim_stride"]
                if type(stride) != list:
                    stride = [stride]
                stride = [str(int(i)) for i in stride]
                stride = "[" + ",".join(stride) + "]"
            shape = "[" + shape + "," + stride + "]"

            layout = tensor["layout"].replace("LAYOUT_", "")
            dtype = tensor["dtype"].replace("DTYPE_", "")
            all_shape.append(shape)
            all_layout.append(layout)
            all_dtype.append(dtype)
            
        if len(set(all_layout)) == 1:
            all_layout = [all_layout[0]]
        if len(set(all_dtype)) == 1:
            all_dtype = [all_dtype[0]]
        all_shape = ",".join(all_shape)
        all_layout = ",".join(all_layout)
        all_dtype = ",".join(all_dtype)

        m = hashlib.md5()
        m.update(prototxt.encode('utf-8'))
        hash = m.hexdigest()

        return op_name, all_dtype, all_shape, all_layout, prototxt, case_path, hash

    def run(self):
        self.read()
        params = parse(self.content)
        return self.create_record(params)


class CaseParser():

    def __init__(self, filename, version, level, bug_id=-1):
        self.filename = filename
        self.version = version
        self.level = level
        self.bug_id = bug_id

    def run(self):
        with open(self.filename, 'r') as f:
            cases_path = f.readlines()
        cases_path = [case.strip() for case in cases_path]

        cases = []
        for case_path in cases_path:
            pp = ProtoParser(case_path)
            op_name, dtype, shape, layout, prototxt, case_path, hash = pp.run()
            case = Case()
            case.op_name = op_name
            case.dtype = dtype
            case.shape = shape
            case.layout = layout
            case.prototxt = prototxt
            case.case_path = case_path
            case.hash = hash

            case.version = self.version
            case.level = self.level
            case.bug_id = self.bug_id
            cases.append(case)
        return cases


class ModelParser():

    def __init__(self, filename, name, image, version):
        self.filename = filename
        self.name = name
        self.image_size = image
        self.version = version

    def run(self):
        model = Model()
        model.name = self.name
        model.image_size = self.image_size
        model.version = self.version

        with open(self.filename, 'r') as f:
            cases_data = json.load(f)
        cases = []
        for case_path, case_count in cases_data.items():
            pp = ProtoParser(case_path)
            op_name, dtype, shape, layout, prototxt, case_path, hash = pp.run()
            case = Case()
            case.op_name = op_name
            case.dtype = dtype
            case.shape = shape
            case.layout = layout
            case.prototxt = prototxt
            case.case_path = case_path
            case.hash = hash

            case.version = self.version
            case.level = 1  # model case
            case.count = case_count
            cases.append(case)

        return model, cases


if __name__ == "__main__":
    # ProtoParser test
    filename = "/data/Hpe_share/project/software_stack/0.9.0/blas/gemm_stride_batched/test2_1_modelGPT_TA_80.prototxt"
    pp = ProtoParser(filename)
    print(pp.run())
    print()

    # CaseParser test
    version = "1.15.0"
    cases_list = "data/cases.txt"
    outfile = "data/cases.json"
    cp = CaseParser(cases_list, version, level=2)
    cases = cp.run()
    cases = [case.to_dict() for case in cases]

    with open(outfile, 'w') as f:
        f.write(json.dumps(cases, indent=4))

    # CaseParser test
    version = "1.14.0"
    cases_list = "data/cases_2.txt"
    outfile = "data/cases_2.json"
    cp = CaseParser(cases_list, version, level=3)
    cases = cp.run()
    cases = [case.to_dict() for case in cases]

    with open(outfile, 'w') as f:
        f.write(json.dumps(cases, indent=4))

    # ModelParser test
    name = "r50_pt_bs16"
    image = "512*512"
    version = "1.14.0"
    cases_json = "/data/Hpe_share/project/models/Q3_Q4_all_model_cases_dump/r50_pt_bs16/unique_cases_list.json"
    outfile = "data/model_bs16.json"
    mp = ModelParser(cases_json, name, image, version)
    model, cases = mp.run()
    model = model.to_dict()
    cases = [case.to_dict() for case in cases]
    res = {
        "model_data": model,
        "case_data": cases,
    }
    with open(outfile, 'w') as f:
        f.write(json.dumps(res, indent=4))

    
    name = "r50_pt_bs32"
    image = "640*640"
    version = "1.15.0"
    cases_json = "/data/Hpe_share/project/models/Q3_Q4_all_model_cases_dump/r50_pt_bs32/unique_cases_list.json"
    outfile = "data/model_bs32.json"
    mp = ModelParser(cases_json, name, image, version)
    model, cases = mp.run()
    model = model.to_dict()
    cases = [case.to_dict() for case in cases]
    res = {
        "model_data": model,
        "case_data": cases,
    }
    with open(outfile, 'w') as f:
        f.write(json.dumps(res, indent=4))
