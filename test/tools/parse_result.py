import xml.etree.ElementTree as ET
import os
import json
import pandas as pd
import stat

from prototxt_parser.prototxt_parser_main import parse

def read_prototxt(file)->dict:
    filtered_lines = []
    with open(file, 'r') as f:
        for line in f:
            index = line.find("#")
            if index != -1:
                line = line[0:index] + "\n"
            filtered_lines.append(line)
    content = "".join(filtered_lines)
    return content, parse(content)

def parse_tensor(tensor):
    tensor_id = tensor["id"]

    shape = tensor["shape"]["dims"]
    if type(shape) != list:
        shape = [shape]
    shape = [int(i) for i in shape]

    stride = []
    if "stride" in tensor["shape"]:
        stride = tensor["shape"]["stride"]
        stride = [int(i) for i in stride]

    layout = tensor["layout"]
    dtype = tensor["dtype"]

    return tensor_id, shape, stride, dtype

def paser_prototxt(params):
    input_dtype = ""
    output_dtype = ""
    input_shapes = {}
    output_shapes = {}

    input_tensors = []
    output_tensors = []
    if "input" in params:
        input_tensors = params["input"]
        if type(input_tensors) != list:
            input_tensors = [input_tensors]
    if "output" in params:
        output_tensors = params["output"]
        if type(output_tensors) != list:
            output_tensors = [output_tensors]
    for tensor in input_tensors:
        tensor_id, shape, stride, dtype = parse_tensor(tensor)
        input_shapes[tensor_id] = [shape, stride]
        input_dtype += dtype + " "
    
    for tensor in output_tensors:
        tensor_id, shape, stride, dtype = parse_tensor(tensor)
        output_shapes[tensor_id] = [shape, stride]
        output_dtype += dtype + " "
    input_shapes.update(output_shapes)
    return input_shapes, input_dtype.strip() + "-" + output_dtype.strip()


def parse_xml(xml):
    tree = ET.parse(xml)
    root = tree.getroot()
    overview_name = root.attrib['name']
    overview_tests = int(root.attrib['tests'])
    overview_failed = int(root.attrib['failures'])
    overview_disabled = int(root.attrib['disabled'])
    all_data = {
        'name': overview_name,
        'testsuite_tests': 0,
        'testsuite_failures': 0,
        'testcase_tests': overview_tests,
        'testcase_failures': overview_failed,
		'testcase_disabled': overview_disabled,
        'testsuites': []
    }

    for testsuite in root:
        testsuite_name = testsuite.attrib['name'].replace("/TestSuite", "")
        idx = testsuite_name.index('_')
        al_type = testsuite_name[0:idx]
        testsuite_name = testsuite_name[idx+1:]
        testsuite_tests = int(testsuite.attrib['tests'])
        testsuite_failed = int(testsuite.attrib['failures'])
        testsuite_disabled = int(testsuite.attrib['disabled'])
        testsuite_data = {
            'name': testsuite_name,
            'tests': testsuite_tests,
            'failures': testsuite_failed,
            'disabled': testsuite_disabled,
            'testcases': []
        }
        all_data['testsuite_tests'] += 1
        if testsuite_failed != 0:
            all_data['testsuite_failures'] += 1


        for testcase in testsuite:
            testcase_name = testcase.attrib['name'].replace("/", "_")
            testcase_time = testcase.attrib['time']
            testcase_status = testcase.attrib['status'].upper()
            testcase_data = {
                'name': testcase_name,
                'time': testcase_time,
                'status': testcase_status,
                "property": {
                            "al_type" : al_type,
                            "op_name":"",
                            "case_name":"",
                            "dtype":"",
                            "shape":""
                            }
            }
            
            
            errors = {}
            testcase_data["property"]["status"] = "PASS"
            for properties in testcase:
                if properties.tag=="failure":
                    testcase_data["property"]["status"] = "FAILED"
                if properties.tag!="properties":
                    continue
                for child in properties:
                    key = child.attrib["name"]
                    value = child.attrib["value"]
                    if not (key.startswith("GPU_") or key.startswith("TECO_") or key.startswith("THRESHOLD_")):
                        testcase_data["property"][key] = value
                    else:
                        tmp, diff = key.split("-")
                        device_type, _, tensor_name = tmp.split("_")
                        diff = device_type + "_" + diff
                        if device_type in ["TECO", "GPU"]:
                            index, baseline_value, device_value, error_value = value.split(",")
                            if diff not in errors:
                                errors[diff] = {}
                            if diff.endswith("DIFF3_MAX"):
                                errors[diff][tensor_name] = {
                                    "max_error":float(error_value),
                                    "index":int(index),
                                    "baseline_value":float(baseline_value),
                                    "compare_value":float(device_value)
                                }
                            else:
                                errors[diff][tensor_name] = {
                                    "max_error":float(error_value)
                                }
                        elif device_type == "THRESHOLD":
                            golden_threshold, true_threshold = value.split(",")
                            if diff not in errors:
                                errors[diff] = {}
                            
                            errors[diff][tensor_name] = {
                                    "golden_threshold":float(golden_threshold),
                                    "true_threshold":float(true_threshold)
                            }
            
            testcase_data["property"].update(errors)
            # get tensor msg from prototxt
            content, params = read_prototxt(testcase_data["property"]["case_path"])
            shapes, dtypes = paser_prototxt(params)
            testcase_data["property"]["dtype"] = dtypes
            testcase_data["property"]["shape"] = shapes
            testcase_data["property"]["prototxt"] = content

            # testcase_data["property"]["dtype"] = testcase_data["property"]["dtype"].replace("DTYPE_", "").lower()

            testsuite_data["testcases"].append(testcase_data)

        all_data["testsuites"].append(testsuite_data)

    return all_data

# op_name:{case_name:{}}
def get_test_data(filename):
    if not os.path.exists(filename):
        print("Test result file {} not exists!!!".format(filename))
        return []
    all_data = parse_xml(filename)
    test_result = {}
    for testsuite in all_data["testsuites"]:
        for testcase in testsuite["testcases"]:
            testcase = testcase["property"]
            op_name = testcase["op_name"]
            case_name = testcase["case_name"]
            if op_name in test_result:
                test_result[op_name][case_name] = testcase
            else:
                test_result[op_name] = {case_name: testcase}

    return test_result


# op_name:{case_name:{}}
def get_cases_info(filepath):
    data = {}
    if not os.path.exists(filepath):
        return data
    cases_info_files = []
    res = os.listdir(filepath)
    for r in res:
        if r.endswith(".json"):
            cases_info_files.append(os.path.join(filepath, r))
    for cases_info_file in cases_info_files:
        with open(cases_info_file) as f:
            op_name = cases_info_file.split("/")[-1].split(".")[0]
            data[op_name] = json.load(f)
    
    return data


# search in excel db
# if not in, insert it
def parse_result(xml_input_file, xls_result_file, cases_info_path):
    if not os.path.exists(xml_input_file):
        print("{} not exists, failed!!!".format(xml_input_file))
        return 
    
    test_result = get_test_data(xml_input_file)
    all_cases_info = get_cases_info(cases_info_path)
    # print(all_cases_info)

    # 1. res不在cases_info中，直接插入
    # 2. res失败，不处理
    # 3. cases_info失败，直接更新
    # 4. res优于cases_info，更新
    # 5. res太差，置为失败
    for op_name, op_value in test_result.items():
        if op_name not in all_cases_info:
            all_cases_info[op_name] = op_value
        else:
            for case_name, case_value in op_value.items():
                if case_name not in all_cases_info[op_name]:
                    all_cases_info[op_name][case_name] = case_value
                else:
                    if case_value["status"] == "PASS":
                        if all_cases_info[op_name][case_name]["status"] == "PASS":
                            now_run_time = float(case_value["hardware_time(ms)"])
                            best_run_time = float(all_cases_info[op_name][case_name]["hardware_time(ms)"])
                            if now_run_time > 1.1 * best_run_time:
                                test_result[op_name][case_name]["status"] = "FAILED"
                                test_result[op_name][case_name]["failed_reason"] = "performance"
                            elif now_run_time < best_run_time:
                                all_cases_info[op_name][case_name] = case_value
                        else:
                            all_cases_info[op_name][case_name] = case_value
                            
        # cases info updated to json file
        cases_info_file = os.path.join(cases_info_path, "{}.json".format(op_name))
        with open(cases_info_file, 'w') as f:
            json.dump(all_cases_info[op_name], f)
        os.chmod(cases_info_file, stat.S_IRWXO | stat.S_IRWXG | stat.S_IRWXU)

        # cases info updated to excel file
        # list -> file
        sd_data = list(all_cases_info[op_name].values())
        df = pd.DataFrame(sd_data)
        cases_info_file_xls = cases_info_file.replace(".json", ".xlsx")
        df.to_excel(cases_info_file_xls, index=False)
        os.chmod(cases_info_file_xls, stat.S_IRWXO | stat.S_IRWXG | stat.S_IRWXU)
    

    # test result to file
    # list -> file
    test_value = []
    for op_value in test_result.values():
        for case_value in op_value.values():
            test_value.append(case_value)      
    df = pd.DataFrame(test_value)
    df.to_excel(xls_result_file, index=False)


import sys
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("input error")
    else:
        xml_input_file = sys.argv[1]
        xls_result_file = sys.argv[2]
        cases_info_path = sys.argv[3]
        parse_result(xml_input_file, xls_result_file, cases_info_path)
