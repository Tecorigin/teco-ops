import os

def upper_camel(op_name):
    res = ""
    slices = op_name.split('_')
    for item in slices:
        res += item.capitalize()
    return res

def get_op_params(al_type):
    param_strs = []
    import_strs = []
    files = os.listdir(al_type)
    for file in files:
        if file.endswith(".proto"):
            op_name = file.replace(".proto", "")
            op_name = op_name.replace("custom_", "")
            if op_name in ["common", "convolution"]:
                continue
            file = os.path.join(al_type, file)
            with open(file, 'r') as f:
                fstr = f.read()
            
            op_param = upper_camel(op_name) + "Param"
            # print(f"  {op_name=}  {op_param=}")
            if op_param not in fstr:
                print(f"  [Error] not found {op_param} in {file}")
            param_str = "    optional {} {}_param           = {{}};\n".format(op_param, op_name)
            import_str = 'import "{}";\n'.format(file)
            param_strs.append(param_str)
            import_strs.append(import_str)
    return param_strs, import_strs

def update_al_proto(al_type, param_strs, import_strs):
    filename = "{}.proto.template".format(al_type)
    with open(filename, 'r') as f:
        lines = f.readlines()

    # find import index, then add
    index = -1
    for i in range(0, len(lines)):
        if "import" in lines[i] and "proto" in lines[i]:
            index = i + 1
            break
    if index > 0:
        for import_str in import_strs:
            # print(f"  {import_str=}")
            lines.insert(index, import_str)
            index += 1

    # find param index, then add
    index = -1
    for i in range(len(lines)-1 , -1 , -1):
        if lines[i].strip() == "}" and i >= 1:
            index = int(lines[i-1].split("=")[-1].strip(";\n")) + 1
            break
    if index > 0:
        for param_str in param_strs:
            param_str = param_str.format(index)
            # print(f"  {param_str=}")
            lines.insert(i, param_str)
            i += 1
            index += 1

    filename = "{}.proto".format(al_type)
    with open(filename, 'w') as f:
        for line in lines:
            f.write(line)

if __name__ == "__main__":
    al_types = ["tecoal"]
    for al_type in al_types:
        # print(f"{al_type=}")
        param_strs, import_strs = get_op_params(al_type)
        update_al_proto(al_type, param_strs, import_strs)
