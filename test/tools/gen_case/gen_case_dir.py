import os

testcase_dir = "../../testcase/"
case_types = ["bug", "1.5.2b0"]
al_types = ["dnn", "blas",'custom']
al_dir = "../../zoo/"

def get_ops(op_dir):
    ops = []
    dirs = os.listdir(op_dir)
    for dir_ in dirs:
        if os.path.isdir(os.path.join(op_dir, dir_)):
            ops.append(dir_)
    
    return ops


def generate_cases_dir():
    for case_type in case_types:
        case_type_dir = os.path.join(testcase_dir, case_type)     
        for al_type in al_types:
            al_type_dir = os.path.join(case_type_dir, al_type)
            al_path = os.path.join(al_dir, al_type)
            ops = get_ops(al_path)
            for op in ops:
                op_dir = os.path.join(al_type_dir, op, "test_case")
                if not os.path.exists(op_dir):
                    os.makedirs(op_dir)
                with open(os.path.join(op_dir, ".gitkeep"), 'w') as f:
                    pass

generate_cases_dir()

        

    
