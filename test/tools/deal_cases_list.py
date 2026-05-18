import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--op', dest='op', type=str, help='op: mv, rm, cp, rm_data')
    # (TODO)
    # parser.add_argument('--src_dir', dest='src_dir', type=str, help='src_dir')
    parser.add_argument('--case_path', dest='case_path', type=str, help='case path')
    parser.add_argument('--cases_list', dest='cases_list', type=str, help='cases list')
    parser.add_argument('--dst_dir', dest='dst_dir', type=str, help='dst_dir')
    args = parser.parse_args()

    op = 'mv'
    if args.op is not None:
        op = args.op
        if op == 'cp':
            op += " -r"

    if op == "mv" or op == "cp -r":
        if args.dst_dir is None:
            print("need --dst_dir")
            exit(-1)
        else:
            dst_dir = args.dst_dir
            if(~os.path.exists(dst_dir)):
                print(dst_dir, 'is not exist. will mkdir dir')
                os.system('mkdir -p ' + dst_dir)
    
    if op == 'rm':
        dst_dir = ''

    if args.cases_list is None and args.case_path is None:
        print('need --cases_list or --case_path')
        exit(-1)

    cases_list = []

    if args.case_path is not None:
        cases_list.append(args.case_path)

    if args.cases_list is not None:
        with open(args.cases_list, "r") as f:
            cases = f.readlines()
            cases = [x.strip() for x in cases]
        cases_list = cases

    for case_path in cases_list:
        case_name = case_path.rstrip('.prototxt')
        if op != 'rm_data':
            cmd = op + ' ' + case_name + '*' + ' ' + dst_dir
        else:
            cmd = 'rm -rf ' + case_name

        os.system(cmd)

    if(op != 'rm' and op != 'rm_data'):
        os.system('chmod -R 777 ' + dst_dir)
