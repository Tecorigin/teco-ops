import argparse

test_param='''
test_param {{ {} {}
}}
'''

threshold = '''
  error_func: {}
  error_threshold: {} '''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--case_path', dest='case_path', type=str, help='case path')
    parser.add_argument('--cases_list', dest='cases_list', type=str, help='case path')
    parser.add_argument('--diff3_max', dest='diff3_max', type=str, help='diff3_max threshold')
    parser.add_argument('--diff3_mean', dest='diff3_mean', type=str, help='diff3_mean threshold')
    args = parser.parse_args()

    diff3_max = ''
    if args.diff3_max is not None:
        diff3_max = threshold.format('DIFF3_MAX', args.diff3_max)

    diff3_mean = ''
    if args.diff3_mean is not None:
        diff3_mean = threshold.format('DIFF3_MEAN', args.diff3_mean)

    cases_list = []

    if args.case_path is not None:
        cases_list.append(args.case_path)

    if args.cases_list is not None:
        with open(args.cases_list, "r") as f:
            cases = f.readlines()
            cases = [x.strip() for x in cases]
        cases_list = cases

    for case_path in cases_list:
        with open(case_path, "a") as f:
            f.write(test_param.format(diff3_max, diff3_mean))
