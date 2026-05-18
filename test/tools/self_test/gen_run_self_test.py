import argparse
from GitRepository import GitRepository

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tecoal_path", 
                        dest="tecoal_path", 
                        type=str, 
                        default="/data/dnn/zhuangxin/tecoal/",
                        help="tecoal path")
    parser.add_argument("--tecotest_path", 
                        dest="tecotest_path", 
                        type=str, 
                        default="/data/dnn/zhuangxin/tecoal/test/frame_work/tecotest/",
                        help="tecotest path")
    parser.add_argument("--mode", 
                        dest="mode", 
                        type=str, 
                        default="pr_test",
                        help="test mode ,must be pr_test or coverage")
    args = parser.parse_args()
    tecoal_path = args.tecoal_path
    tecotest_path = args.tecotest_path
    mode = args.mode

    tecoal = GitRepository(local_path = tecoal_path, repo_url = "ssh://gerrit@10.10.30.1:29418/tecoal")
    tecotest = GitRepository(local_path = tecotest_path, repo_url = "ssh://gerrit@10.10.30.1:29418/tecotest", is_submodule=True)
    latest_commit = tecoal.latest_commit()

    latest_commit_detail_list = latest_commit.split("\n")
    for i in latest_commit_detail_list:
        if "cases_list" in i:
            cases_list = i.split(":")[-1].strip()
        if "tecotest_branch" in i:
            tecotest_branch = i.split(":")[-1].strip()
    print(cases_list)
    print(tecotest_branch)
    tecotest.change_to_branch(tecotest_branch)
    # subprocess.run()

    sh_name = "run_self_test.sh"
    fw1 = open(sh_name,'w',encoding='utf-8')
    sh_head = '''#!/bin/bash
source /opt/tecoai/setvars.sh
export DNN_CHECK_ALL_MEM=1
'''
    fw1.write(sh_head)
    fw1.write("cd " + tecotest_path + '\n')
    # fw1.write("git checkout -b " + tecotest_branch + " origin/" + tecotest_branch + '\n')
    fw1.write("source env.sh && cd tools && bash make_tecotest.sh\n")
    fw1.write("export LD_LIBRARY_PATH=" + tecoal_path + "build/lib:" + tecotest_path + "test_lib/lib:$LD_LIBRARY_PATH\n")
    if mode == "coverage":
        fw1.write("python3 unit_test_v2.py --gid=0,1,2 --rand_n=1000 --test_stable off --perf_repeat=0 --warm_repeat=0 --gtest_repeat=1 --cases_list=" + cases_list + '\n')
    else:
        fw1.write("python3 unit_test_v2.py --gid=0,1,2 --rand_n=1000 --cases_list=" + cases_list + '\n')
    fw1.write("mkdir -p " + tecoal_path + "test_result/\n")
    fw1.write("mv *.xlsx " + tecoal_path + "test_result/self_test.xlsx")
