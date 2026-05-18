#!/usr/bin/python

# configparser python2/python3
try:
    import ConfigParser as configparser
except:
    import configparser

from string import Template

import os
import sys
import re

def grep_info(ini_file_path):
    print("registering %s" % os.path.basename(ini_file_path))
    ini = configparser.ConfigParser()
    ini.read(ini_file_path)

    op_name = ini.get('meta', 'name')
    op_branches = [i.strip() for i in
        ini.get('meta', 'branches').split(",")]

    return (op_name, op_branches)

def upper_camel(op_name):
    res = ""
    slices = op_name.split('_')
    for item in slices:
        res += item.capitalize()
    return res

if __name__ == "__main__":
    device_arch = sys.argv[1]
    al_type = sys.argv[2]
    al_types = []
    if al_type == "all":
        al_types = ["dnn", "blas", "customblas", "custom"]
    elif al_type == "al":
        al_types = ["dnn", "blas"]
    elif al_type == "custom":
        al_types = ["customblas", "custom"]
    else:
        al_types = [al_type]

    specific_ops = []
    if len(sys.argv) > 3 and len(sys.argv[3]) > 0:
        specific_ops = sys.argv[3:]
    print(specific_ops)
    
    inis_list = []
    for al_type in al_types:
        curr_path = os.path.dirname(os.path.abspath(__file__))
        device_arch_path = os.path.join(curr_path, "../%s/%s" % (device_arch, al_type))
        inis = []
        if os.path.exists(device_arch_path):
            for path in os.listdir(device_arch_path):
                if path.startswith("__"):
                    continue
                if len(specific_ops) > 0:
                    if path not in specific_ops:
                        continue
                if os.path.isdir(os.path.join(device_arch_path, path)):
                    inis.append(path)
        inis.sort()
        inis_list.append(inis)

    # generate case_list.cpp
    header = """/************************************************************************
 *
 *  @Automatically generated test code
 *
 *
 **************************************************************************/
#include <dirent.h>
#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <algorithm>
#include <cstdlib>
#include "common/variable.h"
#include "case/case_collector.h"
#include "suite/suite.h"

using namespace ::testing;
TEST_P(TestOpFixture, kernel) { Run(); }

// register op testcase here.
// AUTO GENERATE START
"""
    tailer = "// AUTO GENERATE END"
    cast_list_file = os.path.join(curr_path + "/../src/suite/", "case_list.cpp")
    file = open(cast_list_file, "w")
    file.write(header)
    for i in range(len(al_types)):
        al_type = al_types[i]
        inis = inis_list[i]
        for op in inis:
            addstr="INSTANTIATE_TEST_SUITE_P(%s_%s, TestOpFixture, Combine(Values(\"%s\"), Range(size_t(0), Collector(\"%s\", \"%s\").num())));\n" %(al_type, op, op, al_type, op)
            file.write(addstr)
    file.write(tailer)
    print("Success generate case_list.cpp.")


    # generate op_register.h
    filename = os.path.join(curr_path + "/../src/suite/", "op_register.h")
    file = open(filename, "w")
    header = """#ifndef SUITE_OP_REGISTER_H_
#define SUITE_OP_REGISTER_H_

#include <string>
#include "suite/executor.h"
// AUTO GENERATE HEADER START
"""
    tailer = """// AUTO GENERATE HEADER END
std::shared_ptr<optest::Executor> getOpExecutor(std::string op_name);
#endif  // SUITE_OP_REGISTER_H_
"""
    file.write(header)
    for i in range(len(al_types)):
        al_type = al_types[i]
        inis = inis_list[i]
        for op in inis:
            addstr = "#include \"%s/%s/%s/%s.h\"\n" % (device_arch, al_type, op, op)
            file.write(addstr)
    file.write(tailer)


    # generate op_register.cpp
    filename = os.path.join(curr_path + "/../src/suite/", "op_register.cpp")
    file = open(filename, "w")

    header = """ #include "op_register.h" 
std::shared_ptr<optest::Executor> getOpExecutor(std::string op_name) {
  if (false) {
"""
    tailer = """} else {
    ALLOG(ERROR) << "UnKnown op: " << op_name;
    exit(1);
  }
}
"""
    file.write(header)
    for i in range(len(al_types)):
        al_type = al_types[i]
        inis = inis_list[i]
        for op in inis:
            addstr = "  } else if (op_name == \"%s\") {\n" % (op)
            file.write(addstr)
            addstr = "    return std::make_shared<optest::%sExecutor>();\n" % upper_camel(op)
            file.write(addstr)
    file.write(tailer)
    file.close()
    print("Success generate op_register.cpp.")
    exit()
