import os

header_code_template = '''
#ifndef {}  // NOLINT
#define {}

#include "tensor.h"
#include "zoo/teco/executor.h"

namespace optest {{
class {}Executor : public TecoExecutor {{
 public:
    {}Executor() {{}}
    ~{}Executor() {{}}

    void paramCheck();
    void paramParse();
    void paramGeneration();
    void compute();
    void cpuCompute();
    void gpuCompute();
    int64_t getTheoryOps() override;
    int64_t getTheoryIoSize() override;
    void destroy();

 private:
}};
}};  // namespace optest

#endif
'''

cpp_code_template = """
#include <stdio.h>
#include <iostream>
#include <string>
#include <tensor.h>
#include "zoo/teco/convert.h"
#include "common/time.hpp"

#include "zoo/teco/{}/{}.h"
#include "tensor.h"
#include "{}/{}.h"

namespace optest {{

void {}Executor::destroy() {{
}}

void {}Executor::paramCheck() {{
}}

void {}Executor::paramParse() {{
}}

void {}Executor::paramGeneration() {{
}}

void {}Executor::compute() {{
#ifdef USE_TECO
    {}();
#endif
}}

int64_t {}Executor::getTheoryOps() {{
    int64_t theory_ops = parser_->input(0)->shape_count;
    return theory_ops;
}}

int64_t {}Executor::getTheoryIoSize() {{
    // return getIoSizeWithBeta(beta_);
    return 0;
}}

void {}Executor::cpuCompute() {{
}}

void {}Executor::gpuCompute() {{
#ifdef USE_CUDA
    {}();
#endif
}}

}}  // namespace optest

"""

def snake_to_camel(op_name):
#     components = snake_str.split('_')
#     return ''.join(x.title() for x in components[0:])
# def upper_camel(op_name):
    res = ""
    slices = op_name.split('_')
    for item in slices:
        res += item.capitalize()
    return res



filename = ''

teco_kernel_test_project_dir = os.path.join(os.path.dirname(__file__), '../')
teco_support_op_name = list(set(os.listdir('../teco')) - set(['CMakeLists.txt']))

def get_op_name_header_test_code_str(op_name):
    class_name = snake_to_camel(op_name)
    header_head = 'ZOO_TECO_{}_{}_H_'.format(op_name.upper(), op_name.upper())
    code_str = header_code_template.format(header_head, header_head, class_name, class_name, class_name)
    return code_str

def get_op_name_cpp_test_code_str(op_name):
    class_name = snake_to_camel(op_name)
    print(class_name)
    api_name = op_name + '_launch'
    code_str = cpp_code_template.format(op_name, op_name, op_name, op_name, 
                                        class_name, class_name, class_name,class_name,
                                        class_name,api_name, class_name, class_name,
                                        class_name, class_name, api_name)
    return code_str

for op_name in teco_support_op_name:
    header_code = get_op_name_header_test_code_str(op_name)
    cpp_code = get_op_name_cpp_test_code_str(op_name)

    test_code_dir = f'{teco_kernel_test_project_dir}/test/zoo/todo/{op_name}'
    header_path = os.path.join(test_code_dir, f'{op_name}.h')
    cpp_path = os.path.join(test_code_dir, f'{op_name}.cpp')
    os.system(f'mkdir -p {test_code_dir}')
    with open(header_path, 'w') as f:
        f.write(header_code)
    with open(cpp_path, 'w') as f:
        f.write(cpp_code)
