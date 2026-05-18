import subprocess
import os
import glob

COMMON_FILES = {'base_op.hpp', 'macro.h', 'convert.h', 'log.h', 'status.h', 'def.h', 'args.h'}

def is_common_file(filename):
    basename = os.path.basename(filename)
    if basename in COMMON_FILES:
        return True
    if basename.startswith('find_'):
        return True
    return False

result = subprocess.getstatusoutput('git show --name-only --pretty=""')

ops = set()
for filename in result[1].split('\n'):
    if not filename:
        continue
    parts = filename.split('/')

    if filename.startswith('teco/interface/ops/'):
        if len(parts) >= 3:
            basename = parts[-1]
            if basename.endswith('.cpp'):
                opname = basename.replace('.cpp', '')
                if not is_common_file(opname):
                    ops.add(opname)

    elif filename.startswith('teco/ual/ops/'):
        if len(parts) >= 4 and parts[3]:
            opname = parts[3]
            if not is_common_file(opname):
                ops.add(opname)

    elif filename.startswith('teco/ual/kernel/'):
        if len(parts) >= 4 and parts[3]:
            opname = parts[3]
            if not is_common_file(opname):
                ops.add(opname)

    elif filename.startswith('test/zoo/teco/'):
        if len(parts) >= 4:
            opname = parts[3]
            if not is_common_file(opname):
                ops.add(opname)

    elif filename.startswith('python_api_test/test_') and filename.endswith('.py'):
        basename = os.path.basename(filename)
        opname = basename.replace('test_', '').replace('.py', '')
        ops.add(opname)

op_string = ";".join(sorted(ops))
print(op_string)