"""Resolve the operators affected by the current commit (AFFECT_OPS).

The public operator names are the basenames of ``teco/interface/ops/*.cpp``.
Paths under ``teco/ual/kernel`` and ``teco/ual/ops`` may carry local
decorations (``z_``/``zz_``/``zzz_`` directory prefixes, ``find_`` file
prefixes, source extensions), so a candidate token is normalised and then
resolved to the public name it belongs to: an exact public name wins, else the
longest public name followed by a ``_`` boundary is used. Tokens that match no
public operator are kept unchanged.

Usage::

    python tools/get_op.py                 # operators touched by HEAD
    AFFECT_OPS=all python tools/get_op.py  # print an empty line
    python tools/get_op.py --paths "teco/ual/kernel/zzz_flash_attention_d128/x.scpp"
"""

import glob
import os
import re
import subprocess
import sys

COMMON_FILES = {'base_op.hpp', 'macro.h', 'convert.h', 'log.h', 'status.h', 'def.h', 'args.h', 'executor.py'}

SOURCE_EXTENSION_RE = re.compile(r'\.(?:cpp|h|scpp)$')
LEADING_Z_RE = re.compile(r'^z{1,3}_')

# Shared container directories under ``teco/ual/ops/`` that hold helper classes for other
# operators but are NOT public operators themselves (no ``teco/interface/ops/<name>.cpp``
# exists).  Without this exclusion ``collect_ops`` would emit a phantom ``gemv`` operator from
# the generic part of ``teco/ual/ops/gemv/`` even though only ``weight_stream_gemv`` lives there.
UAL_OPS_CONTAINER_DIRS = {'gemv'}


def is_common_file(filename):
    basename = os.path.basename(filename)
    if basename in COMMON_FILES:
        return True
    if basename.startswith('find_'):
        return True
    return False


def normalize_token(token):
    """Strip source extension and local prefixes from a candidate token."""
    token = os.path.basename(token.replace('\\', '/'))
    token = SOURCE_EXTENSION_RE.sub('', token)
    if token.startswith('find_'):
        token = token[len('find_'):]
    token = LEADING_Z_RE.sub('', token)
    return token


def load_public_ops():
    """Public operator names = ``teco/interface/ops/<op>.cpp`` basenames."""
    ops = set()
    for path in glob.glob(os.path.join('teco', 'interface', 'ops', '*.cpp')):
        opname = os.path.basename(path)[:-len('.cpp')]
        if not is_common_file(opname):
            ops.add(opname)
    return ops


def resolve_candidates(candidates, public_ops):
    """Map candidate tokens to a public operator name, or ``None``."""
    tokens = [normalize_token(candidate) for candidate in candidates]
    exact = [token for token in tokens if token in public_ops]
    if exact:
        return max(exact, key=len)
    best = ''
    for token in tokens:
        for opname in public_ops:
            if token.startswith(opname + '_') and len(opname) > len(best):
                best = opname
    return best or None


def add_resolved(tokens, raw_token, ops, public_ops):
    if is_common_file(raw_token):
        return
    resolved = resolve_candidates(tokens, public_ops)
    ops.add(resolved if resolved else raw_token)


def collect_ops(filenames):
    public_ops = load_public_ops()
    ops = set()
    for filename in filenames:
        filename = filename.strip().replace('\\', '/')
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
                # ``teco/ual/ops/gemv/`` (see UAL_OPS_CONTAINER_DIRS) is a shared container dir --
                # generic gemv helpers reused by other operators, not a public operator.  The
                # directory name itself must never become an operator token (that is how an
                # unresolved helper used to emit a phantom ``gemv``); files inside it
                # (``weight_stream_gemv``, ``find_weight_stream_gemv``, ...) still resolve to
                # their public operator as usual, with the file basename as the fallback token.
                if parts[3] in UAL_OPS_CONTAINER_DIRS:
                    if len(parts) >= 5:
                        # a real file inside the container: only the file part participates
                        add_resolved([parts[-1]], parts[-1], ops, public_ops)
                    else:
                        # A bare container dir path is not an operator. Say so on stderr
                        # (stdout stays the machine-readable AFFECT_OPS list) so a CI log
                        # reader can tell an intentional exclusion from a silent gap.
                        sys.stderr.write(
                            '[get_op] skip shared operator container: %s\n' % parts[3])
                else:
                    add_resolved([parts[3], parts[-1]], parts[3], ops, public_ops)
        elif filename.startswith('teco/ual/kernel/'):
            if len(parts) >= 4 and parts[3]:
                add_resolved([parts[3]], parts[3], ops, public_ops)

        elif filename.startswith('test/zoo/teco/'):
            if len(parts) >= 4:
                opname = parts[3]
                if not is_common_file(opname):
                    ops.add(opname)

        elif filename.startswith('python_api_test/test_') and filename.endswith('.py'):
            basename = os.path.basename(filename)
            opname = basename.replace('test_', '').replace('.py', '')
            ops.add(opname)

    return ops


def main(argv):
    args = argv[1:]
    paths = None
    if args:
        if args[0] in ('-h', '--help'):
            print(__doc__)
            return 0
        if args[0] == '--paths' and len(args) >= 2:
            paths = args[1]
        elif args[0].startswith('--paths='):
            paths = args[0].split('=', 1)[1]
        else:
            sys.stderr.write('usage: get_op.py [--paths "<path> <path> ..."]\n')
            return 2

    if paths is not None:
        ops = collect_ops(paths.replace('\n', ' ').split())
    else:
        affect_ops = os.environ.get('AFFECT_OPS', '')
        if affect_ops == 'all':
            print('')
            return 0
        if affect_ops:
            ops = set(affect_ops.split(';'))
        else:
            result = subprocess.getstatusoutput('git show --name-only --pretty=""')
            ops = collect_ops(result[1].split('\n'))

    print(';'.join(sorted(ops)))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
