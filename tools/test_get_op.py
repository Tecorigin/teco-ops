# BSD 3- Clause License Copyright (c) 2024, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.

"""Pure-Python, device-free unit tests for tools/get_op.py.

Runs standalone (no repo build, no SDAA device, no pytest):

    python tools/test_get_op.py

The module under test is imported by path so the test keeps working no matter
which working directory (or package layout) the CI invokes it from.
"""

import importlib.util
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULE_PATH = os.path.join(_HERE, 'get_op.py')
_spec = importlib.util.spec_from_file_location('get_op_under_test', _MODULE_PATH)
get_op = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(get_op)

# The whitelist that ``load_public_ops`` would derive from
# ``teco/interface/ops/*.cpp``; used everywhere a fixed, device-free whitelist is
# needed (``collect_ops`` re-derives its own when driven end to end).
PUBLIC_OPS = {
    'causal_conv1d', 'flash_attention', 'flatten_rays', 'memset',
    'morton3D_invert', 'reduce_variance', 'reshape_and_cache', 'rms_norm',
}


def _with_public_ops(monkey_patched):
    """Run ``collect_ops`` with a fixed whitelist instead of the repo glob."""
    original = get_op.load_public_ops
    get_op.load_public_ops = lambda: PUBLIC_OPS
    try:
        return monkey_patched()
    finally:
        get_op.load_public_ops = original


class NormalizeTokenTest(unittest.TestCase):
    def test_strips_source_extensions(self):
        for token, expect in (('x.cpp', 'x'), ('x.hpp', 'x.hpp'), ('x.scpp', 'x'), ('x.h', 'x')):
            self.assertEqual(get_op.normalize_token(token), expect)

    def test_extension_strip_matches_collect_ops_usage(self):
        # ``.cpp`` ``.scpp`` and ``.h`` are stripped; other extensions stay intact.
        self.assertEqual(get_op.normalize_token('gemv.hpp'), 'gemv.hpp')

    def test_strips_leading_z_prefixes(self):
        self.assertEqual(get_op.normalize_token('z_foo.scpp'), 'foo')
        self.assertEqual(get_op.normalize_token('zz_foo.scpp'), 'foo')
        self.assertEqual(get_op.normalize_token('zzz_flash_attention_d128.scpp'),
                         'flash_attention_d128')
        self.assertEqual(get_op.normalize_token('reshape_and_cache_dma.scpp'),
                         'reshape_and_cache_dma')

    def test_strips_find_prefix(self):
        self.assertEqual(get_op.normalize_token('find_weight_stream_gemv.cpp'),
                         'weight_stream_gemv')

    def test_no_path_component_leaks(self):
        self.assertEqual(
            get_op.normalize_token('teco/ual/kernel/zz_ms_deform_attn_backward_ha/x.scpp'),
            'x')


class ResolveCandidatesTest(unittest.TestCase):
    def test_exact_public_name_wins(self):
        self.assertEqual(get_op.resolve_candidates(['rms_norm'], PUBLIC_OPS), 'rms_norm')

    def test_prefixed_kernel_tu_resolves_to_public_op(self):
        self.assertEqual(
            get_op.resolve_candidates(['zzz_flash_attention_d128'], PUBLIC_OPS),
            'flash_attention')
        self.assertEqual(
            get_op.resolve_candidates(['z_reshape_and_cache_dma'], PUBLIC_OPS),
            'reshape_and_cache')

    def test_internal_ha_suffix_tu_not_resolvable_without_whitelist_entry(self):
        # ``ms_deform_attn_backward`` is not in the CURRENT main-tree whitelist
        # (no teco/interface/ops/ms_deform_attn_backward.cpp yet, it comes with PR #40);
        # until it lands the token is kept as the raw file name (add_resolved fallback),
        # which is exactly what keeps CI honest instead of silently guessing.
        self.assertIsNone(
            get_op.resolve_candidates(['zz_ms_deform_attn_backward_ha'], PUBLIC_OPS))

    def test_find_prefix_token_leaves_unknown_op_unresolved(self):
        # ``weight_stream_gemv`` rides in with PR #44; against the fast-path whitelist
        # it is not an exact public op and matches no ``<op>_`` prefix, so it stays
        # unresolved (the raw file name is kept by add_resolved).
        self.assertIsNone(
            get_op.resolve_candidates(['weight_stream_gemv'], PUBLIC_OPS))

    def test_find_prefix_token_resolves_when_op_is_public(self):
        ops = {'weight_stream_gemv'}
        self.assertEqual(
            get_op.resolve_candidates(['weight_stream_gemv'], ops), 'weight_stream_gemv')

    def test_unknown_token_resolves_to_none(self):
        self.assertIsNone(get_op.resolve_candidates(['totally_unrelated'], PUBLIC_OPS))

    def test_longest_public_prefix_wins(self):
        # a token that matches two public ops as a prefix resolves to the longer one
        self.assertEqual(
            get_op.resolve_candidates(['rms_norm_extra'], PUBLIC_OPS), 'rms_norm')


class CollectOpsTest(unittest.TestCase):
    def test_public_op_interface_dir(self):
        ops = _with_public_ops(lambda: get_op.collect_ops(['teco/interface/ops/rms_norm.cpp']))
        self.assertEqual(ops, {'rms_norm'})

    def test_common_files_do_not_become_ops(self):
        # ``base_op.hpp`` under teco/ual/ops/ is not inside ``teco/interface/ops/``
        # (that branch requires .cpp there), so it contributes nothing; a ``find_``
        # file under a real op dir resolves to that op -- the dir name
        # participates for find_ files, which is intended (e.g.
        # ``find_rms_norm_compat.cpp -> rms_norm_compat``, verified by the official review).
        ops = _with_public_ops(lambda: get_op.collect_ops([
            'teco/ual/ops/base_op.hpp',
            'teco/ual/ops/rms_norm/find_rms_norm.cpp',
        ]))
        self.assertEqual(ops, {'rms_norm'})

    def test_find_file_without_matching_dir_stays_silent(self):
        # ... but a find_ file whose dir matches NO public op adds nothing
        ops = _with_public_ops(lambda: get_op.collect_ops([
            'teco/ual/ops/gemv/find_weight_stream_gemv.cpp',
        ]))
        self.assertEqual(ops, set())

    def test_common_helper_files_added_nothing(self):
        ops = _with_public_ops(lambda: get_op.collect_ops([
            'teco/interface/ops/base_op.hpp',
            'teco/ual/kernel/macro.h',
        ]))
        self.assertEqual(ops, set())

    def test_prefixed_kernel_tus_resolve_to_public_ops(self):
        ops = _with_public_ops(lambda: get_op.collect_ops([
            'teco/ual/kernel/zzz_flash_attention_d128/zzz_flash_attention_d128.scpp',
            'teco/ual/kernel/z_reshape_and_cache_dma/reshape_and_cache_dma.scpp',
        ]))
        self.assertEqual(ops, {'flash_attention', 'reshape_and_cache'})

    def test_whitelist_absent_tu_kept_raw(self):
        # same PR-family behaviour as above: when the public op is not yet in the
        # whitelist (its interface cpp comes with another PR), the normalized raw
        # token is kept so CI still runs a full-range C++ regression rather than
        # skipping silently.
        ops = _with_public_ops(lambda: get_op.collect_ops([
            'teco/ual/kernel/zz_ms_deform_attn_backward_ha/zz_ms_deform_attn_backward_ha.scpp',
        ]))
        self.assertEqual(ops, {'zz_ms_deform_attn_backward_ha'})

    def test_shared_container_dir_gemv_bare_path_silent(self):
        # the container dir itself (no file part) must not emit a phantom ``gemv``
        ops = _with_public_ops(lambda: get_op.collect_ops([
            'teco/ual/ops/gemv',
        ]))
        self.assertEqual(ops, set())

    def test_shared_container_dir_gemv_bare_helper_file_not_phantom(self):
        # a generic helper inside teco/ual/ops/gemv/ must not emit a phantom
        # ``gemv`` operator; the file basename is kept raw (unresolved) instead,
        # and ``gemv`` itself never appears.
        ops = _with_public_ops(lambda: get_op.collect_ops([
            'teco/ual/ops/gemv/gemv.hpp',
        ]))
        self.assertEqual(ops, {'gemv.hpp'})
        self.assertNotIn('gemv', ops)

    def test_shared_container_dir_gemv_resolves_wsg_pr51_tree(self):
        # against the current PR #51 tree whitelist (no weight_stream_gemv yet),
        # a find_ file inside the container resolves to nothing and emits nothing
        ops = _with_public_ops(lambda: get_op.collect_ops([
            'teco/ual/ops/gemv/find_weight_stream_gemv.cpp',
        ]))
        self.assertEqual(ops, set())

    def test_shared_container_dir_gemv_resolves_wsg_when_public(self):
        # with ``weight_stream_gemv`` in the whitelist (PR #44 landed), files in the
        # shared gemv container still resolve to the real public op
        original = get_op.load_public_ops
        get_op.load_public_ops = lambda: PUBLIC_OPS | {'weight_stream_gemv'}
        try:
            ops = get_op.collect_ops([
                'teco/ual/ops/gemv/find_weight_stream_gemv.cpp',
                'teco/ual/kernel/weight_stream_gemv/weight_stream_gemv.scpp',
            ])
        finally:
            get_op.load_public_ops = original
        self.assertEqual(ops, {'weight_stream_gemv'})

    def test_zoo_dir_maps_to_op_name(self):
        ops = _with_public_ops(lambda: get_op.collect_ops([
            'test/zoo/teco/rotary_embedding/test_case/0.prototxt',
        ]))
        self.assertEqual(ops, {'rotary_embedding'})

    def test_python_api_test_maps_to_op_name(self):
        ops = _with_public_ops(lambda: get_op.collect_ops([
            'python_api_test/test_rms_norm.py',
        ]))
        self.assertEqual(ops, {'rms_norm'})

    def test_unrelated_file_ignored(self):
        # a path matching no known prefix branch is simply not an operator signal
        ops = _with_public_ops(lambda: get_op.collect_ops([
            'api/tecoops/__init__.py',
        ]))
        self.assertEqual(ops, set())

    def test_empty_input(self):
        ops = _with_public_ops(lambda: get_op.collect_ops(['', '  ']))
        self.assertEqual(ops, set())


class MainBehaviourTest(unittest.TestCase):
    def test_paths_mode_prints_sorted_ops(self):
        rc, out = _capture_main(['--paths',
                                 'teco/ual/kernel/zzz_flash_attention_d128/x.scpp '
                                 'teco/ual/kernel/weight_stream_gemv/x.scpp'])
        self.assertEqual(rc, 0)
        self.assertEqual(out.strip(), 'flash_attention;weight_stream_gemv')

    def test_all_keyword_prints_empty(self):
        env = {'AFFECT_OPS': 'all'}
        old = os.environ.get('AFFECT_OPS')
        os.environ['AFFECT_OPS'] = 'all'
        try:
            rc, out = _capture_main([])
        finally:
            if old is None:
                os.environ.pop('AFFECT_OPS', None)
            else:
                os.environ['AFFECT_OPS'] = old
        self.assertEqual(rc, 0)
        self.assertEqual(out.strip(), '')

    def test_semicolon_list_is_passed_through(self):
        env = 'rms_norm;flash_attention'
        old = os.environ.get('AFFECT_OPS')
        os.environ['AFFECT_OPS'] = env
        try:
            rc, out = _capture_main([])
        finally:
            if old is None:
                os.environ.pop('AFFECT_OPS', None)
            else:
                os.environ['AFFECT_OPS'] = old
        self.assertEqual(rc, 0)
        self.assertEqual(out.strip(), 'flash_attention;rms_norm')


def _capture_main(argv):
    """Run get_op.main(argv) while capturing stdout."""
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = get_op.main(['get_op.py'] + argv)
    return rc, buf.getvalue()


if __name__ == '__main__':
    unittest.main(verbosity=2)
