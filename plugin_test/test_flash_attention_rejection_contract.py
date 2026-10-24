# BSD 3-Clause License
# Copyright (c) 2024, Tecorigin Co., Ltd. All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""No-device tests for SDK rejection and direct C API assertion boundaries."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from flash_attention_c_api import check_unsupported
from flash_attention_rejection import assert_sdk_rejected


class RejectionContractTest(unittest.TestCase):
    def sdk(self, message, output=None):
        ctx = Mock()
        ctx.executor_run.side_effect = RuntimeError(message)
        ctx.get_output.return_value = np.zeros(16) if output is None else output
        return ctx

    def test_sdk_reallocated_zero_output_is_diagnostic(self):
        ctx = self.sdk("rejected parameters (status 6)")
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            assert_sdk_rejected(ctx, "unsupported")
        ctx.set_output.assert_not_called()
        for field in ("min=", "max=", "mean=", "std=", "first8=",
                      "SDK_buffer_alias_guaranteed=False", "SDK sentinel not asserted"):
            self.assertIn(field, log.getvalue())

    def test_sdk_erased_exception_with_nonzero_output_is_diagnostic(self):
        # These bytes cannot establish aliasing or infer a kernel/fallback write.
        assert_sdk_rejected(self.sdk("std::exception", np.arange(16)), "unsupported")

    def test_sdk_unreadable_output_does_not_claim_sentinel(self):
        ctx = self.sdk("status=6")
        ctx.get_output.side_effect = RuntimeError("output not allocated")
        assert_sdk_rejected(ctx, "unsupported")

    def test_sdk_readable_wrong_status_is_rejected(self):
        for message in ("status 5", "rejected parameters (status=7)", "std::exception status: 4"):
            with self.subTest(message=message), self.assertRaises(AssertionError):
                assert_sdk_rejected(self.sdk(message), "unsupported")

    def test_sdk_unrelated_error_or_success_is_rejected(self):
        with self.assertRaises(AssertionError):
            assert_sdk_rejected(self.sdk("out of memory"), "unsupported")
        ctx = self.sdk("unused")
        ctx.executor_run.side_effect = None
        with self.assertRaises(AssertionError):
            assert_sdk_rejected(ctx, "unsupported")

    def test_direct_c_api_requires_status_and_untouched_sentinel(self):
        with tempfile.NamedTemporaryFile() as core:
            path = Path(core.name)
            with patch("flash_attention_c_api.run_flash", return_value=(6, np.full(8, 7.0))) as run:
                check_unsupported(path)
                self.assertEqual(run.call_count, 2)
            for result in ((0, np.full(8, 7.0)), (6, np.zeros(8))):
                with self.subTest(result=result[0]), patch(
                    "flash_attention_c_api.run_flash", return_value=result
                ), self.assertRaises(AssertionError):
                    check_unsupported(path)


if __name__ == "__main__":
    unittest.main()
