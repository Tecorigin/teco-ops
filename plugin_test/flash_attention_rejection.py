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

"""SDK rejection contract; direct C API status/sentinel checks must run first."""
import re

import numpy as np


def _output_diagnostics(ctx, name):
    # SDK set_output/get_output have no buffer-alias guarantee. These values
    # diagnose SDK allocation behavior; they cannot prove whether a kernel wrote.
    print(f"  {name}: SDK_buffer_alias_guaranteed=False; output is diagnostic only")
    try:
        output = np.asarray(ctx.get_output(0)).astype(np.float32).reshape(-1)
        if output.size:
            with np.errstate(invalid="ignore", over="ignore"):
                print(f"  {name}: SDK_output min={output.min()} max={output.max()} "
                      f"mean={output.mean()} std={output.std()} first8={output[:8].tolist()}")
        else:
            print(f"  {name}: SDK_output empty; first8=[]")
    except Exception as exc:
        print(f"  {name}: SDK_output unreadable: {type(exc).__name__}: {exc}")


def assert_sdk_rejected(ctx, name):
    """Require SDK rejection and status 6 whenever its exception exposes a code.

    Some SDK versions erase runtime_error.what() to std::exception. That exact
    transport limitation is accepted only after the caller has independently
    asserted status 6 and an untouched sentinel on the engine's mapped C API.
    """
    try:
        ctx.executor_run()
    except RuntimeError as exc:
        _output_diagnostics(ctx, name)
        message = str(exc)
        codes = re.findall(r"status\s*[:=]?\s*(\d+)", message, flags=re.IGNORECASE)
        if codes:
            assert all(int(code) == 6 for code in codes), f"{name}: expected SDK status 6: {message}"
        else:
            assert "rejected parameters" in message or "std::exception" in message, (
                f"{name}: unexpected SDK exception: {message}"
            )
        status = "SDK_status=6" if codes else "SDK_status=unavailable; C_API_status=6 verified separately"
        print(f"  {name}: PASSED (EXPECTED_FAIL, {status}; SDK sentinel not asserted)")
    else:
        _output_diagnostics(ctx, name)
        raise AssertionError(f"{name}: SDK accepted an unsupported configuration")
