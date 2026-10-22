#!/bin/bash
set -e

trap 'catch $? $LINENO' EXIT
catch() {
    if [ $1 -ne 0 ]; then
        echo "======================================================"
        echo "脚本失败于第 $2 行，退出码: $1"
        echo "失败阶段: $FAILED_STEP"
        exit 1
    fi
}

FAILED_STEP="安装依赖"
python -m pip install -r ../requirements.txt

FAILED_STEP="编译 teco kernel"
echo "======================================================"
echo "compile teco kernel"
pushd ..
    source /opt/tecoai/setvars.sh && bash ./build.sh --build teco
popd

FAILED_STEP="编译测试"
echo "======================================================"
echo "compile test"
pushd ../test
    source ./env.sh
    bash ./build.sh --arch teco
popd

ops=$(python ./get_op.py)

# get_op.py prints an empty result for AFFECT_OPS=all, which makes every later
# per-operator loop a silent no-op; say so explicitly so nobody mistakes the
# empty python/plugin stages for a failure.
if [ -z "$ops" ]; then
    echo "======================================================"
    echo "AFFECT_OPS=all (or no affected operator): python api and plugin test stages are skipped"
    echo "(get_op.py resolved an empty operator list; the C++ kernel stage above still ran fully)"
    echo "======================================================"
fi

FAILED_STEP="运行 teco kernel 测试"
echo "======================================================"
echo "run teco kernel"
mkdir -pv ../test/tools/result_teco
pushd ../test/tools/result_teco
    set -x
    python ../unit_test_v2.py --gid=0 --cases_dir ../../zoo/ --perf_repeat 50 --warm_repeat 3 --test_name "$ops"
    mv *xlsx result_teco.xlsx 2>/dev/null || true
    mv ./log/*/*log ./result_teco.log 2>/dev/null || true
popd

FAILED_STEP="构建和安装 python api"
echo "======================================================"
echo "build and install python api"
pushd ..
    python -m pip uninstall -y tecoops 2>/dev/null || true
    python setup.py bdist_wheel
    python -m pip install --force-reinstall ./dist/tecoops-*.whl
popd

# ---------------------------------------------------------------------------
# Gate for the python / plugin test stages.
#
# These test scripts are NOT pytest: several of them print a per-case "FAILED"
# line and still exit 0 (e.g. python_api_test/test_rms_norm.py returns
# all_passed to nobody). Checking only "$?" can never fail the pipeline, which
# is how a real boundary failure in test_rms_norm.py stayed invisible. A test
# counts as passed only when BOTH hold:
#   1. its process exit code is 0, and
#   2. its captured output has no unexpected "FAILED" line.
#
# CONVENTION for every test script in this repo:
#   * do NOT print a bare "FAILED" token for anything that is not a failure;
#   * a deliberate failure-path exercise must mark the line with EXPECTED_FAIL
#     (e.g. print("FAILED EXPECTED_FAIL: invalid shape rejected")); marked lines
#     are filtered out here;
#   * normal success summaries such as "0 FAILED" / "no FAILED" / "zero FAILED"
#     are recognised and NOT counted as failures (see FAILED_OK_RE below) -- if a
#     script needs another phrasing, extend that regex rather than dropping the
#     gate.
# Every test's output is kept in a log directory so a failure is diagnosable
# after the fact; the directory is printed at the end of the run (override it
# with CI_TEST_LOG_DIR=<dir> so CI can archive the per-test logs).
# ---------------------------------------------------------------------------
CI_TEST_LOG_DIR="${CI_TEST_LOG_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/tecoops-ci-tests.XXXXXX")}"
mkdir -p "${CI_TEST_LOG_DIR}"
CI_FAILED=()
EXPECTED_FAIL_MARK="EXPECTED_FAIL"

# A bare FAILED token anywhere (start of line or after a non-identifier char).
FAILED_RE='(^|[^[:alnum:]_])FAILED'
# ...but these success summaries legitimately contain the token. Keep this list
# explicit: widening it must not turn a real failure into a pass.
FAILED_OK_RE='(^|[^[:alnum:]_])(0|no|zero)[[:space:]]+FAILED'

run_gated_test() {
    # $1 = stage, $2 = test script (relative), $3 = log basename
    local stage="$1" script="$2" log rc=0 failed
    log="${CI_TEST_LOG_DIR}/$3"
    set +e
    python "$script" 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    set -e
    if [ "$rc" -ne 0 ]; then
        CI_FAILED+=("${stage}: ${script} exited ${rc}")
        echo "  ${script}: FAILED (exit code ${rc})"
        return 0
    fi
    # Sub-string matching alone would flag normal summaries such as "0 FAILED" or
    # "no FAILED"; filter those first, then drop EXPECTED_FAIL-marked lines.
    failed=$(grep -E "$FAILED_RE" "$log" \
             | grep -Eiv "$FAILED_OK_RE" \
             | grep -vF "$EXPECTED_FAIL_MARK" || true)
    if [ -n "$failed" ]; then
        CI_FAILED+=("${stage}: ${script} printed FAILED while exiting 0: $(printf '%s' "$failed" | head -n 3 | tr '\n' '|')")
        echo "  ${script}: FAILED (FAILED line in output, exit code 0)"
        return 0
    fi
    echo "  ${script}: PASSED (exit code 0, no FAILED line)"
    return 0
}

ci_report() {
    echo "======================================================"
    if [ ${#CI_FAILED[@]} -ne 0 ]; then
        echo "测试失败：${#CI_FAILED[@]} 个测试失败"
        printf '  - %s\n' "${CI_FAILED[@]}"
        echo "per-test logs kept in: ${CI_TEST_LOG_DIR}"
        FAILED_STEP="python/plugin 测试失败"
        exit 1
    fi
    echo "测试成功"
    echo "per-test logs kept in: ${CI_TEST_LOG_DIR}"
    exit 0
}
# ---- end of python/plugin gate helpers ----

FAILED_STEP="运行 python api 测试"
echo "======================================================"
echo "run python api tests"
cd ../python_api_test
IFS=';' read -ra OPS_ARRAY <<< "$ops"
for op in "${OPS_ARRAY[@]}"; do
    if [ -f "test_${op}.py" ]; then
        echo "Running test for operator: ${op}"
        run_gated_test "python" "test_${op}.py" "python__${op}.log"
    fi
done

FAILED_STEP="运行 plugin api 测试"
echo "======================================================"
echo "run plugin api tests"
export LD_LIBRARY_PATH=$(python -c "import tvm; import os; print(os.path.dirname(tvm.__file__))" | tail -n 1):${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=$(realpath ../api/tecoops):${LD_LIBRARY_PATH}
cd ../plugin_test
IFS=';' read -ra OPS_ARRAY <<< "$ops"
for op in "${OPS_ARRAY[@]}"; do
    if [ -f "test_plugin_${op}.py" ]; then
        echo "Running plugin test for operator: ${op}"
        run_gated_test "plugin" "test_plugin_${op}.py" "plugin__${op}.log"
    fi
done

FAILED_STEP="python/plugin 测试"
ci_report
