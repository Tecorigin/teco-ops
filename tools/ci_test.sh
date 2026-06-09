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

FAILED_STEP="运行 python api 测试"
echo "======================================================"
echo "run python api tests"
cd ../python_api_test
IFS=';' read -ra OPS_ARRAY <<< "$ops"
for op in "${OPS_ARRAY[@]}"; do
    if [ -f "test_${op}.py" ]; then
        echo "Running test for operator: ${op}"
        python "test_${op}.py"
        echo "test_${op}.py: $?"
    fi
done

FAILED_STEP="运行 plugin api 测试"
echo "======================================================"
echo "run plugin api tests"
export LD_LIBRARY_PATH=$(realpath ../api/tecoops):${LD_LIBRARY_PATH}
cd ../plugin_test
IFS=';' read -ra OPS_ARRAY <<< "$ops"
for op in "${OPS_ARRAY[@]}"; do
    if [ -f "test_plugin_${op}.py" ]; then
        echo "Running plugin test for operator: ${op}"
        python "test_plugin_${op}.py"
        echo "test_plugin_${op}.py: $?"
    fi
done

echo "======================================================"
echo "测试成功"
exit 0