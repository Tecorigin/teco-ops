#!/bin/bash
mkdir -p build && cd build

BUILD_SUPPORT=("teco" "cuda")
BUILD_TARGETS=("teco" "cuda")
while [[ $# -gt 0 ]]; do
    case "$1" in
        --build)
            IFS=';' read -ra BUILD_TARGETS <<< "$2"
            shift 2
            ;;
        *)
            echo "未知选项: $1"
            exit 1
            ;;
    esac
done

# 检查是否包含特定目标
if [[ " ${BUILD_TARGETS[*]} " =~ " teco " ]]; then
    echo "构建teco"
    # thirdparty
    cd ..
    python3 tools/deps.py
    cd build
    rm -rf ./teco && mkdir teco
    pushd ./teco
        # Extra configure flags (e.g. the CI-only strict device-section-order gate).
        # Intentional word splitting: values containing whitespace are not supported.
        cmake ../../teco/ ${TECO_OPS_CMAKE_ARGS:-}
        make -j
    popd
fi

if [[ " ${BUILD_TARGETS[*]} " =~ " cuda " ]]; then
    echo "构建cuda"
    rm -rf ./cuda && mkdir cuda
    pushd ./cuda
        cmake ../../cuda/
        make -j
    popd
fi
