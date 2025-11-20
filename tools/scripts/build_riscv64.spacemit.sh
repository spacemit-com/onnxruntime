#!/bin/bash
# bash build_riscv64.spacemit.sh <onnxruntime_src_dir> <arch> <config>,
# arch is x86 or rv64, config is Release or RelWithDebInfo or Debug
# before run this script, please ensure that RISCV_ROOT_PATH is set to the path of riscv toolchain
# https://archive.spacemit.com/toolchain/spacemit-toolchain-linux-glibc-x86_64-v1.1.2.tar.xz

ARCH=$(uname -m)
MATCH_ARCH=""
if [ "${ARCH}" = "x86_64" ]; then
    MATCH_ARCH="x86"
    elif [ "${ARCH}" = "riscv64" ]; then
    MATCH_ARCH="rv64"
fi

EXTERN_ARGS=

if [ "${2}" = "${MATCH_ARCH}" ]; then
    echo "BUILD ORT With Pyhon"
    EXTERN_ARGS="${EXTERN_ARGS} --enable_pybind --build_wheel"
fi

BUILD_DIR=${1}/build/Linux/${2}

EXTERN_ARGS="${EXTERN_ARGS} \
    --cmake_extra_defines \
    onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS=ON \
    CMAKE_INSTALL_PREFIX=installed"

python3 ${1}/tools/ci_build/build.py --build_dir ${BUILD_DIR} --config ${3} \
    --update --build --build_shared_lib --parallel 20 \
    --compile_no_warning_as_error --allow_running_as_root \
    --riscv_toolchain_root=${RISCV_ROOT_PATH} \
    --build_micro_benchmarks \
    --skip_submodule_sync \
    --use_mimalloc \
    --${2} \
    --skip_tests \
    ${EXTERN_ARGS}

pushd ${BUILD_DIR}/${3}
    make install
popd
