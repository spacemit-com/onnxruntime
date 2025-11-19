#!/bin/bash
# Copyright (c) 2023 SpacemiT. All rights reserved.
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

if [ "${3}" = "RISCV64_SPACEMIT_IME1" ]; then
    BUILD_DIR=${BUILD_DIR}-ime1-clang
    IME_SPEC=spacemit-ime1
elif [ "${3}" = "RISCV64_SPACEMIT_IME2" ]; then
    BUILD_DIR=${BUILD_DIR}-ime2-clang
    IME_SPEC=spacemit-ime2
else
    echo "IME Spec ${3} is not supported"
    exit
fi


EXTERN_ARGS="${EXTERN_ARGS} \
    --cmake_extra_defines \
    onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS=ON \
    CMAKE_INSTALL_PREFIX=${BUILD_DIR}/${4}/installed \
    GERRIT_USER=${5}"

# For Clang
# EXTERN_ARGS="${EXTERN_ARGS} CMAKE_CXX_COMPILER=clang++ CMAKE_C_COMPILER=clang"

python3 ${1}/tools/ci_build/build.py --build_dir ${BUILD_DIR} --config ${4} \
    --update --build --build_shared_lib --parallel 10 \
    --compile_no_warning_as_error --allow_running_as_root \
    --riscv_toolchain_root=${RISCV_ROOT_PATH} \
    --riscv_ime_spec=${IME_SPEC} \
    --build_micro_benchmarks \
    --skip_submodule_sync \
    --use_mimalloc \
    --riscv_use_clang \
    --${2} \
    --skip_tests \
    ${EXTERN_ARGS}

pushd ${BUILD_DIR}/${4}
    make install
popd
