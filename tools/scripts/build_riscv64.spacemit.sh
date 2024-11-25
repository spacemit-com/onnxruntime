#!/bin/bash
# bash build_riscv64.spacemit.sh <onnxruntime_src_dir> <config> <arch>

export http_proxy=
export https_proxy=

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

EXTERN_ARGS="${EXTERN_ARGS} --cmake_extra_defines \
    onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS=ON \
    CMAKE_INSTALL_PREFIX=${1}/build/Linux/${2}/${3}/installed"

# For Clang
# EXTERN_ARGS="${EXTERN_ARGS} CMAKE_CXX_COMPILER=clang++ CMAKE_C_COMPILER=clang"

python3 ${1}/tools/ci_build/build.py --build_dir ${1}/build/Linux/${2} --config ${3} \
    --update --build --build_shared_lib --parallel 10 \
    --compile_no_warning_as_error --allow_running_as_root \
    --riscv_toolchain_root=${RISCV_ROOT_PATH} \
    --riscv_ime_spec=spacemit-ime1 \
    --build_micro_benchmarks \
    --skip_submodule_sync \
    --${2} \
    --skip_tests \
    ${EXTERN_ARGS}

pushd ${1}/build/Linux/${2}/${3}
    make install
popd
