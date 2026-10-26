#!/bin/bash
# bash build_riscv64.spacemit.sh <onnxruntime_src_dir> <arch> <config> [os] [cc] [python_dir]
# arch:   x86 | rv64
# config: Release | RelWithDebInfo | Debug
# os:     Linux (default) | OHOS
# cc:     gcc (default) | clang
# python_dir: optional target Python prefix for Linux rv64 cross compilation
# before run this script, please ensure that RISCV_ROOT_PATH is set to the path of riscv toolchain
# https://archive.spacemit.com/toolchain/spacemit-toolchain-linux-glibc-x86_64-v1.1.2.tar.xz

ORT_SRC=${1}
ARCH_ARG=${2}
CONFIG=${3}
OS=${4:-Linux}
CC=${5:-gcc}
PYTHON_DIR=${6:-}

if [[ "$OS" != "Linux" && "$OS" != "OHOS" ]]; then
    echo "Error: os must be Linux or OHOS" >&2; exit 1
fi
if [[ "$CC" != "gcc" && "$CC" != "clang" ]]; then
    echo "Error: cc must be gcc or clang" >&2; exit 1
fi

# Detect host arch for native-build check
HOST_ARCH=$(uname -m)
MATCH_ARCH=""
[ "$HOST_ARCH" = "x86_64" ] && MATCH_ARCH="x86"
[ "$HOST_ARCH" = "riscv64" ] && MATCH_ARCH="rv64"

# Build directory: append -clang suffix when using clang
if [ "$CC" = "clang" ]; then
    BUILD_DIR=${ORT_SRC}/build/${OS}/${ARCH_ARG}-clang
else
    BUILD_DIR=${ORT_SRC}/build/${OS}/${ARCH_ARG}
fi

# Micro-benchmarks are not part of the shipping artifact and, on 1.31.1, the
# benchmark target does not link against the shared library, so the build fails with
# undefined references to ort_env / gtest internals. Keep them off for RISC-V builds.
BUILD_ARGS=""
EXTERN_ARGS=""
PYTHON_ARGS=()

# Native build: enable python bindings; Linux+gcc native rv64 skips micro benchmarks
if [ "$ARCH_ARG" = "$MATCH_ARCH" ]; then
    echo "BUILD ORT With Python"
    EXTERN_ARGS="--enable_pybind --build_wheel"
    if [ "$ARCH_ARG" = "rv64" ]; then
        BUILD_ARGS="--skip_onnx_tests"
    fi
elif [ "$ARCH_ARG" = "rv64" ] && [ -n "$PYTHON_DIR" ]; then
    if [ "$OS" != "Linux" ]; then
        echo "Error: Python wheel cross compilation requires Linux" >&2; exit 1
    fi
    echo "BUILD ORT With Python (cross compilation)"
    EXTERN_ARGS="--enable_pybind --build_wheel"
    PYTHON_ARGS=(--riscv_python_root "$PYTHON_DIR")
fi

# Toolchain flags: always except Linux+gcc cross-compiling non-rv64
if [ "$ARCH_ARG" = "rv64" ]; then
    BUILD_ARGS="${BUILD_ARGS} --${ARCH_ARG} --riscv_toolchain_root=${RISCV_ROOT_PATH}"
fi

# OS-specific flags
if [ "$OS" = "OHOS" ]; then
    BUILD_ARGS="${BUILD_ARGS} --build_ohos"
fi

# CC-specific flags
if [ "$CC" = "clang" ]; then
    BUILD_ARGS="${BUILD_ARGS} --riscv_use_clang"
fi

EXTERN_ARGS="${EXTERN_ARGS} \
    --cmake_extra_defines \
    onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS=ON \
    CMAKE_INSTALL_PREFIX=installed"

echo "BUILD_ARGS: ${BUILD_ARGS}"
echo "EXTERN_ARGS: ${EXTERN_ARGS}"

python3 ${ORT_SRC}/tools/ci_build/build.py --build_dir ${BUILD_DIR} --config ${CONFIG} \
    --update --build --build_shared_lib --parallel 20 \
    --compile_no_warning_as_error --allow_running_as_root \
    ${BUILD_ARGS} \
    --skip_submodule_sync \
    --use_mimalloc \
    --skip_tests \
    "${PYTHON_ARGS[@]}" ${EXTERN_ARGS}

pushd ${BUILD_DIR}/${CONFIG}
    make install
popd
