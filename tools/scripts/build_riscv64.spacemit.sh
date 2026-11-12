#!/bin/bash
set -euo pipefail

# Build and install an ORT SDK, including onnxruntime_perf_test.
# Usage: build_riscv64.spacemit.sh <src> <x86|rv64> <config> [Linux|OHOS] [gcc|clang] [python_dir]
# Linux defaults to GCC; OHOS uses its dedicated Clang toolchain.
# ORT_BUILD_ROOT overrides the configuration parent directory.
# RISCV_ROOT_PATH selects the target toolchain; PYTHON_EXECUTABLE selects the host Python.
# CMAKE_BUILD_PARALLEL_LEVEL controls parallelism. python_dir enables a Linux rv64 cross wheel
# with target Python/NumPy headers matching the host Python version and ABI.
if [[ $# -lt 3 || $# -gt 6 ]]; then
    echo "Usage: $0 <src> <x86|rv64> <config> [Linux|OHOS] [gcc|clang] [python_dir]" >&2
    exit 1
fi
ORT_SRC="$(cd "$1" && pwd)"
ARCH_ARG=$2
CONFIG=$3
TARGET_OS=${4:-Linux}
TARGET_CC=${5:-}
PYTHON_DIR=${6:-}
PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-python3}
case "${ARCH_ARG}" in
    x86|rv64) ;;
    *) echo "Architecture must be x86 or rv64." >&2; exit 1 ;;
esac
case "${CONFIG}" in
    Release|RelWithDebInfo|Debug|MinSizeRel) ;;
    *) echo "Invalid build configuration: ${CONFIG}" >&2; exit 1 ;;
esac
case "${TARGET_OS}" in
    Linux) TARGET_CC=${TARGET_CC:-gcc} ;;
    OHOS) TARGET_CC=${TARGET_CC:-clang} ;;
    *) echo "OS must be Linux or OHOS." >&2; exit 1 ;;
esac
case "${TARGET_CC}" in
    gcc|clang) ;;
    *) echo "Compiler must be gcc or clang." >&2; exit 1 ;;
esac
if [[ "${TARGET_OS}" == OHOS ]]; then
    if [[ "${ARCH_ARG}" != rv64 || "${TARGET_CC}" != clang ]]; then
        echo "OHOS builds require rv64 and the OHOS Clang toolchain." >&2
        exit 1
    fi
    if [[ -n "${PYTHON_DIR}" ]]; then
        echo "Python wheel cross compilation requires Linux." >&2
        exit 1
    fi
fi

COMPILER_SUFFIX=""
if [[ "${TARGET_OS}" == Linux && "${TARGET_CC}" == clang ]]; then
    COMPILER_SUFFIX=-clang
fi
BUILD_DIR="${ORT_BUILD_ROOT:-${ORT_SRC}/build/${TARGET_OS}/${ARCH_ARG}${COMPILER_SUFFIX}}"
mkdir -p "${BUILD_DIR}"
BUILD_DIR="$(cd "${BUILD_DIR}" && pwd)"
INSTALL_DIR="${BUILD_DIR}/${CONFIG}/installed"
BUILD_ARGS=(--build_dir "${BUILD_DIR}" --config "${CONFIG}" --update --build --build_shared_lib
    --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-$(getconf _NPROCESSORS_ONLN)}"
    --compile_no_warning_as_error --allow_running_as_root --skip_submodule_sync --use_mimalloc --skip_tests)
CMAKE_DEFINES=("CMAKE_INSTALL_PREFIX=${INSTALL_DIR}" onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS=OFF)
if [[ "${ARCH_ARG}" == rv64 ]]; then
    : "${RISCV_ROOT_PATH:?set RISCV_ROOT_PATH to the target toolchain root}"
    BUILD_ARGS+=(--rv64 --riscv_toolchain_root "${RISCV_ROOT_PATH}")
    if [[ "${TARGET_OS}" == OHOS ]]; then
        # --riscv_use_clang selects the Linux toolchain and must not accompany --build_ohos.
        BUILD_ARGS+=(--build_ohos)
    elif [[ "${TARGET_CC}" == clang ]]; then
        BUILD_ARGS+=(--riscv_use_clang)
    fi
elif [[ "${TARGET_CC}" == clang ]]; then
    CMAKE_DEFINES+=(CMAKE_C_COMPILER=clang CMAKE_CXX_COMPILER=clang++)
fi
HOST_ARCH=$(uname -m)
if [[ "${TARGET_OS}" == Linux ]]; then
    if [[ ( "${ARCH_ARG}" == x86 && "${HOST_ARCH}" == x86_64 ) ||
          ( "${ARCH_ARG}" == rv64 && "${HOST_ARCH}" == riscv64 ) ]]; then
        BUILD_ARGS+=(--enable_pybind --build_wheel)
    elif [[ "${ARCH_ARG}" == rv64 && -n "${PYTHON_DIR}" ]]; then
        PYTHON_DIR="$(cd "${PYTHON_DIR}" && pwd)"
        PYTHON_VERSION=$("${PYTHON_EXECUTABLE}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}{sys.abiflags}")')
        PYTHON_INCLUDE_DIR="${PYTHON_DIR}/include/python${PYTHON_VERSION}"
        NUMPY_INCLUDE_DIR="${PYTHON_DIR}/lib/python${PYTHON_VERSION}/site-packages/numpy/_core/include"
        if [[ ! -f "${NUMPY_INCLUDE_DIR}/numpy/arrayobject.h" ]]; then
            NUMPY_INCLUDE_DIR="${PYTHON_DIR}/lib/python${PYTHON_VERSION}/site-packages/numpy/core/include"
        fi
        if [[ ! -f "${PYTHON_INCLUDE_DIR}/Python.h" || ! -f "${PYTHON_INCLUDE_DIR}/pyconfig.h" ||
              ! -f "${NUMPY_INCLUDE_DIR}/numpy/arrayobject.h" ]]; then
            echo "python_dir must contain Python ${PYTHON_VERSION} and NumPy development headers matching the host Python." >&2
            exit 1
        fi
        BUILD_ARGS+=(--enable_pybind --build_wheel)
        CMAKE_DEFINES+=("Python_INCLUDE_DIR=${PYTHON_INCLUDE_DIR}" "Python_NumPy_INCLUDE_DIR=${NUMPY_INCLUDE_DIR}")
        # setuptools must tag the cross-built wheel for the target platform.
        export _PYTHON_HOST_PLATFORM=linux-riscv64
    fi
fi
"${PYTHON_EXECUTABLE}" "${ORT_SRC}/tools/ci_build/build.py" "${BUILD_ARGS[@]}" \
    --cmake_extra_defines "${CMAKE_DEFINES[@]}"
cmake --install "${BUILD_DIR}/${CONFIG}" --config "${CONFIG}"
mkdir -p "${INSTALL_DIR}/bin"
install -m 755 "${BUILD_DIR}/${CONFIG}/onnxruntime_perf_test" "${INSTALL_DIR}/bin/"
