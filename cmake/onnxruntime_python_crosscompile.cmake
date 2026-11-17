# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

if(NOT CMAKE_CROSSCOMPILING OR NOT CMAKE_SYSTEM_NAME STREQUAL "Linux"
   OR NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "riscv64")
  message(FATAL_ERROR "onnxruntime_RISCV_PYTHON_ROOT requires a Linux RISC-V cross build")
endif()
if(onnxruntime_REQUIRE_PYTHON_EMBED_LIB)
  message(FATAL_ERROR "Embedding target Python is not supported for RISC-V cross builds")
endif()

# Find only the host interpreter. A combined Interpreter/Development lookup
# would require matching host/target versions and, with CMP0190, an emulator.
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
    "import json, sys; sys.path.insert(0, sys.argv[1]); from riscv_python import load_target_python, load_target_numpy; target = load_target_python(sys.argv[2]); target['numpy'] = load_target_numpy(target); print(json.dumps(target))"
    "${PROJECT_SOURCE_DIR}/../tools/ci_build" "${onnxruntime_RISCV_PYTHON_ROOT}"
  OUTPUT_VARIABLE _target_python_json
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
string(JSON Python_INCLUDE_DIRS GET "${_target_python_json}" include_dir)
string(JSON _target_python_version GET "${_target_python_json}" version)

# Linux extension modules resolve Python symbols from the importing interpreter.
# Do not link libpython or include the host interpreter's development headers.
add_library(Python::Module INTERFACE IMPORTED)
set_target_properties(Python::Module PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${Python_INCLUDE_DIRS}")

# Read the target wheel's headers/metadata without importing its RISC-V binary.
string(JSON Python_NumPy_INCLUDE_DIRS GET "${_target_python_json}" numpy include_dir)
string(JSON Python_NumPy_VERSION GET "${_target_python_json}" numpy version)
add_library(Python::NumPy INTERFACE IMPORTED)
set_target_properties(Python::NumPy PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${Python_NumPy_INCLUDE_DIRS}"
  INTERFACE_LINK_LIBRARIES Python::Module)
message(STATUS "Target Python ${_target_python_version}: ${Python_INCLUDE_DIRS}")
message(STATUS "Target NumPy ${Python_NumPy_VERSION}: ${Python_NumPy_INCLUDE_DIRS}")
