set(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_SYSTEM_PROCESSOR riscv64)
set(CMAKE_SYSTEM_VERSION 1)

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(riscv)")
message(STATUS "HOST SYSTEM ${CMAKE_HOST_SYSTEM_PROCESSOR}")
else()
set(GNU_MACHINE riscv64-unknown-linux-gnu CACHE STRING "GNU compiler triple")
if(DEFINED ENV{RISCV_ROOT_PATH})
    file(TO_CMAKE_PATH $ENV{RISCV_ROOT_PATH} RISCV_ROOT_PATH)
else()
    message(FATAL_ERROR "RISCV_ROOT_PATH env must be defined")
endif()

set(RISCV_ROOT_PATH ${RISCV_ROOT_PATH} CACHE STRING "root path to riscv toolchain")
set(CMAKE_C_COMPILER ${RISCV_ROOT_PATH}/bin/riscv64-unknown-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER ${RISCV_ROOT_PATH}/bin/riscv64-unknown-linux-gnu-g++)
set(CMAKE_STRIP ${RISCV_ROOT_PATH}/bin/riscv64-unknown-linux-gnu-strip)
set(CMAKE_FIND_ROOT_PATH "${RISCV_ROOT_PATH}/sysroot")
set(CMAKE_SYSROOT "${RISCV_ROOT_PATH}/sysroot")
endif()

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(riscv)")
    if(CMAKE_C_COMPILER)
        set(RISCV_GCC_COMPILER ${CMAKE_C_COMPILER})
    else()
        find_program(RISCV_GCC_COMPILER NAMES gcc riscv64-unknown-linux-gnu-gcc REQUIRED)
    endif()
else()
    set(RISCV_GCC_COMPILER ${CMAKE_C_COMPILER})
endif()

execute_process(
    COMMAND ${RISCV_GCC_COMPILER} -dumpfullversion -dumpversion
    OUTPUT_VARIABLE RISCV_GCC_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)

set(RISCV_MARCH_FLAGS "-march=rv64gcv_zfh_zvfh_zba_zicbop_zihintpause")
if(RISCV_GCC_VERSION VERSION_GREATER_EQUAL 15)
    set(RISCV_MARCH_FLAGS "${RISCV_MARCH_FLAGS}_xsmtvdotii")
endif()

set(CMAKE_C_FLAGS "${RISCV_MARCH_FLAGS} -mabi=lp64d -fno-tree-vectorize -fno-tree-loop-vectorize ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${RISCV_MARCH_FLAGS} -mabi=lp64d -fno-tree-vectorize -fno-tree-loop-vectorize ${CMAKE_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -latomic -lrt -lpthread")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --sysroot=${CMAKE_SYSROOT}")
