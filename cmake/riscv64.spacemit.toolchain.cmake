set(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_SYSTEM_PROCESSOR riscv64)
set(CMAKE_SYSTEM_VERSION 1)

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(riscv)")
    message(STATUS "HOST SYSTEM ${CMAKE_HOST_SYSTEM_PROCESSOR}")
else()
    list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES RISCV_TOOLCHAIN_ROOT)
    set(GNU_MACHINE riscv64-unknown-linux-gnu CACHE STRING "GNU compiler triple")
    set(CMAKE_C_COMPILER ${RISCV_TOOLCHAIN_ROOT}/bin/riscv64-unknown-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER ${RISCV_TOOLCHAIN_ROOT}/bin/riscv64-unknown-linux-gnu-g++)
    set(CMAKE_STRIP ${RISCV_TOOLCHAIN_ROOT}/bin/riscv64-unknown-linux-gnu-strip)
    set(CMAKE_FIND_ROOT_PATH "${RISCV_TOOLCHAIN_ROOT}/sysroot")
    set(CMAKE_SYSROOT "${RISCV_TOOLCHAIN_ROOT}/sysroot")

    # iconv is built into glibc; no separate libiconv needed
    set(Iconv_IS_BUILT_IN TRUE)

    # Cross-compilation: the regex feature checks need to run target binaries, which
    # cannot execute on the host. Prefer std::regex (always available with C++17) so that
    # google_benchmark does not fall back to <gnuregex.h>, which glibc no longer ships.
    set(HAVE_STD_REGEX 1 CACHE INTERNAL "")
    set(HAVE_GNU_POSIX_REGEX 0 CACHE INTERNAL "")
    set(HAVE_POSIX_REGEX 0 CACHE INTERNAL "")
endif()

set(CMAKE_CROSSCOMPILING TRUE)

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
