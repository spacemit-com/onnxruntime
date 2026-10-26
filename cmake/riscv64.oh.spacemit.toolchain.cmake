set(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_SYSTEM_PROCESSOR riscv64)
set(CMAKE_SYSTEM_VERSION 1)

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(riscv)")
    message(STATUS "HOST SYSTEM ${CMAKE_HOST_SYSTEM_PROCESSOR}")
else()
    list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES RISCV_TOOLCHAIN_ROOT)
    set(CMAKE_C_COMPILER "${RISCV_TOOLCHAIN_ROOT}/bin/clang")
    set(CMAKE_ASM_COMPILER "${RISCV_TOOLCHAIN_ROOT}/bin/clang")
    set(CMAKE_CXX_COMPILER "${RISCV_TOOLCHAIN_ROOT}/bin/clang++")
    set(CMAKE_STRIP ${RISCV_TOOLCHAIN_ROOT}/bin/llvm-strip)
    set(CMAKE_FIND_ROOT_PATH ${RISCV_TOOLCHAIN_ROOT})
    set(CMAKE_SYSROOT "${RISCV_TOOLCHAIN_ROOT}/sysroot")
    set(CMAKE_INCLUDE_PATH "${RISCV_TOOLCHAIN_ROOT}/sysroot/usr/include/")
    set(CMAKE_LIBRARY_PATH "${RISCV_TOOLCHAIN_ROOT}/sysroot/usr/lib/")
    set(CMAKE_PROGRAM_PATH "${RISCV_TOOLCHAIN_ROOT}/sysroot/usr/bin/")
    set(CMAKE_CROSSCOMPILING TRUE)
endif()

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

add_definitions(-D__OHOS__)

# flatbuffers' base.h auto-enables FLATBUFFERS_LOCALE_INDEPENDENT on OHOS
# (it reports _XOPEN_VERSION >= 700), which makes util.h call the locale-aware
# parsers strtoull_l/strtoll_l/strtod_l/strtof_l. The OHOS musl sysroot only
# declares strtof_l/strtod_l/strtold_l -- strtoull_l and strtoll_l do not exist
# in musl under any feature macro -- so the build fails with "use of undeclared
# identifier 'strtoull_l'". Force locale-independent mode off so flatbuffers uses
# the portable strtoull/strtoll path instead. Works on both musl and glibc OHOS
# sysroots; ORT does not switch the C locale at runtime.
add_definitions(-DFLATBUFFERS_LOCALE_INDEPENDENT=0)

set(STACK_SIZE_BYTES 16777216)

set(RISCV_MARCH_FLAGS "-march=rv64gcv_zfh_zvfh_zba_zicbop_zihintpause_xsmtvdotii")

set(CMAKE_C_FLAGS "${RISCV_MARCH_FLAGS} -mabi=lp64d -Wno-unused-command-line-argument -fuse-ld=lld -Wl,-z,stack-size=${STACK_SIZE_BYTES} ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${RISCV_MARCH_FLAGS} -mabi=lp64d -Wno-unused-command-line-argument -fuse-ld=lld -stdlib=libc++ -static-libstdc++ -Wl,--push-state,-Bstatic -lc++ -lc++abi -Wl,--pop-state -Wl,-z,stack-size=${STACK_SIZE_BYTES} ${CXX_FLAGS}")

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libc++ -static-libgcc -static-libstdc++ -Wl,--push-state,-Bstatic -lgcc -lc++ -lc++abi -Wl,--pop-state -lm -Wl,-z,stack-size=${STACK_SIZE_BYTES}")

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -latomic -lm -mabi=lp64d -Wl,-z,stack-size=${STACK_SIZE_BYTES}")
set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,-z,stack-size=${STACK_SIZE_BYTES}")
