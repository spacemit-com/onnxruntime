# Copyright (c) 2023 SpacemiT. All rights reserved.
set(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_SYSTEM_PROCESSOR riscv64)

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(riscv)")
    message(STATUS "HOST SYSTEM ${CMAKE_HOST_SYSTEM_PROCESSOR}")
else()
    list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES RISCV_TOOLCHAIN_ROOT)
    set(CMAKE_C_COMPILER "${RISCV_TOOLCHAIN_ROOT}/bin/riscv64-unknown-linux-musl-clang")
    set(CMAKE_ASM_COMPILER "${RISCV_TOOLCHAIN_ROOT}/bin/riscv64-unknown-linux-musl-clang")
    set(CMAKE_CXX_COMPILER "${RISCV_TOOLCHAIN_ROOT}/bin/riscv64-unknown-linux-musl-clang++")
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

set(CMAKE_C_FLAGS "--target=riscv64 -march=rv64gcv_zfh_zba -latomic -mabi=lp64d ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "--target=riscv64 -march=rv64gcv_zfh_zba -latomic -mabi=lp64d ${CXX_FLAGS}")

add_definitions(-D__fp16=_Float16)

# 设置栈大小为16MB (16777216字节)
set(STACK_SIZE_BYTES 16777216)

# 在C/CXX编译标志中添加栈大小设置
set(CMAKE_C_FLAGS "-mcpu=spacemit-x60 -march=rv64gcv_zfh_zba -mabi=lp64d -fuse-ld=lld -Wl,-z,stack-size=${STACK_SIZE_BYTES} ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "-mcpu=spacemit-x60 -march=rv64gcv_zfh_zba -mabi=lp64d -fuse-ld=lld -stdlib=libc++ -static-libstdc++ -Wl,--push-state,-Bstatic -lc++ -lc++abi -Wl,--pop-state -Wl,-z,stack-size=${STACK_SIZE_BYTES} ${CXX_FLAGS}")

# 确保链接器标志中也包含栈大小设置
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libc++ -static-libgcc -static-libstdc++ -Wl,--push-state,-Bstatic -lgcc -lc++ -lc++abi -Wl,--pop-state -Wl,-z,stack-size=${STACK_SIZE_BYTES}")

# 为可执行文件和共享库明确设置栈大小
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,stack-size=${STACK_SIZE_BYTES}")
set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,-z,stack-size=${STACK_SIZE_BYTES}")

add_definitions(-D__fp16=_Float16)

# 调试信息：打印栈大小设置
message(STATUS "Stack size set to: ${STACK_SIZE_BYTES} bytes (16MB)")
message(STATUS "CMAKE_EXE_LINKER_FLAGS: ${CMAKE_EXE_LINKER_FLAGS}")
message(STATUS "CMAKE_C_FLAGS: ${CMAKE_C_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
