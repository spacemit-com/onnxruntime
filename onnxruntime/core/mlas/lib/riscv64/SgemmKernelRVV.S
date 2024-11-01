// Copyright (c) 2023 SpacemiT. All rights reserved.
// Licensed under the MIT License.

.align 4

// save vtype vl to t0-t1, save v0-v31, s0-s11
.macro preserve_caller_vec
    csrr x5, vtype
    csrr x6, vl
    vsetvli x7, x0, e8, m8
    sub sp, sp, x7
    vse8.v v0, (sp)
    sub sp, sp, x7
    vse8.v v8, (sp)
    sub sp, sp, x7
    vse8.v v16, (sp)
    sub sp, sp, x7
    vse8.v v24, (sp)
    addi sp, sp, -8
    sd x8, (sp)
    addi sp, sp, -8
    sd x9, (sp)
    addi sp, sp, -8
    sd x18, (sp)
    addi sp, sp, -8
    sd x19, (sp)
    addi sp, sp, -8
    sd x20, (sp)
    addi sp, sp, -8
    sd x21, (sp)
    addi sp, sp, -8
    sd x22, (sp)
    addi sp, sp, -8
    sd x23, (sp)
    addi sp, sp, -8
    sd x24, (sp)
    addi sp, sp, -8
    sd x25, (sp)
    addi sp, sp, -8
    sd x26, (sp)
    addi sp, sp, -8
    sd x27, (sp)
    addi sp, sp, -8
    sd x28, (sp)
    addi sp, sp, -8
    sd x29, (sp)
    addi sp, sp, -8
    sd x30, (sp)
    addi sp, sp, -8
    sd x31, (sp)
.endm

// recover s11-s0, v31-v0, vtype vl from t0-t1
.macro restore_caller_vec
    ld x31, (sp)
    addi sp, sp, 8
    ld x30, (sp)
    addi sp, sp, 8
    ld x29, (sp)
    addi sp, sp, 8
    ld x28, (sp)
    addi sp, sp, 8
    ld x27, (sp)
    addi sp, sp, 8
    ld x26, (sp)
    addi sp, sp, 8
    ld x25, (sp)
    addi sp, sp, 8
    ld x24, (sp)
    addi sp, sp, 8
    ld x23, (sp)
    addi sp, sp, 8
    ld x22, (sp)
    addi sp, sp, 8
    ld x21, (sp)
    addi sp, sp, 8
    ld x20, (sp)
    addi sp, sp, 8
    ld x19, (sp)
    addi sp, sp, 8
    ld x18, (sp)
    addi sp, sp, 8
    ld x9, (sp)
    addi sp, sp, 8
    ld x8, (sp)
    addi sp, sp, 8
    vsetvli x7, x0, e8, m8
    vle8.v v24, (sp)
    add sp, sp, x7
    vle8.v v16, (sp)
    add sp, sp, x7
    vle8.v v8, (sp)
    add sp, sp, x7
    vle8.v v0, (sp)
    add sp, sp, x7
    vsetvl x7, x6, x5
.endm

//
// 8 * 16 single precise floating point matric multiplication for X60 256bit rvv1.0
//
//    --              --      --                 --     --            --         --                  --
//    | A00 - - - - - - |      |  B0  B1  ...  B15 |     | alpha |         | A0B0 A0B1 .. A0B15 |
//    | A10 - - - - - - |      |  .   .   ...   .  |     | alpha |         | A1B0 A1B1 .. A1B15 |
//    | A20 - - - - - - |      |  .   .   ...   .  |     | alpha |         | A2B0 A2B1 .. A2B15 |
//    | A30 - - - - - - |      |  .   .   ...   .  |     | alpha |         | A3B0 A3B1 .. A3B15 |
//    | A40 - - - - - - |      |  .   .   ...   .  |     | alpha |         | A4B0 A4B1 .. A4B15 |
//    | A50 - - - - - - |      |  .   .   ...   .  |     | alpha |         | A5B0 A5B1 .. A5B15 |
//    | A60 - - - - - - |      |  .   .   ...   .  |     | alpha |         | A6B0 A6B1 .. A6B15 |
//    | A70 - - - - - - |  x   |  .   .   ...   .  |  x  | alpha |    +=   | A7B0 A7B1 .. A7B15 |
//    --              --      --                 --     --            --         --                  --
//      A 8 x k                 B k x 16                  alpha              C 8 x 16
//
// interface args:
//      x10/a0 arg0 float* A, ro
//      x11/a1 arg1 float* B, r & w
//      x12/a2 arg2 float* C, r & w
//      x13/a3 arg3 CountK, ro
//      x14/a4 arg4 CountN, r & w
//      x15/a5 arg5 lda, ro
//      x16/a6 arg6 ldc, ro
//      f10/fa0 arg7 alpha ro
//
//
// GPR Definition
//
// GP arg reg
// a0-a7 = x10-x17, caller saved
#define     A          x10
#define     B          x11
#define     C          x12
#define     K          x13
#define     N          x14
#define     lda        x15
#define     ldc        x16
// #define     alpha      f10
//
// GP tem reg
// t0 = x5 = vtype, t1 = x6 = vl, t2 = x7 = T
#define     T          x7 // tmp
// reserved t3-t6 = x28-x31
#define     B0         x28
#define     B1         x29
#define     B2         x30
#define     B3         x31
//
// GP saved reg
// s0-s11 = x8 x9, x18-x27
#define     A0         x8
#define     A1         x9
#define     A2         x18
#define     A3         x19
#define     A4         x20
#define     A5         x21
#define     A6         x22
#define     A7         x23
#define     K_BK       x24
#define     L          x25 // K / unloop
#define     NR         x26
#define     C_BK       x27
//
// FPR Definition
//
// FP arg reg
// fa0-fa7 = f10-f17, caller saved
// f10-f17 = input A data double buffer
//
// FP tem reg
// ft0-ft11 = f0-f7, f28-f31, caller saved
// f0-f7 = input A data
#define     alpha      f8
//
// FP saved reg
// fs0-fs11 = f8-f9, f18-f27, reserved
//
//
// VPR Definition
// v0 - v15 input B data double buffer and for load C
// v16 - v31 for accmulate
//
//
// return value:
// return: mr


.macro KERNEL_8x4x16_I
    flw f0, (A0)
    addi A0, A0, 4
    flw f1, (A1)
    addi A1, A1, 4
    flw f2, (A2)
    addi A2, A2, 4
    flw f3, (A3)
    addi A3, A3, 4
    flw f4, (A4)
    addi A4, A4, 4
    flw f5, (A5)
    addi A5, A5, 4
    flw f6, (A6)
    addi A6, A6, 4
    flw f7, (A7)
    addi A7, A7, 4

    vle32.v v0, (B0)
    addi B0, B0, 256
    vle32.v v2, (B1)
    addi B1, B1, 256
    vle32.v v4, (B2)
    addi B2, B2, 256
    vle32.v v6, (B3)
    addi B3, B3, 256

    vfmacc.vf v16, f0, v0
    flw f10, (A0)
    vfmacc.vf v18, f1, v0
    flw f11, (A1)
    vfmacc.vf v20, f2, v0
    flw f12, (A2)
    vfmacc.vf v22, f3, v0
    flw f13, (A3)
    vfmacc.vf v24, f4, v0
    flw f14, (A4)
    vfmacc.vf v26, f5, v0
    flw f15, (A5)
    vfmacc.vf v28, f6, v0
    flw f16, (A6)
    vfmacc.vf v30, f7, v0
    flw f17, (A7)

    vfmacc.vf v16, f10, v2
    flw f0, 4(A0)
    vfmacc.vf v18, f11, v2
    flw f1, 4(A1)
    vfmacc.vf v20, f12, v2
    flw f2, 4(A2)
    vfmacc.vf v22, f13, v2
    flw f3, 4(A3)
    vfmacc.vf v24, f14, v2
    flw f4, 4(A4)
    vfmacc.vf v26, f15, v2
    flw f5, 4(A5)
    vfmacc.vf v28, f16, v2
    flw f6, 4(A6)
    vfmacc.vf v30, f17, v2
    flw f7, 4(A7)

    vle32.v v8, (B0)
    addi B0, B0, 256
    vle32.v v10, (B1)
    addi B1, B1, 256
    vle32.v v12, (B2)
    addi B2, B2, 256
    vle32.v v14, (B3)
    addi B3, B3, 256

    vfmacc.vf v16, f0, v4
    flw f10, 8(A0)
    vfmacc.vf v18, f1, v4
    flw f11, 8(A1)
    vfmacc.vf v20, f2, v4
    flw f12, 8(A2)
    vfmacc.vf v22, f3, v4
    flw f13, 8(A3)
    vfmacc.vf v24, f4, v4
    flw f14, 8(A4)
    vfmacc.vf v26, f5, v4
    flw f15, 8(A5)
    vfmacc.vf v28, f6, v4
    flw f16, 8(A6)
    vfmacc.vf v30, f7, v4
    flw f17, 8(A7)

    vfmacc.vf v16, f10, v6
    flw f0, 12(A0)
    addi A0, A0, 16
    vfmacc.vf v18, f11, v6
    flw f1, 12(A1)
    addi A1, A1, 16
    vfmacc.vf v20, f12, v6
    flw f2, 12(A2)
    addi A2, A2, 16
    vfmacc.vf v22, f13, v6
    flw f3, 12(A3)
    addi A3, A3, 16
    vfmacc.vf v24, f14, v6
    flw f4, 12(A4)
    addi A4, A4, 16
    vfmacc.vf v26, f15, v6
    flw f5, 12(A5)
    addi A5, A5, 16
    vfmacc.vf v28, f16, v6
    flw f6, 12(A6)
    addi A6, A6, 16
    vfmacc.vf v30, f17, v6
    flw f7, 12(A7)
    addi A7, A7, 16

.endm

.macro KERNEL_8x4x16_M1
    vfmacc.vf v16, f0, v8
    flw f10, (A0)
    vfmacc.vf v18, f1, v8
    flw f11, (A1)
    vfmacc.vf v20, f2, v8
    flw f12, (A2)
    vfmacc.vf v22, f3, v8
    flw f13, (A3)
    vfmacc.vf v24, f4, v8
    flw f14, (A4)
    vfmacc.vf v26, f5, v8
    flw f15, (A5)
    vfmacc.vf v28, f6, v8
    flw f16, (A6)
    vfmacc.vf v30, f7, v8
    flw f17, (A7)

    vfmacc.vf v16, f10, v10
    flw f0, 4(A0)
    flw f1, 4(A1)
    vfmacc.vf v18, f11, v10
    flw f2, 4(A2)
    flw f3, 4(A3)
    vfmacc.vf v20, f12, v10
    flw f4, 4(A4)
    flw f5, 4(A5)
    vfmacc.vf v22, f13, v10
    flw f6, 4(A6)
    flw f7, 4(A7)
    vfmacc.vf v24, f14, v10
    vfmacc.vf v26, f15, v10
    vfmacc.vf v28, f16, v10
    vfmacc.vf v30, f17, v10

    vle32.v v0, (B0)
    addi B0, B0, 256
    vle32.v v2, (B1)
    addi B1, B1, 256
    vle32.v v4, (B2)
    addi B2, B2, 256
    vle32.v v6, (B3)
    addi B3, B3, 256

    vfmacc.vf v16, f0, v12
    vfmacc.vf v18, f1, v12
    vfmacc.vf v20, f2, v12
    vfmacc.vf v22, f3, v12
    vfmacc.vf v24, f4, v12
    flw f10, 8(A0)
    flw f11, 8(A1)
    vfmacc.vf v26, f5, v12
    flw f12, 8(A2)
    flw f13, 8(A3)
    vfmacc.vf v28, f6, v12
    flw f14, 8(A4)
    flw f15, 8(A5)
    vfmacc.vf v30, f7, v12
    flw f16, 8(A6)
    flw f17, 8(A7)

    vfmacc.vf v16, f10, v14
    flw f0, 12(A0)
    addi A0, A0, 16
    vfmacc.vf v18, f11, v14
    flw f1, 12(A1)
    addi A1, A1, 16
    vfmacc.vf v20, f12, v14
    flw f2, 12(A2)
    addi A2, A2, 16
    vfmacc.vf v22, f13, v14
    flw f3, 12(A3)
    addi A3, A3, 16
    vfmacc.vf v24, f14, v14
    flw f4, 12(A4)
    addi A4, A4, 16
    vfmacc.vf v26, f15, v14
    flw f5, 12(A5)
    addi A5, A5, 16
    vfmacc.vf v28, f16, v14
    flw f6, 12(A6)
    addi A6, A6, 16
    vfmacc.vf v30, f17, v14
    flw f7, 12(A7)
    addi A7, A7, 16

.endm

.macro KERNEL_8x4x16_M2
    vfmacc.vf v16, f0, v0
    flw f10, (A0)
    vfmacc.vf v18, f1, v0
    flw f11, (A1)
    vfmacc.vf v20, f2, v0
    flw f12, (A2)
    vfmacc.vf v22, f3, v0
    flw f13, (A3)
    vfmacc.vf v24, f4, v0
    flw f14, (A4)
    vfmacc.vf v26, f5, v0
    flw f15, (A5)
    vfmacc.vf v28, f6, v0
    flw f16, (A6)
    vfmacc.vf v30, f7, v0
    flw f17, (A7)

    vfmacc.vf v16, f10, v2
    flw f0, 4(A0)
    flw f1, 4(A1)
    vfmacc.vf v18, f11, v2
    flw f2, 4(A2)
    flw f3, 4(A3)
    vfmacc.vf v20, f12, v2
    flw f4, 4(A4)
    flw f5, 4(A5)
    vfmacc.vf v22, f13, v2
    flw f6, 4(A6)
    flw f7, 4(A7)
    vfmacc.vf v24, f14, v2
    vfmacc.vf v26, f15, v2
    vfmacc.vf v28, f16, v2
    vfmacc.vf v30, f17, v2

    vle32.v v8, (B0)
    addi B0, B0, 256
    vle32.v v10, (B1)
    addi B1, B1, 256
    vle32.v v12, (B2)
    addi B2, B2, 256
    vle32.v v14, (B3)
    addi B3, B3, 256

    vfmacc.vf v16, f0, v4
    vfmacc.vf v18, f1, v4
    vfmacc.vf v20, f2, v4
    vfmacc.vf v22, f3, v4
    vfmacc.vf v24, f4, v4
    flw f10, 8(A0)
    flw f11, 8(A1)
    vfmacc.vf v26, f5, v4
    flw f12, 8(A2)
    flw f13, 8(A3)
    vfmacc.vf v28, f6, v4
    flw f14, 8(A4)
    flw f15, 8(A5)
    vfmacc.vf v30, f7, v4
    flw f16, 8(A6)
    flw f17, 8(A7)

    vfmacc.vf v16, f10, v6
    flw f0, 12(A0)
    addi A0, A0, 16
    vfmacc.vf v18, f11, v6
    flw f1, 12(A1)
    addi A1, A1, 16
    vfmacc.vf v20, f12, v6
    flw f2, 12(A2)
    addi A2, A2, 16
    vfmacc.vf v22, f13, v6
    flw f3, 12(A3)
    addi A3, A3, 16
    vfmacc.vf v24, f14, v6
    flw f4, 12(A4)
    addi A4, A4, 16
    vfmacc.vf v26, f15, v6
    flw f5, 12(A5)
    addi A5, A5, 16
    vfmacc.vf v28, f16, v6
    flw f6, 12(A6)
    addi A6, A6, 16
    vfmacc.vf v30, f17, v6
    flw f7, 12(A7)
    addi A7, A7, 16

.endm


.macro KERNEL_8x4x16_E

    vfmacc.vf v16, f0, v8
    flw f10, (A0)
    vfmacc.vf v18, f1, v8
    flw f11, (A1)
    vfmacc.vf v20, f2, v8
    flw f12, (A2)
    vfmacc.vf v22, f3, v8
    flw f13, (A3)
    vfmacc.vf v24, f4, v8
    flw f14, (A4)
    vfmacc.vf v26, f5, v8
    flw f15, (A5)
    vfmacc.vf v28, f6, v8
    flw f16, (A6)
    vfmacc.vf v30, f7, v8
    flw f17, (A7)

    vfmacc.vf v16, f10, v10
    flw f0, 4(A0)
    vfmacc.vf v18, f11, v10
    flw f1, 4(A1)
    vfmacc.vf v20, f12, v10
    flw f2, 4(A2)
    vfmacc.vf v22, f13, v10
    flw f3, 4(A3)
    vfmacc.vf v24, f14, v10
    flw f4, 4(A4)
    vfmacc.vf v26, f15, v10
    flw f5, 4(A5)
    vfmacc.vf v28, f16, v10
    flw f6, 4(A6)
    vfmacc.vf v30, f17, v10
    flw f7, 4(A7)

    vfmacc.vf v16, f0, v12
    flw f10, 8(A0)
    addi A0, A0, 12
    vfmacc.vf v18, f1, v12
    flw f11, 8(A1)
    addi A1, A1, 12
    vfmacc.vf v20, f2, v12
    flw f12, 8(A2)
    addi A2, A2, 12
    vfmacc.vf v22, f3, v12
    flw f13, 8(A3)
    addi A3, A3, 12
    vfmacc.vf v24, f4, v12
    flw f14, 8(A4)
    addi A4, A4, 12
    vfmacc.vf v26, f5, v12
    flw f15, 8(A5)
    addi A5, A5, 12
    vfmacc.vf v28, f6, v12
    flw f16, 8(A6)
    addi A6, A6, 12
    vfmacc.vf v30, f7, v12
    flw f17, 8(A7)
    addi A7, A7, 12

    vfmacc.vf v16, f10, v14
    vfmacc.vf v18, f11, v14
    vfmacc.vf v20, f12, v14
    vfmacc.vf v22, f13, v14
    vfmacc.vf v24, f14, v14
    vfmacc.vf v26, f15, v14
    vfmacc.vf v28, f16, v14
    vfmacc.vf v30, f17, v14

.endm


#ifdef __APPLE__
.globl _MlasSgemmKernel_8x16
_MlasSgemmKernel_8x16:
#else
.globl MlasSgemmKernel_8x16
MlasSgemmKernel_8x16:
#endif
    preserve_caller_vec

    fmv.x.w T, f10
    fmv.w.x alpha, T

    beq N, x0, sgemm_kernel_quit // N == 0

    slli lda, lda, 2 // lda = lda * 4 byte
    slli ldc, ldc, 2 // ldc = ldc * 4 byte

    addi B0, B, 0

sgemm_kernel_8_x_nr_start:

    vsetvli NR, N, e32, m2
    sub N, N, NR // n = n - nr
    slli NR, NR, 2 // nr = nr * 4

    beq K, x0, sgemm_kernel_8_x_nr_end // K == 0

    vxor.vv v16, v16, v16
    vxor.vv v18, v18, v18
    vxor.vv v20, v20, v20
    vxor.vv v22, v22, v22
    vxor.vv v24, v24, v24
    vxor.vv v26, v26, v26
    vxor.vv v28, v28, v28
    vxor.vv v30, v30, v30

    add A0, A, x0
    add A1, A0, lda
    add A2, A1, lda
    add A3, A2, lda
    add A4, A3, lda
    add A5, A4, lda
    add A6, A5, lda
    add A7, A6, lda

    addi B1, B0, 64
    addi B2, B0, 128
    addi B3, B0, 192

sgemm_kernel_8_x_nr_start_8:

    srl L, K, 3 // L = K / 8

    sltiu T, L, 2 // T = L < 2 ? 1: 0
    blt x0, T, sgemm_kernel_8_x_nr_only_8 // branch if L < 2

    KERNEL_8x4x16_I
    KERNEL_8x4x16_M1

    addi L, L, -2
    beq L, x0, sgemm_kernel_8_x_nr_end_8

sgemm_kernel_8_x_nr_mid_8:

    KERNEL_8x4x16_M2
    KERNEL_8x4x16_M1

    addi L, L, -1
    bne L, x0, sgemm_kernel_8_x_nr_mid_8

sgemm_kernel_8_x_nr_end_8:

    KERNEL_8x4x16_M2
    KERNEL_8x4x16_E

    jal x0, sgemm_kernel_8_x_nr_sub

sgemm_kernel_8_x_nr_only_8:

    beq L, x0, sgemm_kernel_8_x_nr_sub // L == 0

    KERNEL_8x4x16_I
    KERNEL_8x4x16_E

sgemm_kernel_8_x_nr_sub:

    andi L, K, 7 // L = K % 8
    beq L, x0, sgemm_kernel_8_x_nr_save // branch if L == 0

sgemm_kernel_8_x_nr_one:

    flw f0, (A0)
    addi A0, A0, 4
    flw f1, (A1)
    addi A1, A1, 4
    flw f2, (A2)
    addi A2, A2, 4
    flw f3, (A3)
    addi A3, A3, 4
    flw f4, (A4)
    addi A4, A4, 4
    flw f5, (A5)
    addi A5, A5, 4
    flw f6, (A6)
    addi A6, A6, 4
    flw f7, (A7)
    addi A7, A7, 4

    vle32.v v0, (B0)
    addi B0, B0, 64

    vfmacc.vf v16, f0, v0
    vfmacc.vf v18, f1, v0
    vfmacc.vf v20, f2, v0
    vfmacc.vf v22, f3, v0
    vfmacc.vf v24, f4, v0
    vfmacc.vf v26, f5, v0
    vfmacc.vf v28, f6, v0
    vfmacc.vf v30, f7, v0

    addi L, L, -1
    bne L, x0, sgemm_kernel_8_x_nr_one  // branch if L > 0

sgemm_kernel_8_x_nr_save:

    vfmul.vf v16, v16, alpha
    vfmul.vf v18, v18, alpha
    vfmul.vf v20, v20, alpha
    vfmul.vf v22, v22, alpha
    vfmul.vf v24, v24, alpha
    vfmul.vf v26, v26, alpha
    vfmul.vf v28, v28, alpha
    vfmul.vf v30, v30, alpha

    vse32.v v16, (C)
    add C_BK, C, ldc

    vse32.v v18, (C_BK)
    add C_BK, C_BK, ldc

    vse32.v v20, (C_BK)
    add C_BK, C_BK, ldc

    vse32.v v22, (C_BK)
    add C_BK, C_BK, ldc

    vse32.v v24, (C_BK)
    add C_BK, C_BK, ldc

    vse32.v v26, (C_BK)
    add C_BK, C_BK, ldc

    vse32.v v28, (C_BK)
    add C_BK, C_BK, ldc

    vse32.v v30, (C_BK)

    add C, C, NR // C += nr * 4

sgemm_kernel_8_x_nr_end:

    bne N, x0, sgemm_kernel_8_x_nr_start  // branch if N > 0

sgemm_kernel_quit:

    restore_caller_vec
    addi x10, x0, 8
    ret
