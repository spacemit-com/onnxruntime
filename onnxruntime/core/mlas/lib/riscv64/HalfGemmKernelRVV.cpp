// Copyright (c) 2023 SpacemiT. All rights reserved.
// Licensed under the MIT License.

//
// Stack frame layout for the half gemm kernel.
//

/*++

Routine Description:

    This routine is an inner kernel to compute 8 rows of GEMM

Arguments:

    CountM - the number of rows for matrix A and matrix C.
            only process 6 rows

    CountN - the number of columns from matrix B and matrix C

    CountK - the number of columns from matrix A and the
            number of rows from matrix B.

    C      - the address of matrix C.

    ldc    - the first dimension of matrix C.

    Bias   - the address of the Bias vector (optional)

    A      - the address of matrix A

    lda    - the first dimension of matrix A

    B      - the address of matrix B

    ldb    - the first dimension of matrix B

    ZeroMode - true if the output matrix must be zero initialized, else
        if the output matrix is accumulated into

--*/

/****
Main loop processes 8x32 tile, depth 2.

             B 2x32
            --------
            |v0..v1|
            |v2..v3|
  A 8x2     --------
----------  ----------
|ft0..ft0|  |v16..v17|
|ft1..ft1|  |v18..v19|
|ft2..ft2|  |v20..v21|
|ft3..ft3|  |v22..v23|
|ft4..ft4|  |v24..v25|
|ft5..ft5|  |v26..v27|
|ft6..ft6|  |v28..v29|
|ft7..ft7|  |v30..v31|
---------   ----------
****/
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "../halfgemm.h"
#include "../mlasi.h"

extern "C" {
size_t
MlasHalfGemmKernelRVV(size_t m,
                      size_t n,
                      size_t k,
                      __fp16 *c,
                      size_t ldc,
                      __fp16 *Bias,
                      __fp16 *a,
                      size_t lda,
                      __fp16 *b,
                      size_t ldb,
                      bool ZeroMode)
{
    __fp16 *c0, *c1, *c2, *c3;
    __fp16 *a0, *a1, *a2, *a3;

    __fp16 *pa, *pb, *bias;

    for (size_t i = 0; i < m / 4; i++) {
        c0 = (__fp16 *)c + i * 4 * ldc;
        c1 = c0 + ldc;
        c2 = c1 + ldc;
        c3 = c2 + ldc;

        pa = (__fp16 *)a + i * 4 * lda;
        for (size_t j = 0; j < n / 32; j++) {
            bias = (__fp16 *)Bias + j * 32;
            pb = (__fp16 *)b + j * 32;
            a0 = pa;
            a1 = a0 + lda;
            a2 = a1 + lda;
            a3 = a2 + lda;
            // t0 for k
            // ft0-ft7 for A
            // v0-v1 for B
            // v16-v31 for temp C

            asm volatile(
                "vsetvli      t0, zero, e16,m1    \n\t"
                "mv           t0,   %[K]          \n\t"

                "vl2r.v       v16, (%[BIAS])      \n\t"
                //"addi         %[BIAS], %[BIAS], 64\n\t"
                "vor.vv       v18, v16, v16       \n\t"
                "vor.vv       v19, v17, v17       \n\t"
                "vor.vv       v20, v16, v16       \n\t"
                "vor.vv       v21, v17, v17       \n\t"
                "vor.vv       v22, v16, v16       \n\t"
                "vor.vv       v23, v17, v17       \n\t"

                ".align 4                         \n\t"
                "M8X32_MAIN:                      \n\t"
                "srli         t1, t0, 1           \n\t"
                "blez         t1, M4x32_MAINTAIL  \n\t"

                "M4x32_MAINLOOP:                  \n\t"
                // 1
                "vl2r.v       v0, (%[PB])         \n\t"
                "add          %[PB], %[PB], %[LDB]\n\t"
                "flh          ft0, (%[A0])        \n\t"
                "flh          ft1, (%[A1])        \n\t"
                "flh          ft2, (%[A2])        \n\t"
                "flh          ft3, (%[A3])        \n\t"

                "vfmacc.vf    v16, ft0, v0        \n\t"
                "vfmacc.vf    v17, ft0, v1        \n\t"
                "flh          ft4, 2(%[A0])       \n\t"

                "vfmacc.vf    v18, ft1, v0        \n\t"
                "vfmacc.vf    v19, ft1, v1        \n\t"
                "flh          ft5, 2(%[A1])       \n\t"

                "vfmacc.vf    v20, ft2, v0        \n\t"
                "vfmacc.vf    v21, ft2, v1        \n\t"
                "flh          ft6, 2(%[A2])       \n\t"

                "vfmacc.vf    v22, ft3, v0        \n\t"
                "vfmacc.vf    v23, ft3, v1        \n\t"
                "flh          ft7, 2(%[A3])       \n\t"

                "vl2r.v       v2, (%[PB])         \n\t"
                "add          %[PB], %[PB], %[LDB]\n\t"

                // 2
                "vfmacc.vf    v16, ft4, v2        \n\t"
                "addi         %[A0], %[A0], 4     \n\t"
                "vfmacc.vf    v17, ft4, v3        \n\t"

                "vfmacc.vf    v18, ft5, v2        \n\t"
                "addi         %[A1], %[A1], 4     \n\t"
                "vfmacc.vf    v19, ft5, v3        \n\t"

                "vfmacc.vf    v20, ft6, v2        \n\t"
                "addi         %[A2], %[A2], 4     \n\t"
                "vfmacc.vf    v21, ft6, v3        \n\t"

                "vfmacc.vf    v22, ft7, v2        \n\t"
                "addi         %[A3], %[A3], 4     \n\t"
                "vfmacc.vf    v23, ft7, v3        \n\t"

                "addi         t1, t1, -1          \n\t"
                "bgtz         t1, M4x32_MAINLOOP  \n\t"

                "andi         t1, t0, 1           \n\t"
                "blez         t1, RESULT          \n\t"

                "M4x32_MAINTAIL:                  \n\t"
                // 1
                "vl2r.v       v0, (%[PB])         \n\t"
                "add          %[PB], %[PB], %[LDB]\n\t"
                "flh          ft0, (%[A0])        \n\t"
                "flh          ft1, (%[A1])        \n\t"
                "flh          ft2, (%[A2])        \n\t"
                "flh          ft3, (%[A3])        \n\t"

                "vfmacc.vf    v16, ft0, v0        \n\t"
                "vfmacc.vf    v17, ft0, v1        \n\t"

                "vfmacc.vf    v18, ft1, v0        \n\t"
                "vfmacc.vf    v19, ft1, v1        \n\t"

                "vfmacc.vf    v20, ft2, v0        \n\t"
                "vfmacc.vf    v21, ft2, v1        \n\t"

                "vfmacc.vf    v22, ft3, v0        \n\t"
                "vfmacc.vf    v23, ft3, v1        \n\t"

                "RESULT:                          \n\t"
                "vs2r.v       v16, (%[C0])        \n\t"
                "addi         %[C0], %[C0], 2*32  \n\t"

                "vs2r.v       v18, (%[C1])        \n\t"
                "addi         %[C1], %[C1], 2*32  \n\t"

                "vs2r.v       v20, (%[C2])        \n\t"
                "addi         %[C2], %[C2], 2*32  \n\t"

                "vs2r.v       v22, (%[C3])        \n\t"
                "addi         %[C3], %[C3], 2*32  \n\t"

                : [ PB ] "+r"(pb), [ C0 ] "+r"(c0), [ C1 ] "+r"(c1), [ C2 ] "+r"(c2), [ C3 ] "+r"(c3), [ A0 ] "+r"(a0),
                  [ A1 ] "+r"(a1), [ A2 ] "+r"(a2), [ A3 ] "+r"(a3), [ BIAS ] "+r"(bias)
                : [ K ] "r"(k), [ LDB ] "r"(2 * ldb)
                : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "t0", "t1", "v13",
                  "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
                  "v28", "v29", "v30", "v31", "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7");
        }
        if ((n & 31) > 0) {
            pb = (__fp16 *)b + (n / 32) * 32;
            bias = (__fp16 *)Bias + (n / 32) * 32;
            a0 = pa;
            a1 = a0 + lda;
            a2 = a1 + lda;
            a3 = a2 + lda;
            int delta_n = n & 31;
            // t0 for k
            // ft0-ft7 for A
            // v0-v1 for B
            // v16-v31 for temp C

            asm volatile(
                "vsetvli      t0, %[DELTA_N], e16,m2    \n\t"
                "mv           t0,   %[K]          \n\t"

                "vle16.v      v16, (%[BIAS])      \n\t"
                "vor.vv       v18, v16, v16       \n\t"
                "vor.vv       v20, v16, v16       \n\t"
                "vor.vv       v22, v16, v16       \n\t"

                ".align 4                         \n\t"
                "M8X32_MAIN1:                      \n\t"
                "srli         t1, t0, 1           \n\t"
                "blez         t1, M4x32_MAINTAIL1  \n\t"

                "M4x32_MAINLOOP1:                  \n\t"
                // 1
                "vle16.v      v0, (%[PB])         \n\t"
                "add          %[PB], %[PB], %[LDB]\n\t"
                "flh          ft0, (%[A0])        \n\t"
                "flh          ft1, (%[A1])        \n\t"
                "flh          ft2, (%[A2])        \n\t"
                "flh          ft3, (%[A3])        \n\t"

                "vfmacc.vf    v16, ft0, v0        \n\t"
                "flh          ft4, 2(%[A0])       \n\t"

                "vfmacc.vf    v18, ft1, v0        \n\t"
                "flh          ft5, 2(%[A1])       \n\t"

                "vfmacc.vf    v20, ft2, v0        \n\t"
                "flh          ft6, 2(%[A2])       \n\t"

                "vfmacc.vf    v22, ft3, v0        \n\t"
                "flh          ft7, 2(%[A3])       \n\t"

                "vle16.v       v2, (%[PB])         \n\t"
                "add          %[PB], %[PB], %[LDB]\n\t"

                // 2
                "vfmacc.vf    v16, ft4, v2        \n\t"
                "addi         %[A0], %[A0], 4     \n\t"

                "vfmacc.vf    v18, ft5, v2        \n\t"
                "addi         %[A1], %[A1], 4     \n\t"

                "vfmacc.vf    v20, ft6, v2        \n\t"
                "addi         %[A2], %[A2], 4     \n\t"

                "vfmacc.vf    v22, ft7, v2        \n\t"
                "addi         %[A3], %[A3], 4     \n\t"

                "addi         t1, t1, -1          \n\t"
                "bgtz         t1, M4x32_MAINLOOP1  \n\t"

                "andi         t1, t0, 1           \n\t"
                "blez         t1, RESULT1          \n\t"

                "M4x32_MAINTAIL1:                  \n\t"
                // 1
                "vle16.v       v0, (%[PB])         \n\t"
                "add          %[PB], %[PB], %[LDB]\n\t"
                "flh          ft0, (%[A0])        \n\t"
                "flh          ft1, (%[A1])        \n\t"
                "flh          ft2, (%[A2])        \n\t"
                "flh          ft3, (%[A3])        \n\t"

                "vfmacc.vf    v16, ft0, v0        \n\t"

                "vfmacc.vf    v18, ft1, v0        \n\t"

                "vfmacc.vf    v20, ft2, v0        \n\t"

                "vfmacc.vf    v22, ft3, v0        \n\t"

                "RESULT1:                          \n\t"
                "vse16.v       v16, (%[C0])        \n\t"
                "addi         %[C0], %[C0], 2*32  \n\t"

                "vse16.v       v18, (%[C1])        \n\t"
                "addi         %[C1], %[C1], 2*32  \n\t"

                "vse16.v       v20, (%[C2])        \n\t"
                "addi         %[C2], %[C2], 2*32  \n\t"

                "vse16.v       v22, (%[C3])        \n\t"
                "addi         %[C3], %[C3], 2*32  \n\t"

                : [ PB ] "+r"(pb), [ C0 ] "+r"(c0), [ C1 ] "+r"(c1), [ C2 ] "+r"(c2), [ C3 ] "+r"(c3), [ A0 ] "+r"(a0),
                  [ A1 ] "+r"(a1), [ A2 ] "+r"(a2), [ A3 ] "+r"(a3), [ BIAS ] "+r"(bias)
                : [ K ] "r"(k), [ LDB ] "r"(2 * ldb), [ DELTA_N ] "r"(delta_n)
                : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "t0", "t1", "v13",
                  "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "ft0", "ft1", "ft2", "ft3",
                  "ft4", "ft5", "ft6", "ft7");
        }
    }

    for (size_t i = 0; i < (m & 3); i++) {
        c0 = (__fp16 *)c + (m / 4) * 4 * ldc + i * ldc;
        pa = (__fp16 *)a + (m / 4) * 4 * lda + i * lda;

        for (size_t j = 0; j < n / 32; j++) {
            bias = (__fp16 *)Bias + j * 32;
            pb = (__fp16 *)b + j * 32;
            a0 = pa;
            // t0 for k
            // ft0-ft7 for A
            // v0-v1 for B
            // v16-v31 for temp C

            asm volatile(
                "vsetvli      t0, zero, e16,m1    \n\t"
                "mv           t0,   %[K]          \n\t"

                "vl2r.v       v16, (%[BIAS])      \n\t"
                //"addi         %[BIAS], %[BIAS], 32\n\t"

                ".align 4                         \n\t"
                "M8X32_MAIN2:                      \n\t"
                "srli         t1, t0, 1           \n\t"
                "blez         t1, M4x32_MAINTAIL2  \n\t"

                "M4x32_MAINLOOP2:                  \n\t"
                // 1
                "vl2r.v       v0, (%[PB])         \n\t"
                "add          %[PB], %[PB], %[LDB]\n\t"
                "flh          ft0, (%[A0])        \n\t"

                "vfmacc.vf    v16, ft0, v0        \n\t"
                "vfmacc.vf    v17, ft0, v1        \n\t"
                "flh          ft1, 2(%[A0])       \n\t"

                "vl2r.v       v2, (%[PB])         \n\t"
                "add          %[PB], %[PB], %[LDB]\n\t"

                // 2
                "vfmacc.vf    v16, ft1, v2        \n\t"
                "addi         %[A0], %[A0], 4     \n\t"
                "vfmacc.vf    v17, ft1, v3        \n\t"

                "addi         t1, t1, -1          \n\t"
                "bgtz         t1, M4x32_MAINLOOP2  \n\t"

                "andi         t1, t0, 1           \n\t"
                "blez         t1, RESULT2          \n\t"

                "M4x32_MAINTAIL2:                  \n\t"
                // 1
                "vl2r.v       v0, (%[PB])         \n\t"
                "add          %[PB], %[PB], %[LDB]\n\t"
                "flh          ft0, (%[A0])        \n\t"

                "vfmacc.vf    v16, ft0, v0        \n\t"
                "vfmacc.vf    v17, ft0, v1        \n\t"

                "RESULT2:                          \n\t"
                "vs2r.v       v16, (%[C0])        \n\t"
                "addi         %[C0], %[C0], 2*32  \n\t"

                : [ PB ] "+r"(pb), [ C0 ] "+r"(c0), [ A0 ] "+r"(a0), [ BIAS ] "+r"(bias)
                : [ K ] "r"(k), [ LDB ] "r"(2 * ldb)
                : "v0", "v1", "v2", "v3", "v16", "v17", "ft0", "ft1", "t0", "t1");
        }
        if ((n & 31) > 0) {
            pb = (__fp16 *)b + (n / 32) * 32;
            bias = (__fp16 *)Bias + (n / 32) * 32;
            a0 = pa;
            int delta_n = n & 31;
            // t0 for k
            // ft0-ft7 for A
            // v0-v1 for B
            // v16-v31 for temp C

            asm volatile(
                "vsetvli      t0, %[DELTA_N], e16,m2    \n\t"
                "mv           t0,   %[K]          \n\t"

                "vle16.v      v16, (%[BIAS])      \n\t"

                ".align 4                         \n\t"
                "M8X32_MAIN3:                      \n\t"
                "srli         t1, t0, 1           \n\t"
                "blez         t1, M4x32_MAINTAIL3  \n\t"

                "M4x32_MAINLOOP3:                  \n\t"
                // 1
                "vle16.v      v0, (%[PB])         \n\t"
                "add          %[PB], %[PB], %[LDB]\n\t"
                "flh          ft0, (%[A0])        \n\t"

                "vfmacc.vf    v16, ft0, v0        \n\t"
                "flh          ft1, 2(%[A0])       \n\t"

                "vle16.v       v2, (%[PB])         \n\t"
                "add          %[PB], %[PB], %[LDB]\n\t"

                // 2
                "vfmacc.vf    v16, ft1, v2        \n\t"
                "addi         %[A0], %[A0], 4     \n\t"

                "addi         t1, t1, -1          \n\t"
                "bgtz         t1, M4x32_MAINLOOP3  \n\t"

                "andi         t1, t0, 1           \n\t"
                "blez         t1, RESULT3          \n\t"

                "M4x32_MAINTAIL3:                  \n\t"
                // 1
                "vle16.v       v0, (%[PB])         \n\t"
                "add          %[PB], %[PB], %[LDB]\n\t"
                "flh          ft0, (%[A0])        \n\t"

                "vfmacc.vf    v16, ft0, v0        \n\t"

                "RESULT3:                          \n\t"
                "vse16.v       v16, (%[C0])        \n\t"
                "addi         %[C0], %[C0], 2*32  \n\t"

                : [ PB ] "+r"(pb), [ C0 ] "+r"(c0), [ A0 ] "+r"(a0), [ BIAS ] "+r"(bias)
                : [ K ] "r"(k), [ LDB ] "r"(2 * ldb), [ DELTA_N ] "r"(delta_n)
                : "v0", "v1", "v2", "v3", "v16", "v17", "t0", "t1", "ft0", "ft1");
        }
    }

    return 0;
}
}
