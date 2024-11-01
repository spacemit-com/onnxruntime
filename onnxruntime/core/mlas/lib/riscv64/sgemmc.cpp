// Copyright (c) 2023 SpacemiT. All rights reserved.
// Licensed under the MIT License.

#include "../mlasi.h"

extern "C" size_t MlasSgemmKernel_8x16(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha
    );

extern "C" size_t MlasSgemmKernel_8x16_Update(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha
    );

template<bool ZeroMode>
size_t
MlasSgemmKernelAsm(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha
    )
/*++

Routine Description:

    This routine is an inner kernel to compute matrix multiplication for a
    set of rows.

Arguments:

    A - Supplies the address of matrix A.

    B - Supplies the address of matrix B. The matrix data has been packed using
        MlasSgemmCopyPackB or MlasSgemmTransposePackB.

    C - Supplies the address of matrix C.

    CountK - Supplies the number of columns from matrix A and the number of rows
        from matrix B to iterate over.

    CountM - Supplies the maximum number of rows that can be processed for
        matrix A and matrix C. The actual number of rows handled for this
        invocation depends on the kernel implementation.

    CountN - Supplies the number of columns from matrix B and matrix C to
        iterate over.

    lda - Supplies the first dimension of matrix A.

    ldc - Supplies the first dimension of matrix C.

    alpha - Supplies the scalar multiplier (see SGEMM definition).

Return Value:

    Returns the number of rows handled.

--*/
{
    size_t RowsHandled;

    if (!ZeroMode) {
        RowsHandled = MlasSgemmKernel_8x16_Update(A, B, C, CountK, CountN, lda, ldc, alpha);
    } else {
        RowsHandled = MlasSgemmKernel_8x16(A, B, C, CountK, CountN, lda, ldc, alpha);
    }
    return RowsHandled;
}

template<bool ZeroMode, bool ProcessTwoRows>
size_t
MlasSgemmKernel(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha
    )
/*++

Routine Description:

    This routine is an inner kernel to compute matrix multiplication for a
    set of rows.

Arguments:

    A - Supplies the address of matrix A.

    B - Supplies the address of matrix B. The matrix data has been packed using
        MlasSgemmCopyPackB or MlasSgemmTransposePackB.

    C - Supplies the address of matrix C.

    CountK - Supplies the number of columns from matrix A and the number of rows
        from matrix B to iterate over.

    CountN - Supplies the number of columns from matrix B and matrix C to
        iterate over.

    lda - Supplies the first dimension of matrix A.

    ldc - Supplies the first dimension of matrix C.

    alpha - Supplies the scalar multiplier (see SGEMM definition).

Return Value:

    Returns the number of rows handled.

--*/

/****
Main loop processes mr * nr tile, depth kr.
mr = 1,2
kr = 2
nr = 16
A Row Major mr x k
B Row Major n x k, packed [n/nr][k][nr]
            B: kr x nr
            --------
            |v0..v1|
            |v2..v3|
A: mr x kr  --------
----------  ----------
|ft0..ft0|  |v16..v17|
|ft1..ft1|  |v18..v19|
---------   ----------
****/

{
    vfloat32m2_t Row0 = __riscv_vundefined_f32m2();
    vfloat32m2_t Row1 = __riscv_vundefined_f32m2();
    size_t avl = __riscv_vsetvl_e32m2(CountN);

#if defined(_WIN32)

    if (!ProcessTwoRows) {
        UNREFERENCED_PARAMETER(lda);
        UNREFERENCED_PARAMETER(ldc);
    }

#endif

    do {
        avl = __riscv_vsetvl_e32m2(CountN);

        vfloat32m2_t BElements;
        float_t Row0AElements;
        float_t Row1AElements;

        //
        // Clear the block accumulators.
        //

        Row0 = __riscv_vreinterpret_f32m2(__riscv_vxor_vv_i32m2(__riscv_vreinterpret_i32m2(Row0), __riscv_vreinterpret_i32m2(Row0), avl));

        if (ProcessTwoRows) {
            Row1 = __riscv_vreinterpret_f32m2(__riscv_vxor_vv_i32m2(__riscv_vreinterpret_i32m2(Row1), __riscv_vreinterpret_i32m2(Row1), avl));
        }

        //
        // Compute the 16x1 or 16x2 output block.
        //

        const float* a = A;
        size_t k = CountK;

        while (k >= 2) {

            Row0AElements = a[0];

            if (ProcessTwoRows) {
                Row1AElements = a[lda];
            }

            BElements = __riscv_vle32_v_f32m2(B, avl);

            Row0 = __riscv_vfmacc_vf_f32m2(Row0, Row0AElements, BElements, avl);
            Row0AElements = a[0 + 1];

            if (ProcessTwoRows) {
                Row1 = __riscv_vfmacc_vf_f32m2(Row1, Row1AElements, BElements, avl);
                Row1AElements = a[lda + 1];
            }

            BElements = __riscv_vle32_v_f32m2(B + 16, avl);

            Row0 = __riscv_vfmacc_vf_f32m2(Row0, Row0AElements, BElements, avl);

            if (ProcessTwoRows) {
                Row1 = __riscv_vfmacc_vf_f32m2(Row1, Row1AElements, BElements, avl);
            }

            a += 2;
            B += 32;
            k -= 2;
        }

        if (k > 0) {

            // Row0AElements = vld1_dup_f32(a);
            Row0AElements = a[0];

            if (ProcessTwoRows) {
                // Row1AElements = vld1_dup_f32(a + lda);
                Row1AElements = a[lda];
            }

            BElements = __riscv_vle32_v_f32m2(B, avl);

            Row0 = __riscv_vfmacc_vf_f32m2(Row0, Row0AElements, BElements, avl);

            if (ProcessTwoRows) {
                Row1 = __riscv_vfmacc_vf_f32m2(Row1, Row1AElements, BElements, avl);
            }

            B += 16;
        }

        //
        // Multiply by the alpha value.
        //
        Row0 = __riscv_vfmul_vf_f32m2(Row0, alpha, avl);

        if (ProcessTwoRows) {
            Row1 = __riscv_vfmul_vf_f32m2(Row1, alpha, avl);
        }

        if (CountN) {

            //
            // Store the entire output block.
            //

            if (!ZeroMode) {
                Row0 = __riscv_vfadd_vv_f32m2(Row0, __riscv_vle32_v_f32m2(C, avl), avl);
            }
            __riscv_vse32_v_f32m2(C, Row0, avl);

            if (ProcessTwoRows) {

                if (!ZeroMode) {
                    Row1 = __riscv_vfadd_vv_f32m2(Row1, __riscv_vle32_v_f32m2(C + ldc, avl), avl);
                }
                __riscv_vse32_v_f32m2(C + ldc, Row1, avl);
            }

        }

        C += avl;
        CountN -= avl;

    } while (CountN > 0);

    //
    // Compute the number of rows that were processed for this invocation.
    //
    return ProcessTwoRows ? 2 : 1;
}

template<bool ZeroMode>
size_t
MlasSgemmKernel(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountM,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha
    )
/*++

Routine Description:

    This routine is an inner kernel to compute matrix multiplication for a
    set of rows.

Arguments:

    A - Supplies the address of matrix A.

    B - Supplies the address of matrix B. The matrix data has been packed using
        MlasSgemmCopyPackB or MlasSgemmTransposePackB.

    C - Supplies the address of matrix C.

    CountK - Supplies the number of columns from matrix A and the number of rows
        from matrix B to iterate over.

    CountM - Supplies the maximum number of rows that can be processed for
        matrix A and matrix C. The actual number of rows handled for this
        invocation depends on the kernel implementation.

    CountN - Supplies the number of columns from matrix B and matrix C to
        iterate over.

    lda - Supplies the first dimension of matrix A.

    ldc - Supplies the first dimension of matrix C.

    alpha - Supplies the scalar multiplier (see SGEMM definition).

Return Value:

    Returns the number of rows handled.

--*/
{
    size_t RowsHandled;
    if (CountM >= 8) {
        RowsHandled = MlasSgemmKernelAsm<ZeroMode>(A, B, C, CountK, CountN, lda, ldc, alpha);
    } else if (CountM >= 2) {
        RowsHandled = MlasSgemmKernel<ZeroMode, true>(A, B, C, CountK, CountN, lda, ldc, alpha);
    } else {
        RowsHandled = MlasSgemmKernel<ZeroMode, false>(A, B, C, CountK, CountN, lda, ldc, alpha);
    }
    return RowsHandled;
}

size_t
MLASCALL
MlasSgemmKernelZero(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountM,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha
    )
/*++

Routine Description:

    This routine is an inner kernel to compute matrix multiplication for a
    set of rows.

Arguments:

    A - Supplies the address of matrix A.

    B - Supplies the address of matrix B. The matrix data has been packed using
        MlasSgemmCopyPackB or MlasSgemmTransposePackB.

    C - Supplies the address of matrix C.

    CountK - Supplies the number of columns from matrix A and the number of rows
        from matrix B to iterate over.

    CountM - Supplies the maximum number of rows that can be processed for
        matrix A and matrix C. The actual number of rows handled for this
        invocation depends on the kernel implementation.

    CountN - Supplies the number of columns from matrix B and matrix C to
        iterate over.

    lda - Supplies the first dimension of matrix A.

    ldc - Supplies the first dimension of matrix C.

    alpha - Supplies the scalar multiplier (see SGEMM definition).

Return Value:

    Returns the number of rows handled.

--*/
{
    return MlasSgemmKernel<true>(A, B, C, CountK, CountM, CountN, lda, ldc, alpha);
}

size_t
MLASCALL
MlasSgemmKernelAdd(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountM,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha
    )
/*++

Routine Description:

    This routine is an inner kernel to compute matrix multiplication for a
    set of rows.

Arguments:

    A - Supplies the address of matrix A.

    B - Supplies the address of matrix B. The matrix data has been packed using
        MlasSgemmCopyPackB or MlasSgemmTransposePackB.

    C - Supplies the address of matrix C.

    CountK - Supplies the number of columns from matrix A and the number of rows
        from matrix B to iterate over.

    CountM - Supplies the maximum number of rows that can be processed for
        matrix A and matrix C. The actual number of rows handled for this
        invocation depends on the kernel implementation.

    CountN - Supplies the number of columns from matrix B and matrix C to
        iterate over.

    lda - Supplies the first dimension of matrix A.

    ldc - Supplies the first dimension of matrix C.

    alpha - Supplies the scalar multiplier (see SGEMM definition).

Return Value:

    Returns the number of rows handled.

--*/
{
    return MlasSgemmKernel<false>(A, B, C, CountK, CountM, CountN, lda, ldc, alpha);
}
