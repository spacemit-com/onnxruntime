/*++

Copyright (c) SpaceMIT Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    halfgemm_kernel_rvv.cpp

Abstract:

    This module implements half precision GEMM kernel for rvv.

--*/

#include "mlasi.h"
#include "halfgemm.h"

//
// Define the prototypes of the RVV routines written in assembly.
//

extern "C" {

   size_t
    MLASCALL
    MlasHalfGemmKernelRVV(
        const size_t CountM,
        const size_t CountN,
        const size_t CountK,
        _mlas_fp16_* C,
        size_t ldc,
        const _mlas_fp16_* Bias,
        const _mlas_fp16_* A,
        const size_t lda,
        const _mlas_fp16_* B,
        const size_t ldb,
        const bool ZeroMode
        );

}


struct MLAS_HALF_GEMM_KERNEL_RVV {
    static constexpr bool PackNeeded = false;
    static constexpr size_t KernelMaxM = 8;  // max # rows the vectorized kernel can process
    static constexpr size_t PackedK = 1;

    static constexpr MLAS_HALF_GEMM_STRIDES Strides{24, 128, 512};
};

MLAS_FORCEINLINE
void
CvtFloat2Half(
    _mlas_fp16_* dest,
    const float* src,
    size_t len
)
{
#if 0
    while (len > 0) {
        size_t gvl = __riscv_vsetvl_e32m2(len);
        vfloat32m2_t srcData = __riscv_vle32_v_f32m2(srcPtr, gvl);
        vfloat16m1_t dstData = __riscv_vfncvt_f_f_w_f16m1(srcData, gvl);
        __riscv_vse16_v_f16m1(dest, dstData, gvl);

        src += gvl;
        dest += gvl;
        len -= gvl;
    }
#endif
    __fp16 *dst = (__fp16 *)dest;
     while (len > 0) {
         *dst++ = (__fp16)*src++;
         len--;
     }
}
/**
 * @brief Convert a 2D matrix from float to fp16
*/
MLAS_FORCEINLINE
void
CvtFloat2Half2D(
    _mlas_fp16_* dest,
    const float* src,
    size_t stride,
    size_t CntRow,
    size_t CntCol
    )
{
    if (stride == CntCol) {
        const size_t len = CntRow * CntCol;
        CvtFloat2Half(dest, src, len);
        return;
    }
    while (CntRow > 0) {
        CvtFloat2Half(dest, src, CntCol);
        src += stride;
        dest += CntCol;
        CntRow--;
    }
}

template<>
MLAS_FORCEINLINE
void
MlasHalfGemmConvertPackA<MLAS_HALF_GEMM_KERNEL_RVV>(
    _mlas_fp16_* D,
    const float* A,
    size_t lda,
    size_t CountM,
    size_t CountK
)
{
    CvtFloat2Half2D(D, A, lda, CountM, CountK);
}

template<>
MLAS_FORCEINLINE
void
MlasHalfGemmConvertPackB<MLAS_HALF_GEMM_KERNEL_RVV>(
    _mlas_fp16_* D,
    const float* B,
    size_t ldb,
    size_t CountN,
    size_t CountK
)
{
    CvtFloat2Half2D(D, B, ldb, CountK, CountN);
}


template<>
MLAS_FORCEINLINE
void
MlasHalfGemmKernel<MLAS_HALF_GEMM_KERNEL_RVV>(
    size_t CountM,
    size_t CountN,
    size_t CountK,
    _mlas_fp16_* C,
    size_t ldc,
    const _mlas_fp16_* Bias,
    const _mlas_fp16_* A,
    size_t lda,
    const _mlas_fp16_* B,
    size_t ldb,
    const bool ZeroMode)
{
    MlasHalfGemmKernelRVV(
        CountM,
        CountN,
        CountK,
        C,
        ldc,
        Bias,
        A,
        lda,
        B,
        ldb,
        ZeroMode);
}


const MLAS_HALFGEMM_DISPATCH MlasHalfGemmDispatchRvv = {
    MlasHalfGemmOperation<MLAS_HALF_GEMM_KERNEL_RVV>,
    nullptr,
    MlasHalfGemmConvertPackB<MLAS_HALF_GEMM_KERNEL_RVV>,
    MLAS_HALF_GEMM_KERNEL_RVV::PackedK,
    MLAS_HALF_GEMM_KERNEL_RVV::KernelMaxM,
    32 // kernel may read beyond buffer end by 32 bytes
};
