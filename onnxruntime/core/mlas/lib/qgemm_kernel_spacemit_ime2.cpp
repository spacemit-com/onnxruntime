// Copyright (c) 2023 SpacemiT. All rights reserved.
// Licensed under the MIT License.

#include "qgemm_kernel_spacemit_ime.h"

#include <algorithm>
#include <cassert>

#include "mlasi.h"

struct MLAS_GEMM_X8X8_KERNEL_IME2 {
    typedef uint8_t PackedAType;
    typedef uint8_t PackedBType;
    typedef uint8_t OffsetAType;
    typedef uint8_t OffsetBType;

    static constexpr size_t PackedK = 16;
    static constexpr size_t PackedN = 32;

    static constexpr MLAS_GEMM_QUANT_STRIDES Strides{16, 384, 1024};
    static constexpr MLAS_GEMM_QUANT_STRIDES PackedStrides{16, 384, 1024};
};

template <typename MLAS_GEMM_X8X8_KERNEL_IME2_TYPE>
MLAS_FORCEINLINE constexpr int32_t
MlasGemmQuantFixupZeroPointA_impl(int32_t ZeroPointA, bool AIsSigned)
{
    if (AIsSigned) {
        ZeroPointA = (uint8_t)(ZeroPointA ^ 0x80);
    }

    return ZeroPointA;
}

template <typename MLAS_GEMM_X8X8_KERNEL_IME2_TYPE>
MLAS_FORCEINLINE constexpr int32_t
MlasGemmQuantFixupZeroPointB_impl(int32_t ZeroPointB, bool BIsSigned)
{
    if (BIsSigned) {
        ZeroPointB = typename MLAS_GEMM_X8X8_KERNEL_IME2_TYPE::OffsetBType(ZeroPointB ^ 0x80);
    }

    return ZeroPointB;
}

#define REGISTER_MlasGemmQuantFixupZeroPoint(TYPE)                                                           \
    template <>                                                                                              \
    MLAS_FORCEINLINE constexpr int32_t MlasGemmQuantFixupZeroPointA<TYPE>(int32_t ZeroPoint, bool BIsSigned) \
    {                                                                                                        \
        return MlasGemmQuantFixupZeroPointA_impl<TYPE>(ZeroPoint, BIsSigned);                                \
    }                                                                                                        \
    template <>                                                                                              \
    MLAS_FORCEINLINE constexpr int32_t MlasGemmQuantFixupZeroPointB<TYPE>(int32_t ZeroPoint, bool BIsSigned) \
    {                                                                                                        \
        return MlasGemmQuantFixupZeroPointB_impl<TYPE>(ZeroPoint, BIsSigned);                                \
    }

template <typename MLAS_GEMM_X8X8_KERNEL_IME2_TYPE>
void
MlasGemmQuantCopyPackA_impl(typename MLAS_GEMM_X8X8_KERNEL_IME2_TYPE::PackedAType *D,
                            const uint8_t *A,
                            size_t lda,
                            size_t CountM,
                            size_t CountK,
                            int32_t *RowSumBuffer,
                            bool AIsSigned)
{
    const size_t AlignedCountK =
        (CountK + MLAS_GEMM_X8X8_KERNEL_IME2_TYPE::PackedK - 1) & ~(MLAS_GEMM_X8X8_KERNEL_IME2_TYPE::PackedK - 1);

    const uint8_t BitFlipValue = (AIsSigned ? 0x80 : 0);
    constexpr size_t Stride = MLAS_GEMM_X8X8_KERNEL_IME2_TYPE::PackedK * MLAS_GEMM_X8X8_KERNEL_IME2_TYPE::Strides.M;
    //
    // Process a single row of matrix A in a loop.
    //
    auto d = D;
    while (CountM-- > 0) {
        int32_t RowSum = 0;
        const uint8_t *a = A;
        __asm__ volatile(
            // init the index
            "vsetvli       t0, x0, e64, m1      \n\t"
            "vid.v         v2                   \n\t"
            "vsll.vi       v6, v2, 1            \n\t"
            "vadd.vi       v12, v6, 1           \n\t"
            "vsetvli       t0, x0, e32, m1      \n\t"

            "vmv.v.x       v4, zero             \n\t"
            "addi          t1, %[k], 0          \n\t"
            "addi          t2, %[b], 0          \n\t"
            "srli          t3, %[aligned_k], 3  \n\t"

            "LOOP%=:                            \n\t"
            "vsetvli       t0, t1, e8, m2       \n\t"
            "vle8.v        v0, (%[a])           \n\t"
            "add           %[a], %[a], t0       \n\t"
            "vxor.vx       v0, v0, %[f]         \n\t"
            "vsetvli       t0, t1, e16, m4      \n\t"
            "vzext.vf2     v8, v0               \n\t"
            "vwredsumu.vs  v4, v8, v4           \n\t"
            "sub           t1, t1, t0           \n\t"

            "vsetvli       t0, x0, e64, m2      \n\t"
            "vrgather.vv   v2, v0, v6           \n\t"
            "vrgather.vv   v14, v0, v12         \n\t"
            "vsetvli       t0, x0, e64, m1      \n\t"
            "vor.vv        v3, v14, v14         \n\t"
            "vsetvli       t0, t3, e64, m1      \n\t"
            "vssseg2e64.v  v2, (t2), %[s]       \n\t"
            "mul           t0, t0, %[s]         \n\t"
            "add           t2, t2, t0           \n\t"

            "bnez          t1, LOOP%=           \n\t"
            "vsetvli       t0, x0, e32, m1      \n\t"
            "vmv.x.s       %[sum], v4           \n\t"
            :[a]"+r"(a), [sum]"+r"(RowSum)
            :[k]"r"(CountK), [ aligned_k] "r"(AlignedCountK), [f] "r"(BitFlipValue), [b] "r"(D),
             [s]"r"(Stride)
            : "cc", "t0", "t1", "t2", "t3");
        *RowSumBuffer++ = RowSum;
        A += lda;
        D += MLAS_GEMM_X8X8_KERNEL_IME2_TYPE::PackedK;
    }

    if (CountK & (MLAS_GEMM_X8X8_KERNEL_IME2_TYPE::PackedK - 1)) {
        for (size_t k =
                 (AlignedCountK - MLAS_GEMM_X8X8_KERNEL_IME2_TYPE::PackedK) * MLAS_GEMM_X8X8_KERNEL_IME2_TYPE::Strides.M;
             k < AlignedCountK * MLAS_GEMM_X8X8_KERNEL_IME2_TYPE::Strides.M;
             k += MLAS_GEMM_X8X8_KERNEL_IME2_TYPE::PackedK) {
            for (size_t kk = CountK & (MLAS_GEMM_X8X8_KERNEL_IME2_TYPE::PackedK - 1);
                 kk < MLAS_GEMM_X8X8_KERNEL_IME2_TYPE::PackedK; kk++) {
                d[k + kk] = 0;
            }
        }
    }

}

#define REGISTER_MlasGemmQuantCopyPackA(TYPE)                                                             \
    template <>                                                                                           \
    void MlasGemmQuantCopyPackA<TYPE>(TYPE::PackedAType * D, const uint8_t *A, size_t lda, size_t CountM, \
                                      size_t CountK, int32_t *RowSumBuffer, bool AIsSigned)               \
    {                                                                                                     \
        MlasGemmQuantCopyPackA_impl<TYPE>(D, A, lda, CountM, CountK, RowSumBuffer, AIsSigned);            \
    }

template <typename MLAS_GEMM_X8X8_KERNEL_IME2_TYPE>
void
MlasGemmQuantCopyPackB_impl(typename MLAS_GEMM_X8X8_KERNEL_IME2_TYPE::PackedBType *D,
                            const uint8_t *B,
                            size_t ldb,
                            size_t CountN,
                            size_t CountK,
                            int32_t *ColumnSumBuffer,
                            bool BIsSigned)
{
    const uint8_t BitFlipValue = (BIsSigned ? 0x80 : 0);
    const size_t Stride = MLAS_GEMM_X8X8_KERNEL_IME2_TYPE::PackedK * MLAS_GEMM_X8X8_KERNEL_IME2_TYPE::PackedN;
    // Warmup
    auto b = B;
    auto StrideDiff = ldb - CountN;
    __asm__ volatile(
        "addi          t1, %[k], 0          \n\t"
        "LOOPK%=:                           \n\t"
        "addi          t1, t1, -1           \n\t"
        "addi          t2, %[n], 0          \n\t"

        "LOOPN%=:                           \n\t"
        "vsetvli       t0, t2, e8, m2       \n\t"
        "vle8.v        v0, (%[b])           \n\t"
        "add           %[b], %[b], t0       \n\t"
        "sub           t2, t2, t0           \n\t"
        "bnez          t2, LOOPN%=          \n\t"
        "add           %[b], %[b], %[sd]    \n\t"
        "bnez          t1, LOOPK%=          \n\t"
        :[b]"+r"(b)
        :[k]"r"(CountK), [n]"r"(CountN), [sd]"r"(StrideDiff)
        :"cc", "t0", "t1", "t2");
    auto d = D;
    size_t SubCountN = 0;
    for (size_t n = 0; n < CountN; n += SubCountN) {
        SubCountN = std::min(CountN - n, MLAS_GEMM_X8X8_KERNEL_IME2_TYPE::PackedN);
        // init column sum
        for (size_t nn = 0; nn < SubCountN; nn++) {
            ColumnSumBuffer[n + nn] = 0;
        }
        // k loop
        size_t SubCountK = 0;
        size_t k;
        size_t CountKRndDown = CountK & (-MLAS_GEMM_X8X8_KERNEL_IME2_TYPE::PackedK);
        for (k = 0; k < CountKRndDown; k += MLAS_GEMM_X8X8_KERNEL_IME2_TYPE::PackedK) {
            for (size_t kk = 0; kk < MLAS_GEMM_X8X8_KERNEL_IME2_TYPE::PackedK; kk++) {
                for (size_t nn = 0; nn < SubCountN; nn++) {
                    char val = B[(k + kk) * ldb + n + nn] ^ BitFlipValue;
                    d[nn * MLAS_GEMM_X8X8_KERNEL_IME2_TYPE::PackedK + kk] = val;
                    ColumnSumBuffer[n + nn] += val;
                }
            }
            d += Stride;
        }
        SubCountK = CountK & (MLAS_GEMM_X8X8_KERNEL_IME2_TYPE::PackedK - 1);
        if (SubCountK) {
            for (size_t kk = 0; kk < SubCountK; kk++) {
                for (size_t nn = 0; nn < SubCountN; nn++) {
                    char val = B[(k + kk) * ldb + n + nn] ^ BitFlipValue;
                    d[nn * MLAS_GEMM_X8X8_KERNEL_IME2_TYPE::PackedK + kk] = val;
                    ColumnSumBuffer[n + nn] += val;
                }
            }
            for (size_t kk = SubCountK; kk < MLAS_GEMM_X8X8_KERNEL_IME2_TYPE::PackedK; kk++) {
                for (size_t nn = 0; nn < SubCountN; nn++) {
                    d[nn * MLAS_GEMM_X8X8_KERNEL_IME2_TYPE::PackedK + kk] = 0;
                }
            }
            d += Stride;
        }
    }
}

#define REGISTER_MlasGemmQuantCopyPackB(TYPE)                                                             \
    template <>                                                                                           \
    void MlasGemmQuantCopyPackB<TYPE>(TYPE::PackedBType * D, const uint8_t *B, size_t ldb, size_t CountN, \
                                      size_t CountK, int32_t *ColumnSumBuffer, bool BIsSigned)            \
    {                                                                                                     \
        MlasGemmQuantCopyPackB_impl<TYPE>(D, B, ldb, CountN, CountK, ColumnSumBuffer, BIsSigned);         \
    }

#define KERNEL16x32x16_I                                      \
      "vle32.v        v2, (t1)          \n\t"   \
      "addi           t1, t1, 4*128     \n\t"   \
      "vle32.v        v3, (t2)          \n\t"   \
      "addi           t2, t2, 4*128     \n\t"   \
                                                \
      "vle32.v        v4, (t3)          \n\t"   \
      "addi           t3, t3, 4*128     \n\t"   \
      "vle32.v        v5, (t4)          \n\t"   \
      "addi           t4, t4, 4*128     \n\t"   \
                                                \
      "vle32.v        v8, (s1)          \n\t"   \
      "addi           s1, s1, 8*128     \n\t"   \
      "vle32.v        v9, (s2)          \n\t"   \
      "addi           s2, s2, 8*128     \n\t"   \
      "vle32.v        v10, (s3)         \n\t"   \
      "addi           s3, s3, 8*128     \n\t"   \
      "vle32.v        v11, (s4)         \n\t"   \
      "addi           s4, s4, 8*128     \n\t"   \
                                                \
      "vle32.v        v12, (s5)         \n\t"   \
      "addi           s5, s5, 8*128     \n\t"   \
      "vle32.v        v13, (s6)         \n\t"   \
      "addi           s6, s6, 8*128     \n\t"   \
      "vle32.v        v14, (s7)         \n\t"   \
      "addi           s7, s7, 8*128     \n\t"   \
      "vle32.v        v15, (s8)         \n\t"   \
      "addi           s8, s8, 8*128     \n\t"   \
                                                \
      "vmadotu        v16, v2, v8, i8   \n\t"   \
      "vmadotu        v18, v3, v8, i8   \n\t"   \
      "vmadotu        v20, v2, v9, i8   \n\t"   \
      "vmadotu        v22, v3, v9, i8   \n\t"   \
      "vmadotu        v24, v2, v10, i8  \n\t"   \
      "vmadotu        v26, v3, v10, i8  \n\t"   \
      "vle32.v        v8, (s1)          \n\t"   \
      "addi           s1, s1, 8*128     \n\t"   \
      "vmadotu        v28, v2, v11, i8  \n\t"   \
      "vle32.v        v9, (s2)          \n\t"   \
      "addi           s2, s2, 8*128     \n\t"   \
      "vmadotu        v30, v3, v11, i8  \n\t"   \
      "vle32.v        v10, (s3)         \n\t"   \
      "addi           s3, s3, 8*128     \n\t"   \

#define KERNEL16x32x16_M2                       \
      "vmadotu        v16, v4, v12, i8  \n\t"   \
      "vle32.v        v11, (s4)         \n\t"   \
      "addi           s4, s4, 8*128     \n\t"   \
                                                \
      "vmadotu        v18, v5, v12, i8  \n\t"   \
      "vle32.v        v2, (t1)          \n\t"   \
      "addi           t1, t1, 4*128     \n\t"   \
      "vmadotu        v20, v4, v13, i8  \n\t"   \
      "vle32.v        v3, (t2)          \n\t"   \
      "addi           t2, t2, 4*128     \n\t"   \
      "vmadotu        v22, v5, v13, i8  \n\t"   \
      "vle32.v        v6, (t3)          \n\t"   \
      "addi           t3, t3, 4*128     \n\t"   \
      "vmadotu        v24, v4, v14, i8  \n\t"   \
      "vle32.v        v7, (t4)          \n\t"   \
      "addi           t4, t4, 4*128     \n\t"   \
                                                \
      "vmadotu        v26, v5, v14, i8  \n\t"   \
      "vle32.v        v12, (s5)         \n\t"   \
      "addi           s5, s5, 8*128     \n\t"   \
      "vmadotu        v28, v4, v15, i8  \n\t"   \
      "vle32.v        v13, (s6)         \n\t"   \
      "addi           s6, s6, 8*128     \n\t"   \
      "vmadotu        v30, v5, v15, i8  \n\t"   \
      "vle32.v        v14, (s7)         \n\t"   \
      "addi           s7, s7, 8*128     \n\t"   \

#define KERNEL16x32x16_M3                       \
      "vmadotu        v16, v2, v8, i8   \n\t"   \
      "vle32.v        v15, (s8)         \n\t"   \
      "addi           s8, s8, 8*128     \n\t"   \
                                                \
      "vmadotu        v18, v3, v8, i8   \n\t"   \
      "vmadotu        v20, v2, v9, i8   \n\t"   \
      "vmadotu        v22, v3, v9, i8   \n\t"   \
      "vmadotu        v24, v2, v10, i8  \n\t"   \
                                                \
      "vmadotu        v26, v3, v10, i8  \n\t"   \
      "vle32.v        v8, (s1)          \n\t"   \
      "addi           s1, s1, 8*128     \n\t"   \
      "vmadotu        v28, v2, v11, i8  \n\t"   \
      "vle32.v        v9, (s2)          \n\t"   \
      "addi           s2, s2, 8*128     \n\t"   \
      "vmadotu        v30, v3, v11, i8  \n\t"   \
      "vle32.v        v10, (s3)         \n\t"   \
      "addi           s3, s3, 8*128     \n\t"   \

#define KERNEL16x32x16_M4                       \
      "vmadotu        v16, v6, v12, i8  \n\t"   \
      "vle32.v        v11, (s4)         \n\t"   \
      "addi           s4, s4, 8*128     \n\t"   \
                                                \
      "vmadotu        v18, v7, v12, i8  \n\t"   \
      "vle32.v        v2, (t1)          \n\t"   \
      "addi           t1, t1, 4*128     \n\t"   \
      "vmadotu        v20, v6, v13, i8  \n\t"   \
      "vle32.v        v3, (t2)          \n\t"   \
      "addi           t2, t2, 4*128     \n\t"   \
      "vmadotu        v22, v7, v13, i8  \n\t"   \
      "vle32.v        v4, (t3)          \n\t"   \
      "addi           t3, t3, 4*128     \n\t"   \
      "vmadotu        v24, v6, v14, i8  \n\t"   \
      "vle32.v        v5, (t4)          \n\t"   \
      "addi           t4, t4, 4*128     \n\t"   \
                                                \
      "vmadotu        v26, v7, v14, i8  \n\t"   \
      "vle32.v        v12, (s5)         \n\t"   \
      "addi           s5, s5, 8*128     \n\t"   \
      "vmadotu        v28, v6, v15, i8  \n\t"   \
      "vle32.v        v13, (s6)         \n\t"   \
      "addi           s6, s6, 8*128     \n\t"   \
      "vmadotu        v30, v7, v15, i8  \n\t"   \
      "vle32.v        v14, (s7)         \n\t"   \
      "addi           s7, s7, 8*128     \n\t"   \

#define KERNEL16x32x16_M1                       \
      "vmadotu        v16, v2, v8, i8   \n\t"   \
      "vle32.v        v15, (s8)         \n\t"   \
      "addi           s8, s8, 8*128     \n\t"   \
                                                \
      "vmadotu        v18, v3, v8, i8   \n\t"   \
      "vmadotu        v20, v2, v9, i8   \n\t"   \
      "vmadotu        v22, v3, v9, i8   \n\t"   \
      "vmadotu        v24, v2, v10, i8  \n\t"   \
      "vmadotu        v26, v3, v10, i8  \n\t"   \
      "vle32.v        v8, (s1)          \n\t"   \
      "addi           s1, s1, 8*128     \n\t"   \
      "vmadotu        v28, v2, v11, i8  \n\t"   \
      "vle32.v        v9, (s2)          \n\t"   \
      "addi           s2, s2, 8*128     \n\t"   \
      "vmadotu        v30, v3, v11, i8  \n\t"   \
      "vle32.v        v10, (s3)         \n\t"   \
      "addi           s3, s3, 8*128     \n\t"   \

#define KERNEL16x32x16_E3                       \
      "vmadotu        v16, v2, v8, i8   \n\t"   \
      "vle32.v        v15, (s8)         \n\t"   \
      "addi           s8, s8, 8*128     \n\t"   \
      "vmadotu        v18, v3, v8, i8   \n\t"   \
      "vmadotu        v20, v2, v9, i8   \n\t"   \
      "vmadotu        v22, v3, v9, i8   \n\t"   \
      "vmadotu        v24, v2, v10, i8  \n\t"   \
      "vmadotu        v26, v3, v10, i8  \n\t"   \
      "vmadotu        v28, v2, v11, i8  \n\t"   \
      "vmadotu        v30, v3, v11, i8  \n\t"   \

#define KERNEL16x32x16_E4                       \
      "vmadotu        v16, v6, v12, i8  \n\t"   \
      "vmadotu        v18, v7, v12, i8  \n\t"   \
      "vmadotu        v20, v6, v13, i8  \n\t"   \
      "vmadotu        v22, v7, v13, i8  \n\t"   \
      "vmadotu        v24, v6, v14, i8  \n\t"   \
      "vmadotu        v26, v7, v14, i8  \n\t"   \
      "vmadotu        v28, v6, v15, i8  \n\t"   \
      "vmadotu        v30, v7, v15, i8  \n\t"   \

template <typename MLAS_GEMM_X8X8_KERNEL_IME2_TYPE>
size_t
MlasGemmQuantKernel_16x32_IME2(const typename MLAS_GEMM_X8X8_KERNEL_IME2_TYPE::PackedAType *A,
                             const typename MLAS_GEMM_X8X8_KERNEL_IME2_TYPE::PackedBType *B,
                             int32_t *C,
                             size_t PackedCountK,
                             size_t CountM,
                             size_t CountN,
                             size_t ldc,
                             const int32_t *RowSumBuffer,
                             const int32_t *ColumnSumBuffer,
                             const int32_t *ZeroPointB,
                             bool ZeroMode)
{
    int32_t Accumulator[MLAS_GEMM_X8X8_KERNEL_IME2_TYPE::Strides.M * MLAS_GEMM_X8X8_KERNEL_IME2_TYPE::PackedN];
    for (size_t n = 0; n < CountN; n += MLAS_GEMM_X8X8_KERNEL_IME2_TYPE::PackedN) {
        const size_t SubCountN = std::min(CountN - n, MLAS_GEMM_X8X8_KERNEL_IME2_TYPE::PackedN);
        size_t acc_stride = MLAS_GEMM_X8X8_KERNEL_IME2_TYPE::PackedN;
        __asm__ volatile(
            // load bias
            "vsetvli        t0, x0, e32, m1     \n\t"

            "vmv.v.x        v16, x0             \n\t"
            "vmv.v.x        v17, x0             \n\t"
            "vmv.v.x        v18, x0             \n\t"
            "vmv.v.x        v19, x0             \n\t"
            "vmv.v.x        v20, x0             \n\t"
            "vmv.v.x        v21, x0             \n\t"
            "vmv.v.x        v22, x0             \n\t"
            "vmv.v.x        v23, x0             \n\t"
            "vmv.v.x        v24, x0             \n\t"
            "vmv.v.x        v25, x0             \n\t"
            "vmv.v.x        v26, x0             \n\t"
            "vmv.v.x        v27, x0             \n\t"
            "vmv.v.x        v28, x0             \n\t"
            "vmv.v.x        v29, x0             \n\t"
            "vmv.v.x        v30, x0             \n\t"
            "vmv.v.x        v31, x0             \n\t"

            // bias ready

            // load w to s1-s8
            "addi           s1, %[b], 0         \n\t"
            "addi           s2, %[b], 128       \n\t"
            "addi           s3, %[b], 2*128     \n\t"
            "addi           s4, %[b], 3*128     \n\t"
            "addi           s5, %[b], 4*128     \n\t"
            "addi           s6, %[b], 5*128     \n\t"
            "addi           s7, %[b], 6*128     \n\t"
            "addi           s8, %[b], 7*128     \n\t"

            // load a to t1-t4
            "addi           t1, %[a], 0         \n\t"
            "addi           t2, %[a], 128       \n\t"
            "add            t3, %[a], 2*128     \n\t"
            "addi           t4, %[a], 3*128     \n\t"

            // K / 16 / 4
            "srli           t0, %[k], 2         \n\t"
            "blez           t0, M16x32x16_TAIL%=\n\t"

            // preloop
            KERNEL16x32x16_I

            "addi           t0, t0, -1          \n\t"
            "blez           t0, M16x32x16_MAINLOOP_TAIL%=\n\t"

            ".align 4                           \n\t"
            "M16x32x16_MAINLOOP%=:              \n\t"

            KERNEL16x32x16_M2
            KERNEL16x32x16_M3
            KERNEL16x32x16_M4
            KERNEL16x32x16_M1

            "addi           t0, t0, -1          \n\t"
            "bgtz           t0, M16x32x16_MAINLOOP%=\n\t"

            "M16x32x16_MAINLOOP_TAIL%=:         \n\t"

            KERNEL16x32x16_M2
            KERNEL16x32x16_E3
            KERNEL16x32x16_E4

            "M16x32x16_TAIL%=:                  \n\t"

            "andi           t0, %[k], 3         \n\t"
            "blez           t0, M16x32_SAVERESULT%=\n\t"

            ".align 4                           \n\t"
            "M16x32_TAILLOOP%=:                 \n\t"

            "vle32.v        v2, (t1)            \n\t"
            "addi           t1, t1, 2*128       \n\t"
            "vle32.v        v3, (t2)            \n\t"
            "addi           t2, t2, 2*128       \n\t"

            "vle32.v        v8, (s1)            \n\t"
            "addi           s1, s1, 4*128       \n\t"
            "vle32.v        v9, (s2)            \n\t"
            "addi           s2, s2, 4*128       \n\t"
            "vle32.v        v10, (s3)           \n\t"
            "addi           s3, s3, 4*128       \n\t"
            "vle32.v        v11, (s4)           \n\t"
            "addi           s4, s4, 4*128       \n\t"

            "vmadotu        v16, v2, v8, i8     \n\t"
            "vmadotu        v18, v3, v8, i8     \n\t"
            "vmadotu        v20, v2, v9, i8     \n\t"
            "vmadotu        v22, v3, v9, i8     \n\t"
            "vmadotu        v24, v2, v10, i8    \n\t"
            "vmadotu        v26, v3, v10, i8    \n\t"
            "vmadotu        v28, v2, v11, i8    \n\t"
            "vmadotu        v30, v3, v11, i8    \n\t"

            "addi           t0, t0, -1          \n\t"
            "bgtz           t0, M16x32_TAILLOOP%=\n\t"

            "M16x32_SAVERESULT%=:               \n\t"

            // tranform layout
            "vsetvli        t0, x0, e32, m1     \n\t"
            "vwadd.vx       v0, v16, x0         \n\t"
            "vwadd.vx       v2, v20, x0         \n\t"
            "vwadd.vx       v4, v24, x0         \n\t"
            "vwadd.vx       v6, v28, x0         \n\t"
            "vnpack.vv      v10, v0, v2, 3      \n\t"
            "vnpack.vv      v11, v4, v6, 3      \n\t"
            "vnpack.vv      v12, v1, v3, 3      \n\t"
            "vnpack.vv      v13, v5, v7, 3      \n\t"

            "vwadd.vx       v0, v17, x0         \n\t"
            "vwadd.vx       v2, v21, x0         \n\t"
            "vwadd.vx       v4, v25, x0         \n\t"
            "vwadd.vx       v6, v29, x0         \n\t"
            "vnpack.vv      v14, v0, v2, 3      \n\t"
            "vnpack.vv      v15, v4, v6, 3      \n\t"
            "vslidedown.vi  v0, v10, 16         \n\t"
            "vnpack.vv      v16, v1, v3, 3      \n\t"
            "vslidedown.vi  v2, v11, 16         \n\t"
            "vnpack.vv      v17, v5, v7, 3      \n\t"
            "vslidedown.vi  v1, v12, 16         \n\t"
            "vslidedown.vi  v3, v13, 16         \n\t"
            "vslidedown.vi  v4, v14, 16         \n\t"
            "vslidedown.vi  v5, v15, 16         \n\t"
            "vslidedown.vi  v6, v16, 16         \n\t"
            "vslidedown.vi  v7, v17, 16         \n\t"

            // save
            "slli           t3, %[ld_acc], 2    \n\t"
            "vsetvli        t0, x0, e32, mf2, tu, mu\n\t"
            "addi           t1, %[acc_buf], 64  \n\t"
            "add            t4, %[acc_buf], t3  \n\t"
            "vse32.v        v10, (%[acc_buf])   \n\t"
            "vse32.v        v11, (t1)           \n\t"
            "add            t1, t1, t3          \n\t"

            "vse32.v        v0, (t4)            \n\t"
            "vse32.v        v2, (t1)            \n\t"
            "add            t4, t4, t3          \n\t"
            "add            t1, t1, t3          \n\t"

            "vse32.v        v12, (t4)           \n\t"
            "vse32.v        v13, (t1)           \n\t"
            "add            t4, t4, t3          \n\t"
            "add            t1, t1, t3          \n\t"

            "vse32.v        v1, (t4)            \n\t"
            "vse32.v        v3, (t1)            \n\t"
            "add            t4, t4, t3          \n\t"
            "add            t1, t1, t3          \n\t"

            "vse32.v        v14, (t4)           \n\t"
            "vse32.v        v15, (t1)           \n\t"
            "add            t4, t4, t3          \n\t"
            "add            t1, t1, t3          \n\t"

            "vse32.v        v4, (t4)            \n\t"
            "vse32.v        v5, (t1)            \n\t"
            "add            t4, t4, t3          \n\t"
            "add            t1, t1, t3          \n\t"

            "vse32.v        v16, (t4)           \n\t"
            "vse32.v        v17, (t1)           \n\t"
            "add            t4, t4, t3          \n\t"
            "add            t1, t1, t3          \n\t"

            "vse32.v        v6, (t4)            \n\t"
            "vse32.v        v7, (t1)            \n\t"
            "add            t4, t4, t3          \n\t"
            "add            t1, t1, t3          \n\t"

            // tranform layout
            "vsetvli        t0, x0, e32, m1     \n\t"
            "vwadd.vx       v0, v18, x0         \n\t"
            "vwadd.vx       v2, v22, x0         \n\t"
            "vwadd.vx       v4, v26, x0         \n\t"
            "vwadd.vx       v6, v30, x0         \n\t"
            "vnpack.vv      v10, v0, v2, 3      \n\t"
            "vnpack.vv      v11, v4, v6, 3      \n\t"
            "vnpack.vv      v12, v1, v3, 3      \n\t"
            "vnpack.vv      v13, v5, v7, 3      \n\t"

            "vsetvli        t0, x0, e32, m1     \n\t"
            "vwadd.vx       v0, v19, x0         \n\t"
            "vwadd.vx       v2, v23, x0         \n\t"
            "vwadd.vx       v4, v27, x0         \n\t"
            "vwadd.vx       v6, v31, x0         \n\t"
            "vnpack.vv      v14, v0, v2, 3      \n\t"
            "vnpack.vv      v15, v4, v6, 3      \n\t"
            "vslidedown.vi  v0, v10, 16         \n\t"
            "vnpack.vv      v16, v1, v3, 3      \n\t"
            "vslidedown.vi  v2, v11, 16         \n\t"
            "vnpack.vv      v17, v5, v7, 3      \n\t"
            "vslidedown.vi  v1, v12, 16         \n\t"
            "vslidedown.vi  v3, v13, 16         \n\t"
            "vslidedown.vi  v4, v14, 16         \n\t"
            "vslidedown.vi  v5, v15, 16         \n\t"
            "vslidedown.vi  v6, v16, 16         \n\t"
            "vslidedown.vi  v7, v17, 16         \n\t"

            // save
            "vsetvli        t0, x0, e32, mf2, tu, mu\n\t"
            "vse32.v        v10, (t4)           \n\t"
            "vse32.v        v11, (t1)           \n\t"
            "add            t4, t4, t3          \n\t"
            "add            t1, t1, t3          \n\t"

            "vse32.v        v0, (t4)            \n\t"
            "vse32.v        v2, (t1)            \n\t"
            "add            t4, t4, t3          \n\t"
            "add            t1, t1, t3          \n\t"

            "vse32.v        v12, (t4)           \n\t"
            "vse32.v        v13, (t1)           \n\t"
            "add            t4, t4, t3          \n\t"
            "add            t1, t1, t3          \n\t"

            "vse32.v        v1, (t4)            \n\t"
            "vse32.v        v3, (t1)            \n\t"
            "add            t4, t4, t3          \n\t"
            "add            t1, t1, t3          \n\t"

            "vse32.v        v14, (t4)           \n\t"
            "vse32.v        v15, (t1)           \n\t"
            "add            t4, t4, t3          \n\t"
            "add            t1, t1, t3          \n\t"

            "vse32.v        v4, (t4)            \n\t"
            "vse32.v        v5, (t1)            \n\t"
            "add            t4, t4, t3          \n\t"
            "add            t1, t1, t3          \n\t"

            "vse32.v        v16, (t4)           \n\t"
            "vse32.v        v17, (t1)           \n\t"
            "add            t4, t4, t3          \n\t"
            "add            t1, t1, t3          \n\t"

            "vse32.v        v6, (t4)            \n\t"
            "vse32.v        v7, (t1)            \n\t"
            "add            t4, t4, t3          \n\t"
            "add            t1, t1, t3          \n\t"


            "QUIT%=:                            \n\t"
            :
            :[a] "r"(A), [b] "r"(B), [k] "r"(PackedCountK),
             [acc_buf] "r"(Accumulator), [ld_acc] "r"(acc_stride)
            :"cc", "t0", "t1", "t2", "t3", "t4", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8");

        for (size_t m = 0; m < CountM; m++) {
            size_t vl = __riscv_vsetvl_e32m4(SubCountN);
            vint32m4_t vdot_acc = __riscv_vle32_v_i32m4(Accumulator + m * acc_stride, vl);
            if (!ZeroMode) {
                vint32m4_t vsrc = __riscv_vle32_v_i32m4(C + m * ldc, vl);
                vdot_acc = __riscv_vadd_vv_i32m4(vdot_acc, vsrc, vl);
            }

            vint32m4_t col_sum = __riscv_vle32_v_i32m4(ColumnSumBuffer, vl);
            vint32m4_t row_sum = __riscv_vmv_v_x_i32m4(RowSumBuffer[m], vl);
            if (ZeroPointB != nullptr) {
                vint32m4_t zpb = __riscv_vle32_v_i32m4(ZeroPointB, vl);
                row_sum = __riscv_vmul_vv_i32m4(row_sum, zpb, vl);
            }
            vdot_acc = __riscv_vadd_vv_i32m4(vdot_acc, row_sum, vl);
            vdot_acc = __riscv_vadd_vv_i32m4(vdot_acc, col_sum, vl);
            __riscv_vse32_v_i32m4(C + m * ldc, vdot_acc, vl);
        }

        if (ZeroPointB != nullptr) {
            ZeroPointB += SubCountN;
        }
        ColumnSumBuffer += SubCountN;

        B += (SubCountN * MLAS_GEMM_X8X8_KERNEL_IME2_TYPE::PackedK * PackedCountK);
        C += SubCountN;
    }
    return CountM;
}
template <typename MLAS_GEMM_X8X8_KERNEL_IME2_TYPE>
size_t
MlasGemmQuantKernel_impl(const typename MLAS_GEMM_X8X8_KERNEL_IME2_TYPE::PackedAType *A,
                         const typename MLAS_GEMM_X8X8_KERNEL_IME2_TYPE::PackedBType *B,
                         int32_t *C,
                         size_t PackedCountK,
                         size_t CountM,
                         size_t CountN,
                         size_t ldc,
                         const int32_t *RowSumBuffer,
                         const int32_t *ColumnSumBuffer,
                         const int32_t *ZeroPointB,
                         bool ZeroMode)
{
    return MlasGemmQuantKernel_16x32_IME2<MLAS_GEMM_X8X8_KERNEL_IME2_TYPE>(
            A, B, C, PackedCountK, CountM, CountN, ldc, RowSumBuffer, ColumnSumBuffer, ZeroPointB, ZeroMode);
}

#define REGISTER_MlasGemmQuantKernel(TYPE)                                                               \
    template <>                                                                                          \
    size_t MlasGemmQuantKernel<TYPE>(const TYPE::PackedAType *A, const TYPE::PackedBType *B, int32_t *C, \
                                     size_t PackedCountK, size_t CountM, size_t CountN, size_t ldc,      \
                                     const int32_t *RowSumBuffer, const int32_t *ColumnSumBuffer,        \
                                     const int32_t *ZeroPointB, bool ZeroMode)                           \
    {                                                                                                    \
        return MlasGemmQuantKernel_impl<TYPE>(A, B, C, PackedCountK, CountM, CountN, ldc, RowSumBuffer,  \
                                              ColumnSumBuffer, ZeroPointB, ZeroMode);                    \
    }

#define REGISTER_MlasGemmQuantImpl(TYPE, NMAE)                                                                      \
    REGISTER_MlasGemmQuantCopyPackA(TYPE);                                                                          \
    REGISTER_MlasGemmQuantCopyPackB(TYPE);                                                                          \
    REGISTER_MlasGemmQuantKernel(TYPE);                                                                             \
    REGISTER_MlasGemmQuantFixupZeroPoint(TYPE);                                                                     \
    const MLAS_GEMM_QUANT_DISPATCH MlasGemmX8X8DispatchSpacemiTIme2_##NMAE = {MlasGemmQuantOperationIme<TYPE>,       \
                                                                             MlasGemmQuantPackedOperationIme<TYPE>, \
                                                                             MlasGemmQuantCopyPackB<TYPE>,          \
                                                                             TYPE::PackedK,                         \
                                                                             TYPE::Strides.K,                       \
                                                                             TYPE::Strides.M};

REGISTER_MlasGemmQuantImpl(MLAS_GEMM_X8X8_KERNEL_IME2, BASE);