// Copyright (c) 2023 SpacemiT. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cassert>

#include "mlasi.h"
#include "qgemm.h"

struct MLAS_GEMM_X8X8_KERNEL_IME {
    typedef uint8_t PackedAType;
    typedef uint8_t PackedBType;
    typedef uint8_t OffsetAType;
    typedef uint8_t OffsetBType;

    static constexpr size_t PackedK = 8;
    static constexpr size_t PackedN = 16;

    static constexpr MLAS_GEMM_QUANT_STRIDES Strides{8, 128, 1024};
    static constexpr MLAS_GEMM_QUANT_STRIDES PackedStrides{8, 128, 1024};
};

template <typename MLAS_GEMM_X8X8_KERNEL_IME_TYPE>
MLAS_FORCEINLINE constexpr int32_t
MlasGemmQuantFixupZeroPointA_impl(int32_t ZeroPointA, bool AIsSigned)
{
    if (AIsSigned) {
        ZeroPointA = (uint8_t)(ZeroPointA ^ 0x80);
    }

    return ZeroPointA;
}

template <typename MLAS_GEMM_X8X8_KERNEL_IME_TYPE>
MLAS_FORCEINLINE constexpr int32_t
MlasGemmQuantFixupZeroPointB_impl(int32_t ZeroPointB, bool BIsSigned)
{
    if (BIsSigned) {
        ZeroPointB = typename MLAS_GEMM_X8X8_KERNEL_IME_TYPE::OffsetBType(ZeroPointB ^ 0x80);
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

template <typename MLAS_GEMM_X8X8_KERNEL_IME_TYPE>
void
MlasGemmQuantCopyPackA_impl(typename MLAS_GEMM_X8X8_KERNEL_IME_TYPE::PackedAType *D,
                            const uint8_t *A,
                            size_t lda,
                            size_t CountM,
                            size_t CountK,
                            int32_t *RowSumBuffer,
                            bool AIsSigned)
{
    const size_t AlignedCountK =
        (CountK + MLAS_GEMM_X8X8_KERNEL_IME_TYPE::PackedK - 1) & ~(MLAS_GEMM_X8X8_KERNEL_IME_TYPE::PackedK - 1);

    const uint8_t BitFlipValue = (AIsSigned ? 0x80 : 0);
    constexpr size_t Stride = MLAS_GEMM_X8X8_KERNEL_IME_TYPE::PackedK * MLAS_GEMM_X8X8_KERNEL_IME_TYPE::Strides.M;
    //
    // Process a single row of matrix A in a loop.
    //
    auto d = D;
    while (CountM-- > 0) {
        int32_t RowSum = 0;
        const uint8_t *a = A;
        __asm__ volatile(
            "vsetvli       t0,       x0,       e32,         m1\t\n"

            "vmv.v.x       v4,       zero                   \t\n"
            "addi          t1,       %[k],     0            \t\n"
            "addi          t2,       %[b],     0            \t\n"
            "srli          t3,       %[aligned_k], 3        \n\t"

            "LOOP%=:                                        \t\n"
            "vsetvli       t0,       t1,       e8,          m4\t\n"
            "vle8.v        v0,       (%[a])                 \t\n"
            "add           %[a],     %[a],     t0           \t\n"
            "vxor.vx       v0,       v0,       %[f]         \t\n"
            "vsetvli       t0,       t1,       e16,         m8\t\n"
            "vzext.vf2     v8,       v0                     \t\n"
            "vwredsumu.vs  v4,       v8,       v4           \t\n"
            "sub           t1,       t1,       t0           \t\n"

            "vsetvli       t0,       t3,       e64,         m4\t\n"
            "vsse64.v      v0,       (t2),     %[s]         \t\n"
            "mul           t0,       t0,       %[s]         \n\t"
            "add           t2,       t2,       t0           \n\t"

            "bnez          t1,       LOOP%=                 \t\n"
            "vmv.x.s       %[sum],   v4                     \t\n"
            : [ a ] "+r"(a), [ sum ] "+r"(RowSum)
            : [ k ] "r"(CountK), [ aligned_k ] "r"(AlignedCountK), [ f ] "r"(BitFlipValue), [ b ] "r"(D),
              [ s ] "r"(Stride)
            : "cc", "t0", "t1", "t2", "t3");
        *RowSumBuffer++ = RowSum;
        A += lda;
        D += MLAS_GEMM_X8X8_KERNEL_IME_TYPE::PackedK;
    }

    if (CountK & (MLAS_GEMM_X8X8_KERNEL_IME_TYPE::PackedK - 1)) {
        for (size_t k =
                 (AlignedCountK - MLAS_GEMM_X8X8_KERNEL_IME_TYPE::PackedK) * MLAS_GEMM_X8X8_KERNEL_IME_TYPE::Strides.M;
             k < AlignedCountK * MLAS_GEMM_X8X8_KERNEL_IME_TYPE::Strides.M;
             k += MLAS_GEMM_X8X8_KERNEL_IME_TYPE::PackedK) {
            for (size_t kk = CountK & (MLAS_GEMM_X8X8_KERNEL_IME_TYPE::PackedK - 1);
                 kk < MLAS_GEMM_X8X8_KERNEL_IME_TYPE::PackedK; kk++) {
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

template <typename MLAS_GEMM_X8X8_KERNEL_IME_TYPE>
void
MlasGemmQuantCopyPackB_impl(typename MLAS_GEMM_X8X8_KERNEL_IME_TYPE::PackedBType *D,
                            const uint8_t *B,
                            size_t ldb,
                            size_t CountN,
                            size_t CountK,
                            int32_t *ColumnSumBuffer,
                            bool BIsSigned)
{
    const uint8_t BitFlipValue = (BIsSigned ? 0x80 : 0);
    const size_t Stride = MLAS_GEMM_X8X8_KERNEL_IME_TYPE::PackedK * MLAS_GEMM_X8X8_KERNEL_IME_TYPE::PackedN;
    // Warmup
    auto b = B;
    auto StrideDiff = ldb - CountN;
    __asm__ volatile(
        "addi          t1,       %[k],     0            \t\n"
        "LOOPK%=:                                       \t\n"
        "addi          t1,       t1,       -1           \t\n"
        "addi          t2,       %[n],     0            \t\n"

        "LOOPN%=:                                       \t\n"
        "vsetvli       t0,       t2,       e8,          m4\t\n"
        "vle8.v        v0,       (%[b])                 \t\n"
        "add           %[b],     %[b],     t0           \t\n"
        "sub           t2,       t2,       t0           \t\n"
        "bnez          t2,       LOOPN%=                \t\n"
        "add           %[b],     %[b],     %[sd]        \t\n"
        "bnez          t1,       LOOPK%=                \t\n"
        : [ b ] "+r"(b)
        : [ k ] "r"(CountK), [ n ] "r"(CountN), [ sd ] "r"(StrideDiff)
        : "cc", "t0", "t1", "t2");
    auto d = D;
    size_t SubCountN = 0;
    for (size_t n = 0; n < CountN; n += SubCountN) {
        SubCountN = std::min(CountN - n, MLAS_GEMM_X8X8_KERNEL_IME_TYPE::PackedN);
        // init column sum
        for (size_t nn = 0; nn < SubCountN; nn++) {
            ColumnSumBuffer[n + nn] = 0;
        }
        // k loop
        size_t SubCountK = 0;
        size_t k;
        size_t CountKRndDown = CountK & (-MLAS_GEMM_X8X8_KERNEL_IME_TYPE::PackedK);
        for (k = 0; k < CountKRndDown; k += MLAS_GEMM_X8X8_KERNEL_IME_TYPE::PackedK) {
            for (size_t kk = 0; kk < MLAS_GEMM_X8X8_KERNEL_IME_TYPE::PackedK; kk++) {
                for (size_t nn = 0; nn < SubCountN; nn++) {
                    char val = B[(k + kk) * ldb + n + nn] ^ BitFlipValue;
                    d[nn * MLAS_GEMM_X8X8_KERNEL_IME_TYPE::PackedK + kk] = val;
                    ColumnSumBuffer[n + nn] += val;
                }
            }
            d += Stride;
        }
        SubCountK = CountK & (MLAS_GEMM_X8X8_KERNEL_IME_TYPE::PackedK - 1);
        if (SubCountK) {
            for (size_t kk = 0; kk < SubCountK; kk++) {
                for (size_t nn = 0; nn < SubCountN; nn++) {
                    char val = B[(k + kk) * ldb + n + nn] ^ BitFlipValue;
                    d[nn * MLAS_GEMM_X8X8_KERNEL_IME_TYPE::PackedK + kk] = val;
                    ColumnSumBuffer[n + nn] += val;
                }
            }
            for (size_t kk = SubCountK; kk < MLAS_GEMM_X8X8_KERNEL_IME_TYPE::PackedK; kk++) {
                for (size_t nn = 0; nn < SubCountN; nn++) {
                    d[nn * MLAS_GEMM_X8X8_KERNEL_IME_TYPE::PackedK + kk] = 0;
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

#define KERNEL8x8x16_I                                      \
    "vle32.v        v8,         (t1)                  \n\t" \
    "addi           t1,         t1,         4*4*8     \n\t" \
    "vle32.v        v9,         (t2)                  \n\t" \
    "addi           t2,         t2,         4*4*8     \n\t" \
    "vle32.v        v10,        (t3)                  \n\t" \
    "addi           t3,         t3,         4*4*8     \n\t" \
    "vle32.v        v11,        (t4)                  \n\t" \
    "addi           t4,         t4,         4*4*8     \n\t" \
                                                            \
    "vle32.v        v0,         (s1)                  \n\t" \
    "addi           s1,         s1,         8*4*8     \n\t" \
    "vle32.v        v1,         (s2)                  \n\t" \
    "addi           s2,         s2,         8*4*8     \n\t" \
    "vle32.v        v2,         (s3)                  \n\t" \
    "addi           s3,         s3,         8*4*8     \n\t" \
    "vle32.v        v3,         (s4)                  \n\t" \
    "addi           s4,         s4,         8*4*8     \n\t" \
    "vle32.v        v4,         (s5)                  \n\t" \
    "addi           s5,         s5,         8*4*8     \n\t" \
    "vle32.v        v5,         (s6)                  \n\t" \
    "addi           s6,         s6,         8*4*8     \n\t" \
    "vle32.v        v6,         (s7)                  \n\t" \
    "addi           s7,         s7,         8*4*8     \n\t" \
    "vle32.v        v7,         (s8)                  \n\t" \
    "addi           s8,         s8,         8*4*8     \n\t" \
                                                            \
    "vle32.v        v12,        (t1)                  \n\t" \
    "addi           t1,         t1,         4*4*8     \n\t" \
    "vle32.v        v13,        (t2)                  \n\t" \
    "addi           t2,         t2,         4*4*8     \n\t" \
    "vle32.v        v14,        (t3)                  \n\t" \
    "addi           t3,         t3,         4*4*8     \n\t" \
                                                            \
    "vmadotu        v16,        v8,         v0        \n\t" \
    "vmadotu        v24,        v9,         v0        \n\t" \
    "vle32.v        v15,        (t4)                  \n\t" \
    "addi           t4,         t4,         4*4*8     \n\t" \
    "vmadotu        v18,        v8,         v1        \n\t" \
    "vmadotu        v26,        v9,         v1        \n\t" \
    "vle32.v        v0,         (s1)                  \n\t" \
    "addi           s1,         s1,         8*4*8     \n\t" \
    "vmadotu        v20,        v8,         v2        \n\t" \
    "vmadotu        v28,        v9,         v2        \n\t" \
    "vle32.v        v1,         (s2)                  \n\t" \
    "addi           s2,         s2,         8*4*8     \n\t" \
    "vmadotu        v22,        v8,         v3        \n\t" \
    "vmadotu        v30,        v9,         v3        \n\t" \
    "vle32.v        v2,         (s3)                  \n\t" \
    "addi           s3,         s3,         8*4*8     \n\t"

#define KERNEL8x8x16_M1                                     \
    "vmadotu        v16,        v8,         v0        \n\t" \
    "vmadotu        v24,        v9,         v0        \n\t" \
    "vle32.v        v15,        (t4)                  \n\t" \
    "addi           t4,         t4,         4*4*8     \n\t" \
    "vmadotu        v18,        v8,         v1        \n\t" \
    "vmadotu        v26,        v9,         v1        \n\t" \
    "vle32.v        v0,         (s1)                  \n\t" \
    "addi           s1,         s1,         8*4*8     \n\t" \
    "vmadotu        v20,        v8,         v2        \n\t" \
    "vmadotu        v28,        v9,         v2        \n\t" \
    "vle32.v        v1,         (s2)                  \n\t" \
    "addi           s2,         s2,         8*4*8     \n\t" \
    "vmadotu        v22,        v8,         v3        \n\t" \
    "vmadotu        v30,        v9,         v3        \n\t" \
    "vle32.v        v2,         (s3)                  \n\t" \
    "addi           s3,         s3,         8*4*8     \n\t"

#define KERNEL8x8x16_M2                                     \
    "vmadotu        v16,        v10,        v4        \n\t" \
    "vmadotu        v24,        v11,        v4        \n\t" \
    "vle32.v        v3,         (s4)                  \n\t" \
    "addi           s4,         s4,         8*4*8     \n\t" \
    "vmadotu        v18,        v10,        v5        \n\t" \
    "vmadotu        v26,        v11,        v5        \n\t" \
    "vle32.v        v4,         (s5)                  \n\t" \
    "addi           s5,         s5,         8*4*8     \n\t" \
    "vmadotu        v20,        v10,        v6        \n\t" \
    "vmadotu        v28,        v11,        v6        \n\t" \
    "vle32.v        v5,         (s6)                  \n\t" \
    "addi           s6,         s6,         8*4*8     \n\t" \
    "vmadotu        v22,        v10,        v7        \n\t" \
    "vmadotu        v30,        v11,        v7        \n\t" \
    "vle32.v        v6,         (s7)                  \n\t" \
    "addi           s7,         s7,         8*4*8     \n\t" \
    "vle32.v        v7,         (s8)                  \n\t" \
    "addi           s8,         s8,         8*4*8     \n\t" \
    "vle32.v        v8,         (t1)                  \n\t" \
    "addi           t1,         t1,         4*4*8     \n\t" \
    "vle32.v        v9,         (t2)                  \n\t" \
    "addi           t2,         t2,         4*4*8     \n\t" \
    "vle32.v        v10,        (t3)                  \n\t" \
    "addi           t3,         t3,         4*4*8     \n\t"

#define KERNEL8x8x16_M3                                     \
    "vmadotu        v16,        v12,        v0        \n\t" \
    "vmadotu        v24,        v13,        v0        \n\t" \
    "vle32.v        v11,        (t4)                  \n\t" \
    "addi           t4,         t4,         4*4*8     \n\t" \
    "vmadotu        v18,        v12,        v1        \n\t" \
    "vmadotu        v26,        v13,        v1        \n\t" \
    "vle32.v        v0,         (s1)                  \n\t" \
    "addi           s1,         s1,         8*4*8     \n\t" \
    "vmadotu        v20,        v12,        v2        \n\t" \
    "vmadotu        v28,        v13,        v2        \n\t" \
    "vle32.v        v1,         (s2)                  \n\t" \
    "addi           s2,         s2,         8*4*8     \n\t" \
    "vmadotu        v22,        v12,        v3        \n\t" \
    "vmadotu        v30,        v13,        v3        \n\t" \
    "vle32.v        v2,         (s3)                  \n\t" \
    "addi           s3,         s3,         8*4*8     \n\t"

#define KERNEL8x8x16_M4                                     \
    "vmadotu        v16,        v14,        v4        \n\t" \
    "vmadotu        v24,        v15,        v4        \n\t" \
    "vle32.v        v3,         (s4)                  \n\t" \
    "addi           s4,         s4,         8*4*8     \n\t" \
    "vmadotu        v18,        v14,        v5        \n\t" \
    "vmadotu        v26,        v15,        v5        \n\t" \
    "vle32.v        v4,         (s5)                  \n\t" \
    "addi           s5,         s5,         8*4*8     \n\t" \
    "vmadotu        v20,        v14,        v6        \n\t" \
    "vmadotu        v28,        v15,        v6        \n\t" \
    "vle32.v        v5,         (s6)                  \n\t" \
    "addi           s6,         s6,         8*4*8     \n\t" \
    "vmadotu        v22,        v14,        v7        \n\t" \
    "vmadotu        v30,        v15,        v7        \n\t" \
    "vle32.v        v6,         (s7)                  \n\t" \
    "addi           s7,         s7,         8*4*8     \n\t" \
    "vle32.v        v7,         (s8)                  \n\t" \
    "addi           s8,         s8,         8*4*8     \n\t" \
                                                            \
    "vle32.v        v12,        (t1)                  \n\t" \
    "addi           t1,         t1,         4*4*8     \n\t" \
    "vle32.v        v13,        (t2)                  \n\t" \
    "addi           t2,         t2,         4*4*8     \n\t" \
    "vle32.v        v14,        (t3)                  \n\t" \
    "addi           t3,         t3,         4*4*8     \n\t"

#define KERNEL8x8x16_E2                                     \
    "vmadotu        v16,        v10,        v4        \n\t" \
    "vmadotu        v24,        v11,        v4        \n\t" \
    "vle32.v        v3,         (s4)                  \n\t" \
    "addi           s4,         s4,         8*4*8     \n\t" \
    "vmadotu        v18,        v10,        v5        \n\t" \
    "vmadotu        v26,        v11,        v5        \n\t" \
    "vle32.v        v4,         (s5)                  \n\t" \
    "addi           s5,         s5,         8*4*8     \n\t" \
    "vmadotu        v20,        v10,        v6        \n\t" \
    "vmadotu        v28,        v11,        v6        \n\t" \
    "vle32.v        v5,         (s6)                  \n\t" \
    "addi           s6,         s6,         8*4*8     \n\t" \
    "vmadotu        v22,        v10,        v7        \n\t" \
    "vmadotu        v30,        v11,        v7        \n\t" \
    "vle32.v        v6,         (s7)                  \n\t" \
    "addi           s7,         s7,         8*4*8     \n\t"

#define KERNEL8x8x16_E3                                     \
    "vmadotu        v16,        v12,        v0        \n\t" \
    "vmadotu        v24,        v13,        v0        \n\t" \
    "vle32.v        v7,         (s8)                  \n\t" \
    "addi           s8,         s8,         8*4*8     \n\t" \
    "vmadotu        v18,        v12,        v1        \n\t" \
    "vmadotu        v26,        v13,        v1        \n\t" \
                                                            \
    "vmadotu        v20,        v12,        v2        \n\t" \
    "vmadotu        v28,        v13,        v2        \n\t" \
                                                            \
    "vmadotu        v22,        v12,        v3        \n\t" \
    "vmadotu        v30,        v13,        v3        \n\t"

#define KERNEL8x8x16_E4                                     \
    "vmadotu        v16,        v14,        v4        \n\t" \
    "vmadotu        v24,        v15,        v4        \n\t" \
                                                            \
    "vmadotu        v18,        v14,        v5        \n\t" \
    "vmadotu        v26,        v15,        v5        \n\t" \
                                                            \
    "vmadotu        v20,        v14,        v6        \n\t" \
    "vmadotu        v28,        v15,        v6        \n\t" \
                                                            \
    "vmadotu        v22,        v14,        v7        \n\t" \
    "vmadotu        v30,        v15,        v7        \n\t"

template <typename MLAS_GEMM_X8X8_KERNEL_IME_TYPE>
size_t
MlasGemmQuantKernel_8x16_IME(const typename MLAS_GEMM_X8X8_KERNEL_IME_TYPE::PackedAType *A,
                             const typename MLAS_GEMM_X8X8_KERNEL_IME_TYPE::PackedBType *B,
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
    int32_t Accumulator[MLAS_GEMM_X8X8_KERNEL_IME_TYPE::Strides.M * MLAS_GEMM_X8X8_KERNEL_IME_TYPE::PackedN];
    for (size_t n = 0; n < CountN; n += MLAS_GEMM_X8X8_KERNEL_IME_TYPE::PackedN) {
        const size_t SubCountN = std::min(CountN - n, MLAS_GEMM_X8X8_KERNEL_IME_TYPE::PackedN);
        size_t acc_stride = MLAS_GEMM_X8X8_KERNEL_IME_TYPE::PackedN;
        __asm__ volatile(
            "vsetvli        t3,         zero,       e32,    m1, tu, mu\n\t"
            "vmv.v.x        v16,        zero                \n\t"
            "vmv.v.x        v17,        zero                \n\t"
            "vmv.v.x        v18,        zero                \n\t"
            "vmv.v.x        v19,        zero                \n\t"
            "vmv.v.x        v20,        zero                \n\t"
            "vmv.v.x        v21,        zero                \n\t"
            "vmv.v.x        v22,        zero                \n\t"
            "vmv.v.x        v23,        zero                \n\t"
            "vmv.v.x        v24,        zero                \n\t"
            "vmv.v.x        v25,        zero                \n\t"
            "vmv.v.x        v26,        zero                \n\t"
            "vmv.v.x        v27,        zero                \n\t"
            "vmv.v.x        v28,        zero                \n\t"
            "vmv.v.x        v29,        zero                \n\t"
            "vmv.v.x        v30,        zero                \n\t"
            "vmv.v.x        v31,        zero                \n\t"

            "mv             t1,         %[a]                \n\t"
            "addi           t2,         %[a],       4*8     \n\t"
            "add            t3,         %[a],       4*8*2   \n\t"
            "addi           t4,         %[a],       4*8*3   \n\t"

            "mv             s1,         %[b]                \n\t"
            "addi           s2,         %[b],       4*8     \n\t"
            "addi           s3,         %[b],       4*8*2   \n\t"
            "addi           s4,         %[b],       4*8*3   \n\t"
            "addi           s5,         %[b],       4*8*4   \n\t"
            "addi           s6,         %[b],       4*8*5   \n\t"
            "addi           s7,         %[b],       4*8*6   \n\t"
            "addi           s8,         %[b],       4*8*7   \n\t"

            "srli           t0,         %[k],       2       \n\t"
            "blez           t0,         M8x32x16_TAIL%=     \n\t"

            // preloop
            KERNEL8x8x16_I

            "addi           t0,       t0,           -1      \n\t"
            "blez           t0,       M8x32x16_MAINLOOP_TAIL%=\n\t"

            ".align 4                                       \n\t"
            "M8x32x16_MAINLOOP%=:                           \n\t"

            KERNEL8x8x16_M2 KERNEL8x8x16_M3 KERNEL8x8x16_M4 KERNEL8x8x16_M1

            "addi           t0,       t0,           -1      \n\t"
            "bgtz           t0,       M8x32x16_MAINLOOP%=   \n\t"

            "M8x32x16_MAINLOOP_TAIL%=:                      \n\t"

            KERNEL8x8x16_E2 KERNEL8x8x16_E3 KERNEL8x8x16_E4

            "M8x32x16_TAIL%=:                               \n\t"
            "andi           t0,       %[k],         3       \n\t"
            "blez           t0,       M8x16_SAVERESULT%=    \n\t"

            "vsetvli        t3,         zero,       e8,     m1, tu, mu\n\t"
            "M8x16_TAILLOOP%=:                              \n\t"
            "vle8.v         v8,         (t1)                \n\t"
            "addi           t1,         t1,         2*4*8   \n\t"
            "vle8.v         v9,         (t2)                \n\t"
            "addi           t2,         t2,         2*4*8   \n\t"

            "vle8.v         v0,         (s1)                \n\t"
            "addi           s1,         s1,         4*4*8   \n\t"
            "vle8.v         v1,         (s2)                \n\t"
            "addi           s2,         s2,         4*4*8   \n\t"
            "vle8.v         v2,         (s3)                \n\t"
            "addi           s3,         s3,         4*4*8   \n\t"
            "vle8.v         v3,         (s4)                \n\t"
            "addi           s4,         s4,         4*4*8   \n\t"

            "vmadotu        v16,        v8,         v0      \n\t"
            "vmadotu        v18,        v8,         v1      \n\t"
            "vmadotu        v20,        v8,         v2      \n\t"
            "vmadotu        v22,        v8,         v3      \n\t"
            "vmadotu        v24,        v9,         v0      \n\t"
            "vmadotu        v26,        v9,         v1      \n\t"
            "vmadotu        v28,        v9,         v2      \n\t"
            "vmadotu        v30,        v9,         v3      \n\t"

            "addi           t0,         t0,         -1      \n\t"
            "bgtz           t0,         M8x16_TAILLOOP%=    \n\t"

            "M8x16_SAVERESULT%=:                            \n\t"
            "vsetvli        t0,         zero,       e32,    m1, tu, mu\n\t"

            "vmv.v.i        v0,         15                  \n\t"

            "vmv.v.v        v1,         v20                 \n\t"
            "vmv.v.v        v2,         v18                 \n\t"
            "vmv.v.v        v3,         v22                 \n\t"
            "vmv.v.v        v4,         v17                 \n\t"
            "vmv.v.v        v5,         v21                 \n\t"
            "vmv.v.v        v6,         v19                 \n\t"
            "vmv.v.v        v7,         v23                 \n\t"

            "vmv.v.v        v8,         v24                 \n\t"
            "vmv.v.v        v9,         v28                 \n\t"
            "vmv.v.v        v10,        v26                 \n\t"
            "vmv.v.v        v11,        v30                 \n\t"
            "vmv.v.v        v12,        v25                 \n\t"
            "vmv.v.v        v13,        v29                 \n\t"
            "vmv.v.v        v14,        v27                 \n\t"
            "vmv.v.v        v15,        v31                 \n\t"

            "vslideup.vi    v1,         v22,        4       \n\t"
            "vslideup.vi    v4,         v19,        4       \n\t"
            "vslideup.vi    v5,         v23,        4       \n\t"
            "vslideup.vi    v8,         v26,        4       \n\t"
            "vslideup.vi    v9,         v30,        4       \n\t"
            "vslideup.vi    v12,        v27,        4       \n\t"
            "vslideup.vi    v13,        v31,        4       \n\t"

            "vslidedown.vi  v2,         v16,        4,      v0.t\n\t"
            "vslidedown.vi  v3,         v20,        4,      v0.t\n\t"
            "vslidedown.vi  v6,         v17,        4,      v0.t\n\t"
            "vslidedown.vi  v7,         v21,        4,      v0.t\n\t"
            "vslidedown.vi  v10,        v24,        4,      v0.t\n\t"
            "vslidedown.vi  v11,        v28,        4,      v0.t\n\t"
            "vslidedown.vi  v14,        v25,        4,      v0.t\n\t"
            "vslidedown.vi  v15,        v29,        4,      v0.t\n\t"

            "vmv.v.v        v0,         v16                 \n\t"
            "vslideup.vi    v0,         v18,        4       \n\t"

            "M8x16_SAVE%=:                                  \n\t"

            "slli           t3,         %[ld_acc],     2       \n\t"
            "vsetvli        t0,         %[nr],      e32,    m2, tu, mu\n\t"
            "vse32.v        v0,         (%[acc_buf])              \n\t"
            "addi           t1,         %[mr],      -1      \n\t"
            "add            t2,         %[acc_buf],       t3      \n\t"
            "blez           t1,         QUIT%=              \n\t"

            "vse32.v        v2,         (t2)                \n\t"
            "addi           t1,         t1,         -1      \n\t"
            "add            t2,         t2,         t3      \n\t"
            "blez           t1,         QUIT%=              \n\t"

            "vse32.v        v4,         (t2)                \n\t"
            "addi           t1,         t1,         -1      \n\t"
            "add            t2,         t2,         t3      \n\t"
            "blez           t1,         QUIT%=              \n\t"

            "vse32.v        v6,         (t2)                \n\t"
            "addi           t1,         t1,         -1      \n\t"
            "add            t2,         t2,         t3      \n\t"
            "blez           t1,         QUIT%=              \n\t"

            "vse32.v        v8,         (t2)                \n\t"
            "addi           t1,         t1,         -1      \n\t"
            "add            t2,         t2,         t3      \n\t"
            "blez           t1,         QUIT%=              \n\t"

            "vse32.v        v10,        (t2)                \n\t"
            "addi           t1,         t1,         -1      \n\t"
            "add            t2,         t2,         t3      \n\t"
            "blez           t1,         QUIT%=              \n\t"

            "vse32.v        v12,        (t2)                \n\t"
            "addi           t1,         t1,         -1      \n\t"
            "add            t2,         t2,         t3      \n\t"
            "blez           t1,         QUIT%=              \n\t"

            "vse32.v        v14,        (t2)                \n\t"

            "QUIT%=:                                        \n\t"
            : [ cb ] "+r"(ColumnSumBuffer), [ zp ] "+r"(ZeroPointB)
            : [ a ] "r"(A), [ b ] "r"(B), [ k ] "r"(PackedCountK), [ c ] "r"(C), [ ldc ] "r"(ldc),
              [ nr ] "r"(SubCountN), [ mr ] "r"(CountM), [ zeromode ] "r"(ZeroMode), [ rb ] "r"(RowSumBuffer),
              [ acc_buf ] "r"(Accumulator), [ ld_acc ] "r"(acc_stride)
            : "cc", "t0", "t1", "t2", "t3", "t4", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8");

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

        B += (SubCountN * MLAS_GEMM_X8X8_KERNEL_IME_TYPE::PackedK * PackedCountK);
        C += SubCountN;
    }
    return CountM;
}

#define KERNEL4x8x16_I                                      \
    "vle32.v        v8,         (t1)                  \n\t" \
    "addi           t1,         t1,         4*4*8     \n\t" \
    "vle32.v        v10,        (t3)                  \n\t" \
    "addi           t3,         t3,         4*4*8     \n\t" \
                                                            \
    "vle32.v        v0,         (s1)                  \n\t" \
    "addi           s1,         s1,         8*4*8     \n\t" \
    "vle32.v        v1,         (s2)                  \n\t" \
    "addi           s2,         s2,         8*4*8     \n\t" \
    "vle32.v        v2,         (s3)                  \n\t" \
    "addi           s3,         s3,         8*4*8     \n\t" \
    "vle32.v        v3,         (s4)                  \n\t" \
    "addi           s4,         s4,         8*4*8     \n\t" \
    "vle32.v        v4,         (s5)                  \n\t" \
    "addi           s5,         s5,         8*4*8     \n\t" \
    "vle32.v        v5,         (s6)                  \n\t" \
    "addi           s6,         s6,         8*4*8     \n\t" \
    "vle32.v        v6,         (s7)                  \n\t" \
    "addi           s7,         s7,         8*4*8     \n\t" \
    "vle32.v        v7,         (s8)                  \n\t" \
    "addi           s8,         s8,         8*4*8     \n\t" \
                                                            \
    "vle32.v        v12,        (t1)                  \n\t" \
    "addi           t1,         t1,         4*4*8     \n\t" \
    "vle32.v        v14,        (t3)                  \n\t" \
    "addi           t3,         t3,         4*4*8     \n\t" \
                                                            \
    "vmadotu        v16,        v8,         v0        \n\t" \
    "vmadotu        v18,        v8,         v1        \n\t" \
    "vle32.v        v0,         (s1)                  \n\t" \
    "addi           s1,         s1,         8*4*8     \n\t" \
    "vmadotu        v20,        v8,         v2        \n\t" \
    "vle32.v        v1,         (s2)                  \n\t" \
    "addi           s2,         s2,         8*4*8     \n\t" \
    "vmadotu        v22,        v8,         v3        \n\t" \
    "vle32.v        v2,         (s3)                  \n\t" \
    "addi           s3,         s3,         8*4*8     \n\t"

#define KERNEL4x8x16_M1                                     \
    "vmadotu        v16,        v8,         v0        \n\t" \
    "vmadotu        v18,        v8,         v1        \n\t" \
    "vle32.v        v0,         (s1)                  \n\t" \
    "addi           s1,         s1,         8*4*8     \n\t" \
    "vmadotu        v20,        v8,         v2        \n\t" \
    "vle32.v        v1,         (s2)                  \n\t" \
    "addi           s2,         s2,         8*4*8     \n\t" \
    "vmadotu        v22,        v8,         v3        \n\t" \
    "vle32.v        v2,         (s3)                  \n\t" \
    "addi           s3,         s3,         8*4*8     \n\t"

#define KERNEL4x8x16_M2                                     \
    "vmadotu        v16,        v10,        v4        \n\t" \
    "vle32.v        v3,         (s4)                  \n\t" \
    "addi           s4,         s4,         8*4*8     \n\t" \
    "vmadotu        v18,        v10,        v5        \n\t" \
    "vle32.v        v4,         (s5)                  \n\t" \
    "addi           s5,         s5,         8*4*8     \n\t" \
    "vmadotu        v20,        v10,        v6        \n\t" \
    "vle32.v        v5,         (s6)                  \n\t" \
    "addi           s6,         s6,         8*4*8     \n\t" \
    "vmadotu        v22,        v10,        v7        \n\t" \
    "vle32.v        v6,         (s7)                  \n\t" \
    "addi           s7,         s7,         8*4*8     \n\t" \
    "vle32.v        v7,         (s8)                  \n\t" \
    "addi           s8,         s8,         8*4*8     \n\t" \
    "vle32.v        v8,         (t1)                  \n\t" \
    "addi           t1,         t1,         4*4*8     \n\t" \
    "vle32.v        v10,        (t3)                  \n\t" \
    "addi           t3,         t3,         4*4*8     \n\t"

#define KERNEL4x8x16_M3                                     \
    "vmadotu        v16,        v12,        v0        \n\t" \
    "vmadotu        v18,        v12,        v1        \n\t" \
    "vle32.v        v0,         (s1)                  \n\t" \
    "addi           s1,         s1,         8*4*8     \n\t" \
    "vmadotu        v20,        v12,        v2        \n\t" \
    "vle32.v        v1,         (s2)                  \n\t" \
    "addi           s2,         s2,         8*4*8     \n\t" \
    "vmadotu        v22,        v12,        v3        \n\t" \
    "vle32.v        v2,         (s3)                  \n\t" \
    "addi           s3,         s3,         8*4*8     \n\t"

#define KERNEL4x8x16_M4                                     \
    "vmadotu        v16,        v14,        v4        \n\t" \
    "vle32.v        v3,         (s4)                  \n\t" \
    "addi           s4,         s4,         8*4*8     \n\t" \
    "vmadotu        v18,        v14,        v5        \n\t" \
    "vle32.v        v4,         (s5)                  \n\t" \
    "addi           s5,         s5,         8*4*8     \n\t" \
    "vmadotu        v20,        v14,        v6        \n\t" \
    "vle32.v        v5,         (s6)                  \n\t" \
    "addi           s6,         s6,         8*4*8     \n\t" \
    "vmadotu        v22,        v14,        v7        \n\t" \
    "vle32.v        v6,         (s7)                  \n\t" \
    "addi           s7,         s7,         8*4*8     \n\t" \
    "vle32.v        v7,         (s8)                  \n\t" \
    "addi           s8,         s8,         8*4*8     \n\t" \
                                                            \
    "vle32.v        v12,        (t1)                  \n\t" \
    "addi           t1,         t1,         4*4*8     \n\t" \
    "vle32.v        v14,        (t3)                  \n\t" \
    "addi           t3,         t3,         4*4*8     \n\t"

#define KERNEL4x8x16_E2                                     \
    "vmadotu        v16,        v10,        v4        \n\t" \
    "vle32.v        v3,         (s4)                  \n\t" \
    "addi           s4,         s4,         8*4*8     \n\t" \
    "vmadotu        v18,        v10,        v5        \n\t" \
    "vle32.v        v4,         (s5)                  \n\t" \
    "addi           s5,         s5,         8*4*8     \n\t" \
    "vmadotu        v20,        v10,        v6        \n\t" \
    "vle32.v        v5,         (s6)                  \n\t" \
    "addi           s6,         s6,         8*4*8     \n\t" \
    "vmadotu        v22,        v10,        v7        \n\t" \
    "vle32.v        v6,         (s7)                  \n\t" \
    "addi           s7,         s7,         8*4*8     \n\t"

#define KERNEL4x8x16_E3                                     \
    "vmadotu        v16,        v12,        v0        \n\t" \
    "vle32.v        v7,         (s8)                  \n\t" \
    "addi           s8,         s8,         8*4*8     \n\t" \
    "vmadotu        v18,        v12,        v1        \n\t" \
                                                            \
    "vmadotu        v20,        v12,        v2        \n\t" \
                                                            \
    "vmadotu        v22,        v12,        v3        \n\t"

#define KERNEL4x8x16_E4                                     \
    "vmadotu        v16,        v14,        v4        \n\t" \
                                                            \
    "vmadotu        v18,        v14,        v5        \n\t" \
                                                            \
    "vmadotu        v20,        v14,        v6        \n\t" \
                                                            \
    "vmadotu        v22,        v14,        v7        \n\t"

template <typename MLAS_GEMM_X8X8_KERNEL_IME_TYPE>
size_t
MlasGemmQuantKernel_4x16_IME(const typename MLAS_GEMM_X8X8_KERNEL_IME_TYPE::PackedAType *A,
                             const typename MLAS_GEMM_X8X8_KERNEL_IME_TYPE::PackedBType *B,
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
    int32_t Accumulator[MLAS_GEMM_X8X8_KERNEL_IME_TYPE::Strides.M * MLAS_GEMM_X8X8_KERNEL_IME_TYPE::PackedN];
    for (size_t n = 0; n < CountN; n += MLAS_GEMM_X8X8_KERNEL_IME_TYPE::PackedN) {
        const size_t SubCountN = std::min(CountN - n, MLAS_GEMM_X8X8_KERNEL_IME_TYPE::PackedN);
        size_t acc_stride = MLAS_GEMM_X8X8_KERNEL_IME_TYPE::PackedN;
        __asm__ volatile(
            "vsetvli        t3,         zero,       e32,    m1, tu, mu\n\t"
            "vmv.v.x        v16,        zero                \n\t"
            "vmv.v.x        v17,        zero                \n\t"
            "vmv.v.x        v18,        zero                \n\t"
            "vmv.v.x        v19,        zero                \n\t"
            "vmv.v.x        v20,        zero                \n\t"
            "vmv.v.x        v21,        zero                \n\t"
            "vmv.v.x        v22,        zero                \n\t"
            "vmv.v.x        v23,        zero                \n\t"

            "mv             t1,         %[a]                \n\t"
            "addi           t2,         %[a],       4*8     \n\t"
            "add            t3,         %[a],       4*8*2   \n\t"
            "addi           t4,         %[a],       4*8*3   \n\t"

            "mv             s1,         %[b]                \n\t"
            "addi           s2,         %[b],       4*8     \n\t"
            "addi           s3,         %[b],       4*8*2   \n\t"
            "addi           s4,         %[b],       4*8*3   \n\t"
            "addi           s5,         %[b],       4*8*4   \n\t"
            "addi           s6,         %[b],       4*8*5   \n\t"
            "addi           s7,         %[b],       4*8*6   \n\t"
            "addi           s8,         %[b],       4*8*7   \n\t"

            "srli           t0,         %[k],       2       \n\t"
            "blez           t0,         M4x32x16_TAIL%=     \n\t"

            // preloop
            KERNEL4x8x16_I

            "addi           t0,       t0,           -1      \n\t"
            "blez           t0,       M4x32x16_MAINLOOP_TAIL%=\n\t"

            ".align 4                                       \n\t"
            "M4x32x16_MAINLOOP%=:                           \n\t"

            KERNEL4x8x16_M2 KERNEL4x8x16_M3 KERNEL4x8x16_M4 KERNEL4x8x16_M1

            "addi           t0,       t0,           -1      \n\t"
            "bgtz           t0,       M4x32x16_MAINLOOP%=   \n\t"

            "M4x32x16_MAINLOOP_TAIL%=:                      \n\t"

            KERNEL4x8x16_E2 KERNEL4x8x16_E3 KERNEL4x8x16_E4

            "M4x32x16_TAIL%=:                               \n\t"
            "andi           t0,       %[k],         3       \n\t"
            "blez           t0,       M4x16_SAVERESULT%=    \n\t"

            "vsetvli        t3,         zero,       e8,     m1, tu, mu\n\t"
            "M4x16_TAILLOOP%=:                              \n\t"
            "vle8.v         v8,         (t1)                \n\t"
            "addi           t1,         t1,         2*4*8   \n\t"

            "vle8.v         v0,         (s1)                \n\t"
            "addi           s1,         s1,         4*4*8   \n\t"
            "vle8.v         v1,         (s2)                \n\t"
            "addi           s2,         s2,         4*4*8   \n\t"
            "vle8.v         v2,         (s3)                \n\t"
            "addi           s3,         s3,         4*4*8   \n\t"
            "vle8.v         v3,         (s4)                \n\t"
            "addi           s4,         s4,         4*4*8   \n\t"

            "vmadotu        v16,        v8,         v0      \n\t"
            "vmadotu        v18,        v8,         v1      \n\t"
            "vmadotu        v20,        v8,         v2      \n\t"
            "vmadotu        v22,        v8,         v3      \n\t"

            "addi           t0,         t0,         -1      \n\t"
            "bgtz           t0,         M4x16_TAILLOOP%=    \n\t"

            "M4x16_SAVERESULT%=:                            \n\t"
            "vsetvli        t0,         zero,       e32,    m1, tu, mu\n\t"

            "vmv.v.i        v0,         15                  \n\t"
            "vmv.v.v        v1,         v20                 \n\t"
            "vmv.v.v        v2,         v18                 \n\t"
            "vmv.v.v        v3,         v22                 \n\t"
            "vmv.v.v        v4,         v17                 \n\t"
            "vmv.v.v        v5,         v21                 \n\t"
            "vmv.v.v        v6,         v19                 \n\t"
            "vmv.v.v        v7,         v23                 \n\t"

            "vslideup.vi    v1,         v22,        4       \n\t"
            "vslideup.vi    v4,         v19,        4       \n\t"
            "vslideup.vi    v5,         v23,        4       \n\t"

            "vslidedown.vi  v2,         v16,        4,      v0.t\n\t"
            "vslidedown.vi  v3,         v20,        4,      v0.t\n\t"
            "vslidedown.vi  v6,         v17,        4,      v0.t\n\t"
            "vslidedown.vi  v7,         v21,        4,      v0.t\n\t"

            "vmv.v.v        v0,         v16                 \n\t"
            "vslideup.vi    v0,         v18,        4       \n\t"

            "M4x16_SAVE%=:                                  \n\t"

            "slli           t3,         %[ld_acc],     2       \n\t"
            "vsetvli        t0,         %[nr],      e32,    m2, tu, mu\n\t"
            "vse32.v        v0,         (%[acc_buf])              \n\t"
            "addi           t1,         %[mr],      -1      \n\t"
            "add            t2,         %[acc_buf],       t3      \n\t"
            "blez           t1,         QUIT%=              \n\t"

            "vse32.v        v2,         (t2)                \n\t"
            "addi           t1,         t1,         -1      \n\t"
            "add            t2,         t2,         t3      \n\t"
            "blez           t1,         QUIT%=              \n\t"

            "vse32.v        v4,         (t2)                \n\t"
            "addi           t1,         t1,         -1      \n\t"
            "add            t2,         t2,         t3      \n\t"
            "blez           t1,         QUIT%=              \n\t"

            "vse32.v        v6,         (t2)                \n\t"

            "QUIT%=:                                        \n\t"
            : [ cb ] "+r"(ColumnSumBuffer), [ zp ] "+r"(ZeroPointB)
            : [ a ] "r"(A), [ b ] "r"(B), [ k ] "r"(PackedCountK), [ c ] "r"(C), [ ldc ] "r"(ldc),
              [ nr ] "r"(SubCountN), [ mr ] "r"(CountM), [ zeromode ] "r"(ZeroMode), [ rb ] "r"(RowSumBuffer),
              [ acc_buf ] "r"(Accumulator), [ ld_acc ] "r"(acc_stride)
            : "cc", "t0", "t1", "t2", "t3", "t4", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8");

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

        B += (SubCountN * MLAS_GEMM_X8X8_KERNEL_IME::PackedK * PackedCountK);
        C += SubCountN;
    }
    return CountM;
}

template <typename MLAS_GEMM_X8X8_KERNEL_IME_TYPE>
size_t
MlasGemmQuantKernel_impl(const typename MLAS_GEMM_X8X8_KERNEL_IME_TYPE::PackedAType *A,
                         const typename MLAS_GEMM_X8X8_KERNEL_IME_TYPE::PackedBType *B,
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
    if (CountM > 4) {
        return MlasGemmQuantKernel_8x16_IME<MLAS_GEMM_X8X8_KERNEL_IME_TYPE>(
            A, B, C, PackedCountK, CountM, CountN, ldc, RowSumBuffer, ColumnSumBuffer, ZeroPointB, ZeroMode);
    } else {
        return MlasGemmQuantKernel_4x16_IME<MLAS_GEMM_X8X8_KERNEL_IME_TYPE>(
            A, B, C, PackedCountK, CountM, CountN, ldc, RowSumBuffer, ColumnSumBuffer, ZeroPointB, ZeroMode);
    }
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

#define REGISTER_MlasGemmQuantImpl(TYPE, NMAE)                                                                   \
    REGISTER_MlasGemmQuantCopyPackA(TYPE);                                                                       \
    REGISTER_MlasGemmQuantCopyPackB(TYPE);                                                                       \
    REGISTER_MlasGemmQuantKernel(TYPE);                                                                          \
    REGISTER_MlasGemmQuantFixupZeroPoint(TYPE);                                                                  \
    const MLAS_GEMM_QUANT_DISPATCH MlasGemmX8X8DispatchSpacemiTIme_##NMAE = {MlasGemmQuantOperation<TYPE>,       \
                                                                             MlasGemmQuantPackedOperation<TYPE>, \
                                                                             MlasGemmQuantCopyPackB<TYPE>,       \
                                                                             TYPE::PackedK,                      \
                                                                             TYPE::Strides.K,                    \
                                                                             TYPE::Strides.M};

REGISTER_MlasGemmQuantImpl(MLAS_GEMM_X8X8_KERNEL_IME, BASE);
