// Copyright (c) 2023 SpacemiT. All rights reserved.
// Licensed under the MIT License.

#include <mlasi.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <utility>

#include "sqnbitgemm.h"
#include "sqnbitgemm_kernel_spacemit_ime.h"
#include "sqnbitgemm_q8_block.h"

//
// Quantized B data packing function implementation.
//

namespace sqnbitgemm_spacemit_ime
{
void
SQ4BitGemmPackQuantBData(size_t N,
                         size_t K,
                         size_t BlkLen,
                         MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType,
                         const std::byte* QuantBDataBegin,
                         std::byte* PackedQuantBDataBegin,
                         MLAS_THREADPOOL* ThreadPool)
{
    constexpr size_t BlkBitWidth = 4;

    assert(BlkLen >= 16 && BlkLen % 16 == 0);
    assert(ComputeType == MLAS_SQNBIT_GEMM_COMPUTE_TYPE::CompFp32);
    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    const size_t BlkDataSize = MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    const size_t Iterations = N * BlockCountK;  // one iteration per block

    const size_t SubBlkLen = 16;
    const size_t SubBlkDataSize = SubBlkLen / 2;
    const size_t SubBlkBytePairCount = SubBlkLen / 4;

    MlasTrySimpleParallel(ThreadPool, Iterations, [&](ptrdiff_t tid) {
        const size_t n = tid / BlockCountK;
        const size_t k_blk = tid % BlockCountK;

        const size_t data_offset = n * BlockCountK * BlkDataSize + k_blk * BlkDataSize;
        const std::byte* QuantBData = QuantBDataBegin + data_offset;
        std::byte* PackedQuantBData = PackedQuantBDataBegin + data_offset;

        for (size_t kk = 0; kk < BlkLen; kk += SubBlkLen) {
            for (size_t byte_pair_idx = 0; byte_pair_idx < SubBlkBytePairCount; ++byte_pair_idx) {
                const std::byte src0 = QuantBData[byte_pair_idx];
                const std::byte src1 = QuantBData[byte_pair_idx + SubBlkDataSize / 2];

                std::byte& dst0 = PackedQuantBData[2 * byte_pair_idx];
                std::byte& dst1 = PackedQuantBData[2 * byte_pair_idx + 1];

                dst0 = (src0 & std::byte{0x0F}) | ((src1 & std::byte{0x0F}) << 4);
                dst1 = (src0 >> 4) | ((src1 >> 4) << 4);
            }

            QuantBData += SubBlkDataSize;
            PackedQuantBData += SubBlkDataSize;
        }
    });
}

namespace
{
template <size_t NCols, bool HasZeroPoint>
MLAS_FORCEINLINE void
ComputeDotProducts_BlkBitWidth4_CompFp32(size_t BlkLen,
                                         const float* ARowPtr,
                                         const std::byte* QuantBDataColPtr,
                                         const float* QuantBScaleColPtr,
                                         const std::byte* QuantBZeroPointColPtr,
                                         float* SumPtr,
                                         size_t CountK,
                                         size_t StrideQuantBData,
                                         size_t StrideQuantBScale,
                                         size_t StrideQuantBZeroPoint,
                                         const float* BiasPtr)
{
    constexpr size_t BlkBitWidth = 4;

    constexpr size_t SubBlkLen = 16;
    static_assert(NCols == 1 || NCols == 4, "NCols must be 1 or 4");

    assert(BlkLen >= SubBlkLen && BlkLen % SubBlkLen == 0);

    for (size_t i = 0; i < NCols; i++) {
        const std::byte* QuantBData = QuantBDataColPtr;
        const float* QuantBScale = QuantBScaleColPtr;
        [[maybe_unused]] size_t QuantBZeroPointIdx = 0;
        vfloat32m1_t sum = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvlmax_e32m1());
        vfloat32m2_t acc = __riscv_vfmv_v_f_f32m2(0.0f, __riscv_vsetvlmax_e32m2());

        for (size_t k = 0; k < CountK; k += BlkLen) {
            const size_t k_blk_len = std::min(CountK - k, BlkLen);

            float scale = QuantBScale[i * StrideQuantBScale];

            [[maybe_unused]] float offset;
            if constexpr (HasZeroPoint) {
                const std::byte zp_packed = QuantBZeroPointColPtr[i * StrideQuantBZeroPoint + QuantBZeroPointIdx / 2];
                const std::byte zp = ((QuantBZeroPointIdx & 1) == 1) ? (zp_packed >> 4) : (zp_packed & std::byte{0x0F});
                offset = std::to_integer<uint8_t>(zp);
            }

            for (size_t k_idx_in_blk = 0; k_idx_in_blk < k_blk_len; k_idx_in_blk += SubBlkLen) {
                const size_t k_subblk_len = std::min(k_blk_len - k_idx_in_blk, SubBlkLen);

                size_t vl = __riscv_vsetvl_e32m2(k_subblk_len);
                vfloat32m2_t av = __riscv_vle32_v_f32m2(ARowPtr + k + k_idx_in_blk, vl);

                vl = __riscv_vsetvl_e8mf4(SubBlkLen / 2);
                const size_t b_data_block_offset = k_idx_in_blk * BlkBitWidth / 8;

                vuint8mf4_t bv_packed = __riscv_vle8_v_u8mf4(
                    reinterpret_cast<const uint8_t*>(QuantBData) + i * StrideQuantBData + b_data_block_offset, vl);

                vuint8mf4_t bv_low = __riscv_vand_vx_u8mf4(bv_packed, 0x0F, vl);
                vuint8mf4_t bv_high = __riscv_vsrl_vx_u8mf4(bv_packed, 0x04, vl);

                vuint16mf2_t bv_low_16 = __riscv_vwcvtu_x_x_v_u16mf2(bv_low, vl);
                vuint16mf2_t bv_high_16 = __riscv_vwcvtu_x_x_v_u16mf2(bv_high, vl);
                vfloat32m1_t bv_low_f32 = __riscv_vfwcvt_f_xu_v_f32m1(bv_low_16, vl);
                vfloat32m1_t bv_high_f32 = __riscv_vfwcvt_f_xu_v_f32m1(bv_high_16, vl);

                vl = __riscv_vsetvl_e32m2(k_subblk_len);
                vfloat32m2_t bv_f32 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
                bv_f32 = __riscv_vset_v_f32m1_f32m2(bv_f32, 0, bv_low_f32);
                bv_f32 = __riscv_vset_v_f32m1_f32m2(bv_f32, 1, bv_high_f32);

                if constexpr (HasZeroPoint) {
                    bv_f32 = __riscv_vfsub_vf_f32m2(bv_f32, offset, vl);
                } else {
                    bv_f32 = __riscv_vfsub_vf_f32m2(bv_f32, 8.0f, vl);
                }

                bv_f32 = __riscv_vfmul_vf_f32m2(bv_f32, scale, vl);

                acc = __riscv_vfmacc_vv_f32m2(acc, av, bv_f32, vl);
            }

            // increment pointers to next block
            QuantBData += MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
            QuantBScale += 1;
            if constexpr (HasZeroPoint) {
                QuantBZeroPointIdx += 1;
            }
        }
        // reduce and add bias

        size_t vl = std::min(SubBlkLen, CountK);

        sum = __riscv_vfredusum_vs_f32m2_f32m1(acc, sum, vl);
        SumPtr[i] = __riscv_vfmv_f(sum);
        if (BiasPtr) {
            SumPtr[i] += BiasPtr[i];
        }
    }
}

template <bool HasZeroPoint>
void
SQ4BitGemmM1Kernel_CompFp32_Impl(size_t BlkLen,
                                 const float* A,
                                 const std::byte* QuantBData,
                                 const float* QuantBScale,
                                 const std::byte* QuantBZeroPoint,
                                 float* C,
                                 size_t CountN,
                                 size_t CountK,
                                 size_t BlockStrideQuantB,
                                 const float* Bias)
{
    constexpr size_t BlkBitWidth = 4;
    constexpr size_t NCols = 4;

    const float* ARowPtr = A;
    float* CRowPtr = C;

    const size_t BlockCountK = BlockStrideQuantB;

    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    const size_t StrideQuantBScale = BlockCountK;
    const size_t StrideQuantBZeroPoint = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth>(BlockCountK);

    const float* BiasPtr = Bias;

    const std::byte* QuantBDataColPtr = QuantBData;
    const float* QuantBScaleColPtr = QuantBScale;
    const std::byte* QuantBZeroPointColPtr = QuantBZeroPoint;

    float* SumPtr = CRowPtr;

    int64_t nblk = static_cast<int64_t>(CountN) - NCols;

    while (nblk >= 0) {
        ComputeDotProducts_BlkBitWidth4_CompFp32<NCols, HasZeroPoint>(
            BlkLen, ARowPtr, QuantBDataColPtr, QuantBScaleColPtr, QuantBZeroPointColPtr, SumPtr, CountK,
            StrideQuantBData, StrideQuantBScale, StrideQuantBZeroPoint, BiasPtr);

        // move to next `NCols` columns

        QuantBDataColPtr += NCols * StrideQuantBData;
        QuantBScaleColPtr += NCols * StrideQuantBScale;
        if constexpr (HasZeroPoint) {
            QuantBZeroPointColPtr += NCols * StrideQuantBZeroPoint;
        }

        BiasPtr += BiasPtr != nullptr ? NCols : 0;
        SumPtr += NCols;

        nblk -= NCols;
    }

    // left over columns less than `NCols`?
    nblk += NCols;
    for (int64_t n = 0; n < nblk; ++n) {
        ComputeDotProducts_BlkBitWidth4_CompFp32<1, HasZeroPoint>(
            BlkLen, ARowPtr, QuantBDataColPtr, QuantBScaleColPtr, QuantBZeroPointColPtr, SumPtr, CountK,
            StrideQuantBData, StrideQuantBScale, StrideQuantBZeroPoint, BiasPtr);

        // move to next column

        QuantBDataColPtr += StrideQuantBData;
        QuantBScaleColPtr += StrideQuantBScale;
        if constexpr (HasZeroPoint) {
            QuantBZeroPointColPtr += StrideQuantBZeroPoint;
        }

        BiasPtr += BiasPtr != nullptr ? 1 : 0;
        SumPtr += 1;
    }
}

template <bool HasZeroPoint>
void
Q4BitBlkDequantBForSgemm_CompFp32_Impl(size_t BlkLen,
                                       float* FpData,
                                       const std::byte* QuantBData,
                                       const float* QuantBScale,
                                       const std::byte* QuantBZeroPoint,
                                       size_t CountN,
                                       size_t CountK,
                                       size_t BlockStrideQuantB)
{
    constexpr size_t BlkBitWidth = 4;
    auto* QuantBScaleCol = const_cast<float*>(QuantBScale);
    [[maybe_unused]] auto* QuantBZeroPointCol =
        const_cast<std::byte*>(QuantBZeroPoint);  // only used if HasZeroPoint is true

    const size_t StrideQuantBData = BlockStrideQuantB * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    [[maybe_unused]] const size_t StrideQuantBZeroPoint =  // only used if HasZeroPoint is true
        MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth>(BlockStrideQuantB);

    constexpr size_t SubBlkLen = 16;

    for (size_t n = 0; n < CountN; n++) {
        auto* QuantBDataColPtr = reinterpret_cast<const uint8_t*>(QuantBData) + n * StrideQuantBData;
        auto* Dst = FpData + n * CountK;
        for (size_t k = 0, k_blk_idx = 0; k < CountK; k += BlkLen, ++k_blk_idx) {
            const auto scale = QuantBScaleCol[n * BlockStrideQuantB + k_blk_idx];
            [[maybe_unused]] float offset = 0.0f;
            size_t kklen = std::min(CountK - k, BlkLen);
            if constexpr (HasZeroPoint) {
                const std::byte zp_packed = QuantBZeroPointCol[n * StrideQuantBZeroPoint + k_blk_idx / 2];
                const std::byte zp = ((k_blk_idx & 1) == 1) ? (zp_packed >> 4) : (zp_packed & std::byte{0x0F});
                offset = static_cast<float>(std::to_integer<uint8_t>(zp));
            }

            for (size_t k_idx_in_blk = 0; k_idx_in_blk < kklen; k_idx_in_blk += SubBlkLen) {
                const size_t k_subblk_len = std::min(kklen - k_idx_in_blk, SubBlkLen);
                size_t vl = __riscv_vsetvl_e8mf2(SubBlkLen / 2);
                // vl * 2 * 4bit
                // vlen min
                vuint8mf2_t bv_packed = __riscv_vle8_v_u8mf2(QuantBDataColPtr, vl);
                // 0 - (vl - 1)
                vuint8mf2_t bvl = __riscv_vand_vx_u8mf2(bv_packed, 0x0F, vl);
                // vl - (2 * vl - 1)
                vuint8mf2_t bvh = __riscv_vsrl_vx_u8mf2(bv_packed, 4, vl);
                vuint16m1_t bvl_16 = __riscv_vwcvtu_x_x_v_u16m1(bvl, vl);
                vuint16m1_t bvh_16 = __riscv_vwcvtu_x_x_v_u16m1(bvh, vl);
                vfloat32m2_t bvl_f = __riscv_vfwcvt_f_xu_v_f32m2(bvl_16, vl);
                vfloat32m2_t bvh_f = __riscv_vfwcvt_f_xu_v_f32m2(bvh_16, vl);

                if constexpr (HasZeroPoint) {
                    bvl_f = __riscv_vfsub_vf_f32m2(bvl_f, offset, vl);
                    bvh_f = __riscv_vfsub_vf_f32m2(bvh_f, offset, vl);
                } else {
                    bvl_f = __riscv_vfsub_vf_f32m2(bvl_f, 8.0f, vl);
                    bvh_f = __riscv_vfsub_vf_f32m2(bvh_f, 8.0f, vl);
                }
                bvl_f = __riscv_vfmul_vf_f32m2(bvl_f, scale, vl);
                bvh_f = __riscv_vfmul_vf_f32m2(bvh_f, scale, vl);
                
                if (k_subblk_len > SubBlkLen / 2) {
                    vl = __riscv_vsetvl_e32m2(SubBlkLen / 2);
                    __riscv_vse32_v_f32m2(Dst, bvl_f, vl);
                    vl = __riscv_vsetvl_e32m2(k_subblk_len - SubBlkLen / 2);
                    __riscv_vse32_v_f32m2(Dst + SubBlkLen / 2, bvh_f, vl);
                } else {
                    vl = __riscv_vsetvl_e32m2(k_subblk_len);
                    __riscv_vse32_v_f32m2(Dst, bvl_f, vl);
                }
                Dst += SubBlkLen;
                QuantBDataColPtr += SubBlkLen * BlkBitWidth / 8;
            }
        }
    }
}

}  // namespace

void
Q4BitBlkDequantBForSgemm_CompFp32(size_t BlkLen,
                                  float* FpData,
                                  const std::byte* QuantBData,
                                  const float* QuantBScale,
                                  const std::byte* QuantBZeroPoint,
                                  size_t CountN,
                                  size_t CountK,
                                  size_t BlockStrideQuantB)
{
    if (QuantBZeroPoint != nullptr) {
        Q4BitBlkDequantBForSgemm_CompFp32_Impl<true>(BlkLen, FpData, QuantBData, QuantBScale, QuantBZeroPoint, CountN,
                                                     CountK, BlockStrideQuantB);
    } else {
        Q4BitBlkDequantBForSgemm_CompFp32_Impl<false>(BlkLen, FpData, QuantBData, QuantBScale, QuantBZeroPoint, CountN,
                                                      CountK, BlockStrideQuantB);
    }
}

void
SQ4BitGemmM1Kernel_CompFp32(size_t BlkLen,
                            const float* A,
                            const std::byte* QuantBData,
                            const float* QuantBScale,
                            const std::byte* QuantBZeroPoint,
                            float* C,
                            size_t CountN,
                            size_t CountK,
                            size_t BlockStrideQuantB,
                            const float* Bias)
{
    if (QuantBZeroPoint != nullptr) {
        SQ4BitGemmM1Kernel_CompFp32_Impl<true>(BlkLen, A, QuantBData, QuantBScale, QuantBZeroPoint, C, CountN, CountK,
                                               BlockStrideQuantB, Bias);
    } else {
        SQ4BitGemmM1Kernel_CompFp32_Impl<false>(BlkLen, A, QuantBData, QuantBScale, QuantBZeroPoint, C, CountN, CountK,
                                                BlockStrideQuantB, Bias);
    }
}

size_t
SQ4BitGemmMNKernel_CompFp32(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountM,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha) {

    auto* Bptr = const_cast<float*>(B);
    if (CountM == 1) {
        for (size_t n = 0; n < CountN; n++) {
            int64_t k = CountK;
            auto* Aptr = const_cast<float*>(A);
            vfloat32m4_t vacc = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvlmax_e32m4());
            for (size_t vl; k > 0; k -= vl, Aptr += vl, Bptr += vl) {
                vl = __riscv_vsetvl_e32m4(k);
                vfloat32m4_t vA0 = __riscv_vle32_v_f32m4(Aptr, vl);
                vfloat32m4_t vB0 = __riscv_vle32_v_f32m4(Bptr, vl);
                vacc = __riscv_vfmacc_vv_f32m4(vacc, vA0, vB0, vl);
            }
            size_t vl = __riscv_vsetvl_e32m4(CountK);
            vfloat32m1_t vsum = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvlmax_e32m1());
            vsum = __riscv_vfredusum_vs_f32m4_f32m1(vacc, vsum, vl);
            C[n] = __riscv_vfmv_f(vsum);
        }
        return 1;
    } else {
        for (size_t n = 0; n < CountN; n++) {
            int64_t k = CountK;
            auto* Aptr = const_cast<float*>(A);
            vfloat32m4_t vacc0 = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvlmax_e32m4());
            vfloat32m4_t vacc1 = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvlmax_e32m4());
            for (size_t vl; k > 0; k -= vl, Aptr += vl, Bptr += vl) {
                vl = __riscv_vsetvl_e32m4(k);
                vfloat32m4_t vA0 = __riscv_vle32_v_f32m4(Aptr, vl);
                vfloat32m4_t vA1 = __riscv_vle32_v_f32m4(Aptr + lda, vl);
                vfloat32m4_t vB0 = __riscv_vle32_v_f32m4(Bptr, vl);
                vacc0 = __riscv_vfmacc_vv_f32m4(vacc0, vA0, vB0, vl);
                vacc1 = __riscv_vfmacc_vv_f32m4(vacc1, vA1, vB0, vl);
            }
            size_t vl = __riscv_vsetvl_e32m4(CountK);
            vfloat32m1_t vsum0 = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvlmax_e32m1());
            vfloat32m1_t vsum1 = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvlmax_e32m1());
            vsum0 = __riscv_vfredusum_vs_f32m4_f32m1(vacc0, vsum0, vl);
            vsum1 = __riscv_vfredusum_vs_f32m4_f32m1(vacc1, vsum1, vl);
            C[n] = __riscv_vfmv_f(vsum0);
            C[n + ldc] = __riscv_vfmv_f(vsum1);
        }
        return 2;
    }
}

}  // namespace sqnbitgemm_spacemit_ime