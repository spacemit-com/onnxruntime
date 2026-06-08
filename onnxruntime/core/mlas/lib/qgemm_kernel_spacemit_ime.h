// Copyright (c) 2023 SpacemiT. All rights reserved.
// Licensed under the MIT License.

#include <cstdlib>
#include <sstream>
#include <string>

#include "mlasi.h"
#include "qgemm.h"

template <typename KernelType>
MLAS_FORCEINLINE void
MlasGemmQuantThreadInitIme(size_t stride_n = 0, size_t stride_k = 0, size_t packed_stride_k = 0)
{
    const MLAS_GEMM_QUANT_STRIDES Strides = KernelType::Strides;
    const MLAS_GEMM_QUANT_STRIDES PackedStrides = KernelType::PackedStrides;

    const size_t cur_stride_n = stride_n > 0 ? Strides.N : stride_n;
    const size_t cur_stride_k = stride_k > 0 ? Strides.K : stride_k;
    const size_t cur_packed_stride_k = packed_stride_k > 0 ? PackedStrides.K : packed_stride_k;

    const size_t packASize = UpAlignSize(Strides.M * cur_stride_k * sizeof(typename KernelType::PackedAType));
    const size_t packBSize = UpAlignSize(cur_stride_n * cur_stride_k * sizeof(typename KernelType::PackedBType));
    const size_t rowSumSize = UpAlignSize(Strides.M * sizeof(int32_t));
    const size_t colSumSize = UpAlignSize(cur_stride_n * sizeof(int32_t));
    const size_t zpbSize = UpAlignSize(cur_stride_n * sizeof(int32_t));
    const size_t ctempSize = UpAlignSize(PackedStrides.M * PackedStrides.N * sizeof(int32_t));

    const size_t packedASize =
        UpAlignSize(PackedStrides.M * cur_packed_stride_k * sizeof(typename KernelType::PackedAType));

    const size_t bufsize = std::max(packASize + packBSize, packedASize + ctempSize) + rowSumSize + colSumSize + zpbSize;

    MlasThreadedBufAlloc(bufsize);
}

template <typename KernelType>
void
MlasGemmQuantOperationIme(const MLAS_GEMM_QUANT_SHAPE_PARAMS* Shape,
                          const MLAS_GEMM_QUANT_DATA_PARAMS* Data,
                          const size_t RangeStartM,
                          const size_t RangeCountM,
                          const size_t RangeStartN,
                          const size_t RangeCountN)
{
    constexpr MLAS_GEMM_QUANT_STRIDES Strides = KernelType::Strides;
    const size_t K = Shape->K;

    auto PackedStridesN = Strides.N;
    auto PackedStridesK = Strides.K;

    const size_t packASize = UpAlignSize(Strides.M * PackedStridesK * sizeof(typename KernelType::PackedAType));
    const size_t packBSize = UpAlignSize(PackedStridesN * PackedStridesK * sizeof(typename KernelType::PackedBType));
    const size_t rowSumSize = UpAlignSize(Strides.M * sizeof(int32_t));
    const size_t colSumSize = UpAlignSize(PackedStridesN * sizeof(int32_t));

    MlasGemmQuantThreadInitIme<KernelType>(PackedStridesN, PackedStridesK);

    uint8_t* p = ThreadedBufHolder.get();
    typename KernelType::PackedAType* PanelA = reinterpret_cast<typename KernelType::PackedAType*>(p);
    p += packASize;
    typename KernelType::PackedBType* PanelB = reinterpret_cast<typename KernelType::PackedBType*>(p);
    p += packBSize;
    int32_t* RowSumBuffer = reinterpret_cast<int32_t*>(p);
    p += rowSumSize;
    int32_t* ColumnSumBuffer = reinterpret_cast<int32_t*>(p);
    p += colSumSize;
    int32_t* ZeroPointBBuffer = reinterpret_cast<int32_t*>(p);

    const size_t lda = Data->lda;
    const size_t ldb = Data->ldb;
    const size_t ldc = Data->ldc;

    const uint8_t* A = Data->A + RangeStartM * lda;
    const uint8_t* B = (const uint8_t*)Data->B + RangeStartN;
    int32_t* C = Data->C + RangeStartM * ldc + RangeStartN;
    const uint8_t* PackedZeroPointB = Data->PerColumnZeroPoints ? Data->ZeroPointB + RangeStartN : nullptr;
    bool IsAccumulateMode = Shape->IsAccumulateMode;

    int32_t ZeroPointA = typename KernelType::OffsetAType(Data->ZeroPointA);
    int32_t ZeroPointB = typename KernelType::OffsetBType(*Data->ZeroPointB);

    //
    // Try to use a GEMV kernel if supported by this kernel type.
    //

    if ((RangeCountM == 1) && (ZeroPointA == 0) && (PackedZeroPointB == nullptr) && (ZeroPointB == 0) &&
        (Data->OutputProcessor == nullptr)) {
        if (MlasGemmQuantTryGemvKernel<KernelType>(A, B, ldb, C, K, RangeCountN, Shape->AIsSigned, Shape->BIsSigned)) {
            return;
        }
    }

    //
    // Fixup the sign bit of the per-matrix zero point offset of matrix A if the
    // kernel requires signed data.
    //

    ZeroPointA = MlasGemmQuantFixupZeroPointA<KernelType>(ZeroPointA, Shape->AIsSigned);

    //
    // Fixup the sign bit of the per-matrix zero point offset of matrix B if the
    // data is the opposite format of the kernel implementation. This value is
    // ignored if per-column zero point offsets are used instead.
    //

    ZeroPointB = MlasGemmQuantFixupZeroPointB<KernelType>(ZeroPointB, Shape->BIsSigned);

    //
    // Step through each slice of matrix B along the K dimension.
    //

    size_t CountK;

    for (size_t k = 0; k < K; k += CountK) {
        CountK = std::min(K - k, PackedStridesK);

        const size_t PackedCountK = (CountK + KernelType::PackedK - 1) / KernelType::PackedK;

        //
        // Step through each slice of matrix B along the N dimension.
        //

        size_t CountN;

        for (size_t n = 0; n < RangeCountN; n += CountN) {
            CountN = std::min(RangeCountN - n, PackedStridesN);

            //
            // Fixup the sign bit of the per-column zero point offsets of matrix B
            // if the data is the opposite format of the kernel implementation.
            //

            if (PackedZeroPointB != nullptr) {
                MlasGemmQuantFixupZeroPointB<KernelType>(PackedZeroPointB + n, ZeroPointBBuffer, CountN,
                                                         Shape->BIsSigned);
            }

            //
            // Copy a panel of matrix B to a local packed buffer.
            //

            MlasGemmQuantCopyPackB<KernelType>(PanelB, B + n, ldb, CountN, CountK, ColumnSumBuffer, Shape->BIsSigned);

            MlasGemmQuantScaleSumBuffer(ColumnSumBuffer, CountN, -ZeroPointA);

            //
            // Step through each slice of matrix A along the M dimension.
            //

            int32_t* c = C + n;
            size_t CountM;

            for (size_t m = 0; m < RangeCountM; m += CountM) {
                CountM = std::min(RangeCountM - m, Strides.M);

                //
                // Copy a panel of matrix A to a local packed buffer.
                //

                MlasGemmQuantCopyPackA<KernelType>(PanelA, A + m * lda, lda, CountM, CountK, RowSumBuffer,
                                                   Shape->AIsSigned);

                //
                // Apply the global depth value constant without the ZeroPointB scaling from:
                //
                //     (A[i] - ZeroPointA) * (B[i] - ZeroPointB)
                //              ==>
                //     A[i] * B[i] - A[i] * ZeroPointB - B[i] * ZeroPointA + ZeroPointA * ZeroPointB
                //
                // The ZeroPointB term is factored out and either applied below for per-matrix
                // quantization or inside the kernel for per-column quantization.
                //

                for (size_t mm = 0; mm < CountM; mm++) {
                    RowSumBuffer[mm] -= int32_t(CountK) * ZeroPointA;
                }

                //
                // Scale the row sums by the per-matrix zero point offset of matrix B.
                //

                if (PackedZeroPointB == nullptr) {
                    MlasGemmQuantScaleSumBuffer(RowSumBuffer, CountM, -ZeroPointB);
                }

                //
                // Step through the rows of the local packed buffer.
                //

                typename KernelType::PackedAType* pa = PanelA;
                int32_t* RowSums = RowSumBuffer;
                size_t RowsRemaining = CountM;

                bool ZeroMode = (k == 0) && !IsAccumulateMode;
                bool PostProcess = (k + CountK == K);

                while (RowsRemaining > 0) {
                    size_t RowsHandled = MlasGemmQuantKernel<KernelType>(
                        pa, PanelB, c, PackedCountK, RowsRemaining, CountN, ldc, RowSums, ColumnSumBuffer,
                        (PackedZeroPointB != nullptr) ? ZeroPointBBuffer : nullptr, ZeroMode);

                    if (PostProcess && Data->OutputProcessor != nullptr) {
                        Data->OutputProcessor->Process(Data->C, RangeStartM + m + CountM - RowsRemaining,
                                                       RangeStartN + n, RowsHandled, CountN, Data->ldc);
                    }

                    c += ldc * RowsHandled;
                    pa += KernelType::PackedK * PackedCountK * RowsHandled;
                    RowSums += RowsHandled;
                    RowsRemaining -= RowsHandled;
                }
            }
        }

        A += CountK;
        B += CountK * ldb;
    }
}

template <typename KernelType>
void
MlasGemmQuantPackedOperationIme(const MLAS_GEMM_QUANT_SHAPE_PARAMS* Shape,
                                const MLAS_GEMM_QUANT_DATA_PARAMS* Data,
                                const size_t RangeStartM,
                                const size_t RangeCountM,
                                const size_t RangeStartN,
                                const size_t RangeCountN)
{
    constexpr MLAS_GEMM_QUANT_STRIDES Strides = KernelType::PackedStrides;
    const size_t K = Shape->K;

    auto PackedStridesN = Strides.N;
    auto PackedStridesK = Strides.K;

    size_t packASize = UpAlignSize(Strides.M * PackedStridesK * sizeof(typename KernelType::PackedAType));
    size_t rowSumSize = UpAlignSize(Strides.M * sizeof(int32_t));
    size_t colSumSize = UpAlignSize(PackedStridesN * sizeof(int32_t));

    MlasGemmQuantThreadInitIme<KernelType>(PackedStridesN, PackedStridesK, PackedStridesK);

    uint8_t* p = ThreadedBufHolder.get();
    typename KernelType::PackedAType* PanelA = reinterpret_cast<typename KernelType::PackedAType*>(p);
    p += packASize;
    int32_t* RowSumBuffer = reinterpret_cast<int32_t*>(p);
    p += rowSumSize;
    int32_t* ColumnSumBuffer = reinterpret_cast<int32_t*>(p);
    p += colSumSize;
    int32_t* ZeroPointBBuffer = reinterpret_cast<int32_t*>(p);
    p += colSumSize;
    [[maybe_unused]] int32_t* C_temp = reinterpret_cast<int32_t*>(p);

    const size_t lda = Data->lda;
    const size_t ldc = Data->ldc;

    const uint8_t* A = Data->A + RangeStartM * lda;
    const uint8_t* PackedB = (const uint8_t*)Data->B;
    int32_t* C = Data->C + RangeStartM * ldc + RangeStartN;
    const uint8_t* PackedZeroPointB = Data->PerColumnZeroPoints ? Data->ZeroPointB + RangeStartN : nullptr;
    bool IsAccumulateMode = Shape->IsAccumulateMode;

    int32_t ZeroPointA = typename KernelType::OffsetAType(Data->ZeroPointA);
    int32_t ZeroPointB = typename KernelType::OffsetBType(*Data->ZeroPointB);

    //
    // Fixup the sign bit of the per-matrix zero point offset of matrix A if the
    // kernel requires signed data.
    //

    ZeroPointA = MlasGemmQuantFixupZeroPointA<KernelType>(ZeroPointA, Shape->AIsSigned);

    //
    // Fixup the sign bit of the per-matrix zero point offset of matrix B if the
    // data is the opposite format of the kernel implementation. This value is
    // ignored if per-column zero point offsets are used instead.
    //
    ZeroPointB = MlasGemmQuantFixupZeroPointB<KernelType>(ZeroPointB, Shape->BIsSigned);

    //
    // Extract the pointer to the column sum buffer from the packed matrix.
    //

    const size_t AlignedN = (Shape->N + MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1) & ~(MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1);
    const int32_t* PackedColumnSumBuffer = (const int32_t*)PackedB;
    PackedB = (const uint8_t*)(PackedColumnSumBuffer + AlignedN);
    PackedColumnSumBuffer += RangeStartN;

    //
    // Step through each slice of matrix B along the K dimension.
    //
    size_t CountM;
    size_t CountN;
    size_t CountK;

    for (size_t k = 0; k < K; k += CountK) {
        CountK = std::min(K - k, PackedStridesK);

        const size_t PackedCountK = (CountK + KernelType::PackedK - 1) / KernelType::PackedK;

        if (k > 0) {
            std::fill_n(ColumnSumBuffer, PackedStridesN, 0);
        }

        //
        // Step through each slice of matrix B along the N dimension.
        //

        for (size_t n = 0; n < RangeCountN; n += CountN) {
            CountN = std::min(RangeCountN - n, PackedStridesN);

            if (k == 0) {
                MlasGemmQuantScaleSumBuffer(ColumnSumBuffer, PackedColumnSumBuffer + n, CountN, -ZeroPointA);
            }

            //
            // Fixup the sign bit of the per-column zero point offsets of matrix B
            // if the data is the opposite format of the kernel implementation.
            //

            if (PackedZeroPointB != nullptr) {
                MlasGemmQuantFixupZeroPointB<KernelType>(PackedZeroPointB + n, ZeroPointBBuffer, CountN,
                                                         Shape->BIsSigned);
            }

            //
            // Step through each slice of matrix A along the M dimension.
            //
            const uint8_t* b = PackedB + (RangeStartN + n) * KernelType::PackedK * PackedCountK;

            bool ZeroMode = (k == 0) && !IsAccumulateMode;
            bool PostProcess = (k + CountK == K);
            size_t cur_ldc = ldc;
            int32_t* c_out = C + n;
            int32_t* c = c_out;

            for (size_t m = 0; m < RangeCountM; m += CountM) {
                CountM = std::min(RangeCountM - m, Strides.M);
                //
                // Copy a panel of matrix A to a local packed buffer.
                //

                MlasGemmQuantCopyPackA<KernelType>(PanelA, A + m * lda, lda, CountM, CountK, RowSumBuffer,
                                                   Shape->AIsSigned);

                //
                // Apply the global depth value constant without the ZeroPointB scaling from:
                //
                //     (A[i] - ZeroPointA) * (B[i] - ZeroPointB)
                //              ==>
                //     A[i] * B[i] - A[i] * ZeroPointB - B[i] * ZeroPointA + ZeroPointA * ZeroPointB
                //
                // The ZeroPointB term is factored out and either applied below for per-matrix
                // quantization or inside the kernel for per-column quantization.
                //

                for (size_t mm = 0; mm < CountM; mm++) {
                    RowSumBuffer[mm] -= int32_t(CountK) * ZeroPointA;
                }

                //
                // Scale the row sums by the per-matrix zero point offset of matrix B.
                //

                if (PackedZeroPointB == nullptr) {
                    MlasGemmQuantScaleSumBuffer(RowSumBuffer, CountM, -ZeroPointB);
                }

                //
                // Step through the rows of the local packed buffer.
                //

                typename KernelType::PackedAType* pa = PanelA;
                int32_t* RowSums = RowSumBuffer;
                size_t RowsRemaining = CountM;

                while (RowsRemaining > 0) {
                    size_t RowsHandled = MlasGemmQuantKernel<KernelType>(
                        pa, b, c, PackedCountK, RowsRemaining, CountN, cur_ldc, RowSums, ColumnSumBuffer,
                        (PackedZeroPointB != nullptr) ? ZeroPointBBuffer : nullptr, ZeroMode);

                    c += cur_ldc * RowsHandled;

                    if (PostProcess && Data->OutputProcessor != nullptr) {
                        Data->OutputProcessor->Process(Data->C, RangeStartM + m + CountM - RowsRemaining,
                                                       RangeStartN + n, RowsHandled, CountN, Data->ldc);
                    }
                    pa += KernelType::PackedK * PackedCountK * RowsHandled;
                    RowSums += RowsHandled;
                    RowsRemaining -= RowsHandled;
                }
            }
        }

        A += CountK;
        PackedB = (const uint8_t*)PackedB + AlignedN * CountK;
    }
}