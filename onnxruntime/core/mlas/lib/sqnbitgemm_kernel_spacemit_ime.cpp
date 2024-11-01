// Copyright (c) 2023 SpacemiT. All rights reserved.
// Licensed under the MIT License.

#include "sqnbitgemm_kernel_spacemit_ime.h"

#include <mlasi.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <utility>

#include "sqnbitgemm.h"
#include "sqnbitgemm_q8_block.h"

//
// Quantized B data packing function implementation.
//

namespace sqnbitgemm_spacemit_ime
{
size_t
SQ4BitGemmPackQuantBDataSize(
    size_t N, size_t K, size_t BlkLen, MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType, MLAS_SQNBIT_GEMM_SCALE_TYPE ScaleType)
{
    constexpr size_t BlkBitWidth = 4;
    constexpr size_t NBlockSize = 16;
    constexpr size_t SizeAlignment = 512;

    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    if (ComputeType == CompFp32) {
        const size_t PackedQuantBDataSize =
            MlasDivRoundup(N, NBlockSize) * NBlockSize * BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
        return MlasDivRoundup(PackedQuantBDataSize, SizeAlignment) * SizeAlignment;
    } else if (ComputeType == CompInt8) {
        assert(ScaleType == MLAS_SQNBIT_GEMM_SCALE_TYPE::ScaleFp32 ||
               ScaleType == MLAS_SQNBIT_GEMM_SCALE_TYPE::ScaleFp16);

        const size_t ScaleStride = ScaleType == MLAS_SQNBIT_GEMM_SCALE_TYPE::ScaleFp32 ? sizeof(float) : sizeof(__fp16);

        const size_t PackedQuantBSize =
            MlasDivRoundup(N, NBlockSize) * NBlockSize * BlockCountK *
            (MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen) + ScaleStride + sizeof(uint8_t));

        return MlasDivRoundup(PackedQuantBSize, SizeAlignment) * SizeAlignment;
    }
    return 0;
}

//
// Workspace size calculation function implementation.
//

size_t
SQ4BitGemmPerGemmWorkspaceSize(size_t M, size_t N, size_t K, size_t BlkLen, MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType)
{
    MLAS_UNREFERENCED_PARAMETER(N);

    switch (ComputeType) {
        case CompInt8: {
            // workspace buffer is used for block quantization of A to int8
            const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
            const size_t PerGemmWorkspaceSize = M * BlockCountK * Q8BlkSize(BlkLen);
            return PerGemmWorkspaceSize;
        }
        default: {
            return 0;
        }
    }
}

size_t
SQ4BitGemmPerGemmWorkspaceAlignment(size_t BlkLen, MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType)
{
    MLAS_UNREFERENCED_PARAMETER(BlkLen);

    switch (ComputeType) {
        case CompInt8: {
            return Q8BlkAlignment();
        }
        default: {
            return 1;
        }
    }
}

}  // namespace sqnbitgemm_spacemit_ime

//
// Kernel dispatch structure definition.
//

const MLAS_SQNBIT_GEMM_DISPATCH MlasSQNBitGemmDispatchSpacemiTIme = []() {
    MLAS_SQNBIT_GEMM_DISPATCH d;

    d.SQ4BitGemmPackQuantBDataSizeWithScale = sqnbitgemm_spacemit_ime::SQ4BitGemmPackQuantBDataSize;

    d.SQ4BitGemmPackQuantBData = sqnbitgemm_spacemit_ime::SQ4BitGemmPackQuantBData;
    d.SQ4BitGemmPackQuantBDataAndBlkSumWithScale = sqnbitgemm_spacemit_ime::SQ4BitGemmPackQuantBDataAndBlkSum;

    d.SQ4BitGemmPerGemmWorkspaceSize = sqnbitgemm_spacemit_ime::SQ4BitGemmPerGemmWorkspaceSize;
    d.SQ4BitGemmPerGemmWorkspaceAlignment = sqnbitgemm_spacemit_ime::SQ4BitGemmPerGemmWorkspaceAlignment;

    d.SQ4BitGemmM1Kernel_CompFp32 = sqnbitgemm_spacemit_ime::SQ4BitGemmM1Kernel_CompFp32;
    d.SQ4BitGemmMNKernel_CompFp32 = sqnbitgemm_spacemit_ime::SQ4BitGemmMNKernel_CompFp32;
    d.Q4BitBlkDequantBForSgemm_CompFp32 = sqnbitgemm_spacemit_ime::Q4BitBlkDequantBForSgemm_CompFp32;

    d.SQ4BitGemmKernel_CompInt8WithScale = sqnbitgemm_spacemit_ime::SQ4BitGemmKernel_CompInt8;
    d.QuantizeARow_CompInt8 = sqnbitgemm_spacemit_ime::QuantizeARow_CompInt8;

    return d;
}();
