// Copyright (c) 2023 SpacemiT. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "mlas_qnbit.h"
#include "mlasi.h"
#include "qnbitgemm.h"

namespace sqnbitgemm_spacemit_ime
{
size_t
SQ4BitGemmPackQuantBDataSize(size_t N,
                             size_t K,
                             size_t BlkLen,
                             MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType,
                             MLAS_QNBIT_GEMM_SCALE_TYPE ScaleType);

void
SQ4BitGemmPackQuantBData(size_t N,
                         size_t K,
                         size_t BlkLen,
                         MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType,
                         const std::byte* QuantBDataBegin,
                         std::byte* PackedQuantBDataBegin,
                         MLAS_THREADPOOL* ThreadPool);

void
SQ4BitGemmPackQuantBDataAndBlkSum(size_t N,
                                  size_t K,
                                  size_t BlkLen,
                                  MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType,
                                  MLAS_QNBIT_GEMM_SCALE_TYPE ScaleType,
                                  const std::byte* QuantBDataBegin,
                                  const float* QuantBScaleBegin,
                                  bool has_zp_input,
                                  const std::byte* QuantBZeroPointBegin,
                                  PackedQuantBDataStruct<float>& packed_quant_b,
                                  MLAS_THREADPOOL* ThreadPool);

size_t
SQ4BitGemmKernel_CompInt8(size_t BlkLen,
                          const std::byte* QuantA,
                          const std::byte* QuantBData,
                          const float* QuantBScale,
                          const std::byte* QuantBZeroPoint,
                          float* C,
                          size_t CountM,
                          size_t CountN,
                          size_t CountK,
                          size_t BlockCountK,
                          size_t ldc,
                          const float* Bias,
                          const size_t ScaleStride);

void
QuantizeAM4Row_CompInt8(size_t BlkLen, const float* A, size_t CountK, std::byte* QuantA);

void
QuantizeARow_CompInt8(size_t BlkLen, const float* A, size_t CountK, std::byte* QuantA);

void
SQ4BitBlkDequantBForSgemm_CompFp32(size_t BlkLen,
                                  float* FpData,
                                  const std::byte* QuantBData,
                                  const float* QuantBScale,
                                  const std::byte* QuantBZeroPoint,
                                  size_t CountN,
                                  size_t CountK,
                                  size_t BlockStrideQuantB);

size_t
SQ4BitGemmMNKernel_CompFp32(const float* A,
                            const float* B,
                            float* C,
                            size_t CountK,
                            size_t CountM,
                            size_t CountN,
                            size_t lda,
                            size_t ldc,
                            float alpha);

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
                            const float* Bias);

}  // namespace sqnbitgemm_spacemit_ime
