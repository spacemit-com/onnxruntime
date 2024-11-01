/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qpostprocessor.cpp

Abstract:

    This module implements the post processor for QGEMM.

--*/

#include "mlasi.h"

void MLAS_QGEMM_SCALE_BIAS_OUTPUT_PROCESSOR::Process(
    const int32_t* C,
    size_t StartM,
    size_t StartN,
    size_t CountM,
    size_t CountN,
    size_t ldc
    ) const
{
    if (Bias_) {
        if (QuantGran_ == MLAS_QUANTIZATION_GRANULARITY::PerColumn) {
            if (OutputMode_ == MLAS_QGEMM_OUTPUT_MODE::AccumulateMode) {
                ProcessImpl<true, MLAS_QGEMM_OUTPUT_MODE::AccumulateMode, MLAS_QUANTIZATION_GRANULARITY::PerColumn>(
                    C,
                    StartM,
                    StartN,
                    CountM,
                    CountN,
                    ldc);
            } else {
                ProcessImpl<true, MLAS_QGEMM_OUTPUT_MODE::ZeroMode, MLAS_QUANTIZATION_GRANULARITY::PerColumn>(
                    C,
                    StartM,
                    StartN,
                    CountM,
                    CountN,
                    ldc);
            }
        } else if (OutputMode_ == MLAS_QGEMM_OUTPUT_MODE::AccumulateMode) {
            ProcessImpl<true, MLAS_QGEMM_OUTPUT_MODE::AccumulateMode, MLAS_QUANTIZATION_GRANULARITY::PerMatrix>(
                C,
                StartM,
                StartN,
                CountM,
                CountN,
                ldc);
        } else {
            ProcessImpl<true, MLAS_QGEMM_OUTPUT_MODE::ZeroMode, MLAS_QUANTIZATION_GRANULARITY::PerMatrix>(
                C,
                StartM,
                StartN,
                CountM,
                CountN,
                ldc);
        }
    } else {
        if (QuantGran_ == MLAS_QUANTIZATION_GRANULARITY::PerColumn) {
            if (OutputMode_ == MLAS_QGEMM_OUTPUT_MODE::AccumulateMode) {
                ProcessImpl<false, MLAS_QGEMM_OUTPUT_MODE::AccumulateMode, MLAS_QUANTIZATION_GRANULARITY::PerColumn>(
                    C,
                    StartM,
                    StartN,
                    CountM,
                    CountN,
                    ldc);
            } else {
                ProcessImpl<false, MLAS_QGEMM_OUTPUT_MODE::ZeroMode, MLAS_QUANTIZATION_GRANULARITY::PerColumn>(
                    C,
                    StartM,
                    StartN,
                    CountM,
                    CountN,
                    ldc);
            }
        } else if (OutputMode_ == MLAS_QGEMM_OUTPUT_MODE::AccumulateMode) {
            ProcessImpl<false, MLAS_QGEMM_OUTPUT_MODE::AccumulateMode, MLAS_QUANTIZATION_GRANULARITY::PerMatrix>(
                C,
                StartM,
                StartN,
                CountM,
                CountN,
                ldc);
        } else {
            ProcessImpl<false, MLAS_QGEMM_OUTPUT_MODE::ZeroMode, MLAS_QUANTIZATION_GRANULARITY::PerMatrix>(
                C,
                StartM,
                StartN,
                CountM,
                CountN,
                ldc);
        }
    }
}

template<bool HasBias, MLAS_QGEMM_OUTPUT_MODE Mode, MLAS_QUANTIZATION_GRANULARITY QuantGran>
inline
void
MLAS_QGEMM_SCALE_BIAS_OUTPUT_PROCESSOR::ProcessImpl(
    const int32_t* C,
    size_t StartM,
    size_t StartN,
    size_t CountM,
    size_t CountN,
    size_t ldc) const
/*++

Routine Description:

    This routine converts the output matrix C to a floating point format using
    the stored scale and bias parameters.

Arguments:

    C - Supplies the address of matrix C.

    StartM - Supplies the starting row offset relative to the matrix.

    StartN - Supplies the starting column offset relative to the matrix.

    CountM - Supplies the number of rows of the output matrix to process.

    CountN - Supplies the number of columns of the output matrix to process.

    ldc - Supplies the leading dimension of C.

Return Value:

    None.

--*/
{
    float* Output = Output_;
    const float* Bias = Bias_;
    const float* Scale = Scale_;

    if constexpr (HasBias) {
        Bias += StartN;
    }

    if constexpr (QuantGran == MLAS_QUANTIZATION_GRANULARITY::PerColumn){
        Scale += StartN;
    }

#if defined(MLAS_TARGET_RISCV64)
    C += StartM * ldc + StartN;
    Output += StartM * LeadingDimensionOutput_ + StartN;
    float ScaleValue = Scale_[0];
    while (CountM-- > 0) {

        float* c_out = Output;
        const int32_t* c = C;
        const float* bias = Bias;
        const float* scale = Scale;

        size_t n = CountN;

        for (size_t vl; n > 0; n -= vl, c += vl, c_out += vl) {
            vl = __riscv_vsetvl_e32m4(n);
            vfloat32m4_t ScaleVector = __riscv_vfmv_v_f_f32m4(ScaleValue, vl);
            vfloat32m4_t FloatVector = __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(c, vl), vl);

            if constexpr (QuantGran == MLAS_QUANTIZATION_GRANULARITY::PerColumn) {
                ScaleVector = __riscv_vle32_v_f32m4(scale, vl);
                scale += vl;
            }

            if constexpr (Mode == MLAS_QGEMM_OUTPUT_MODE::AccumulateMode) {
                FloatVector = __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(c_out, vl), FloatVector, ScaleVector, vl);
            } else {
                FloatVector = __riscv_vfmul_vv_f32m4(FloatVector, ScaleVector, vl);
            }

            if constexpr (HasBias) {
                FloatVector = __riscv_vfadd_vv_f32m4(FloatVector, __riscv_vle32_v_f32m4(bias, vl), vl);
                bias += vl;
            }

            __riscv_vse32_v_f32m4(c_out, FloatVector, vl);
        }

        C += ldc;
        Output += LeadingDimensionOutput_;
    }
#else
    MLAS_FLOAT32X4 ScaleVector = MlasBroadcastFloat32x4(Scale_);
#if !defined(MLAS_SSE2_INTRINSICS)
    float ScaleValue = MlasExtractLaneFloat32x4<0>(ScaleVector);
#endif

    C += StartM * ldc + StartN;
    Output += StartM * LeadingDimensionOutput_ + StartN;


    while (CountM-- > 0) {

        float* c_out = Output;
        const int32_t* c = C;
        const float* bias = Bias;
        const float* scale = Scale;

        size_t n = CountN;

        while (n >= 4) {

            MLAS_FLOAT32X4 FloatVector = MlasCastToFloat32x4(MlasLoadInt32x4(c));

            if (QuantGran == MLAS_QUANTIZATION_GRANULARITY::PerColumn) {
                ScaleVector = MlasLoadFloat32x4(scale);
                scale += 4;
            }

            if (Mode == MLAS_QGEMM_OUTPUT_MODE::AccumulateMode) {
                FloatVector = MlasMultiplyAddFloat32x4(FloatVector, ScaleVector, MlasLoadFloat32x4(c_out));
            } else {
                FloatVector = MlasMultiplyFloat32x4(FloatVector, ScaleVector);
            }

            if (HasBias) {
                FloatVector = MlasAddFloat32x4(FloatVector, MlasLoadFloat32x4(bias));
                bias += 4;
            }

            MlasStoreFloat32x4(c_out, FloatVector);

            c_out += 4;
            c += 4;
            n -= 4;
        }

        for (size_t offset = 0; offset < n; offset++) {

#if defined(MLAS_SSE2_INTRINSICS)
            __m128 FloatVector = _mm_set_ss(float(c[offset]));

            if (QuantGran == MLAS_QUANTIZATION_GRANULARITY::PerColumn) {
                ScaleVector = _mm_load_ss(&scale[offset]);
            }

            if (Mode == MLAS_QGEMM_OUTPUT_MODE::AccumulateMode) {
                FloatVector = _mm_add_ps(_mm_mul_ss(FloatVector, ScaleVector), _mm_load_ss(&c_out[offset]));
            } else {
                FloatVector = _mm_mul_ss(FloatVector, ScaleVector);
            }

            if (HasBias) {
                FloatVector = _mm_add_ss(FloatVector, _mm_load_ss(&bias[offset]));
            }

            _mm_store_ss(&c_out[offset], FloatVector);
#else
            if (QuantGran == MLAS_QUANTIZATION_GRANULARITY::PerColumn) {
                ScaleValue = scale[offset];
            }

            float result = float(c[offset]) * ScaleValue;
            if (HasBias) {
                result += bias[offset];
            }

            if (Mode == MLAS_QGEMM_OUTPUT_MODE::AccumulateMode) {
                c_out[offset] += result;
            } else {
                c_out[offset] = result;
            }
#endif
        }

        C += ldc;
        Output += LeadingDimensionOutput_;
    }
#endif
}
