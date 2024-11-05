// Copyright (c) 2023 SpacemiT. All rights reserved.
// Licensed under the MIT License.

#include "mlasi.h"

float
MlasReduceMaximumF32Kernel_RVV(const float* Input, const size_t D)
{
    float* src = const_cast<float*>(reinterpret_cast<const float*>(Input));
    float Maximum = std::numeric_limits<float>::lowest();
    int64_t n = D;
    size_t vl;
    vfloat32m8_t vmaxf = __riscv_vfmv_v_f_f32m8(Maximum, __riscv_vsetvlmax_e32m8());
    for (; n > 0; n -= vl, src += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t vsrc = __riscv_vle32_v_f32m8(src, vl);
        vmaxf = __riscv_vfmax_vv_f32m8(vmaxf, vsrc, vl);
    }

    vfloat32m1_t vmaxf_init = __riscv_vfmv_v_f_f32m1(Maximum, __riscv_vsetvlmax_e32m1());
    vl = __riscv_vsetvlmax_e32m8();
    vmaxf_init = __riscv_vfredmax_vs_f32m8_f32m1(vmaxf, vmaxf_init, vl);

    return __riscv_vfmv_f(vmaxf_init);
}

float
MlasComputeSumExpF32Kernel_RVV(const float* Input, float* Output, const size_t D, const float* NegativeMaximum)
{
    float* src = const_cast<float*>(reinterpret_cast<const float*>(Input));
    float* dst = reinterpret_cast<float*>(Output);
    float Accumulator = 0.0f;
    const float Neg_Max = NegativeMaximum[0];

    float LowerRange = -103.9720840454f;
    float UpperRange = 88.7762626647950f;
    float LowerRangeSumExp = -88.3762626647949f;
    float UpperRangeSumExp = 88.3762626647949f;
    float RoundingBias = MLAS_ROUNDING_BIAS_MAGIC;
    float Log2Reciprocal = 1.44269504088896341f;
    float Log2High = -6.93145752e-1f;
    float Log2Low = -1.42860677e-6f;
    float poly_0 = 0x1.694000p-10;
    float poly_1 = 0x1.125edcp-7;
    float poly_2 = 0x1.555b5ap-5;
    float poly_3 = 0x1.555450p-3;
    float poly_4 = 0x1.fffff6p-2;
    float poly_56 = 0x1.000000p+0;
    // int32_t MinimumExponent = int32_t(0xC1000000);    //unused
    int32_t MaximumExponent = int32_t(0x3F800000);

    if (Output != nullptr) {
        __asm__ volatile(

            "mv                   t3, %[LEN]                                  \n\t"
            "mv                   s1, %[SRC]                                  \n\t"
            "mv                   s2, %[DST]                                  \n\t"

            /* 2.0 Compute exp() and accumulate and store to cache_buffer */
            "vsetvli              t0, zero, e32, m4                           \n\t"
            "vxor.vv              v16, v8, v8                                 \n\t"
            "vxor.vv              v0, v8, v8                                  \n\t"

            ".align 4                                                         \n\t"
            "_EXPACC_LEN_LPST:                                                \n\t"
            "vsetvli              t0, t3, e32, m4                             \n\t"

            "vle32.v              v0, (s1)                                    \n\t"
            "add                  s1, s1, 128                                 \n\t"

            /* 2.1 START exp()  */
            "vfadd.vf             v0, v0, %[NEG_MAX]                          \n\t"  // v4 = x - max

            // Ensure that q = RN(x/log(2)) >= e_min, so that 2^q can be computed
            // safely with a simple shift into the exponent field. xmin =
            // round(-126.5 * log(2), single, RU) ~ -87.68311309814453125 const
            // float xmin = -0x1.5ebb82p6;
            "vfmax.vf             v0, v0, %[LowerRangeSumExp]                 \n\t"

            // 2.1.0. Reduction x = s * q ln(2)
            // const float r_ln2f = 0x1.715476p0f;  // single(1/log(2));
            // const float l2uf = 0x1.62e4p-1f;     // round(log(2), 24-8, RN);
            // const float l2lf = 0x1.7f7d1cp-20f;  // round(log(2) - l2uf, single,
            // RN);
            "vfmv.v.f             v4, %[RoundingBias]                         \n\t"
            "vfmacc.vf            v4, %[Log2Reciprocal], v0                   \n\t"  // biased in mlas
            "vfsub.vf             v8, v4, %[RoundingBias]                     \n\t"  // v12_a = float(x - n);

            // Use Cody-Waite range reduction method (note two constants to
            // represent log(2)) to improve accuracy.
            "vfmacc.vf            v0, %[Log2High], v8                         \n\t"
            "vfmacc.vf            v0, %[Log2Low], v8                          \n\t"
            "vfcvt.x.f.V          v8, v4                                      \n\t"

            // 2.1.1. Approximate e^s by degree-6 polynomial approximation
            "vfmv.v.f             v4, %[poly_0]                               \n\t"
            "vfmv.v.f             v12, %[poly_1]                              \n\t"
            "vfmadd.vv            v4, v0, v12                                 \n\t"
            "vfmv.v.f             v12, %[poly_2]                              \n\t"
            "vfmadd.vv            v4, v0, v12                                 \n\t"
            "vfmv.v.f             v12, %[poly_3]                              \n\t"
            "vfmadd.vv            v4, v0, v12                                 \n\t"
            "vfmv.v.f             v12, %[poly_4]                              \n\t"
            "vfmadd.vv            v4, v0, v12                                 \n\t"
            "vfmv.v.f             v12, %[poly_56]                             \n\t"
            "vfmadd.vv            v4, v0, v12                                 \n\t"
            "vfmv.v.f             v12, %[poly_56]                             \n\t"
            "vfmadd.vv            v4, v0, v12                                 \n\t"  // v8 = poly(input - max)

            // 2.1.2. Reconstruction: compute u = u*2^q
            // const int16_t p = (24 - 1);
            // const int16_t bias = (128 - 1);
            "vsll.vi              v8, v8, 23                                  \n\t"
            "vadd.vx              v8, v8, %[MaximumExponent]                  \n\t"
            //"vfcvt.f.x.v          v12, v8                                   \n\t"

            "vfmul.vv             v0, v4, v8                                  \n\t"
            /* 2.1 END exp()  */

            "vse32.v              v0, (s2)                                    \n\t"  // exp(输入-max)输出
            "addi                 s2, s2, 128                                 \n\t"
            "vfadd.vv             v16, v16, v0                                \n\t"
            "sub                  t3, t3, t0                                  \n\t"
            "bgtz                 t3, _EXPACC_LEN_LPST                        \n\t"

            "_EXPACC_LEN_LPND:                                                \n\t"

            "vsetvli              t0, zero, e32, m4                           \n\t"
            "vxor.vv              v24, v8, v8                                 \n\t"
            "vfredusum.vs         v24, v16, v24                               \n\t"
            "vfmv.f.s             %[RTN], v24                                 \n\t"  // ft2 = sum(exp( ))

            : [ RTN ] "=f"(Accumulator), [ SRC ] "+r"(src), [ DST ] "+r"(dst)
            : [ LEN ] "r"(D), [ NEG_MAX ] "f"(Neg_Max), [ LowerRange ] "f"(LowerRange), [ UpperRange ] "f"(UpperRange),
              [ LowerRangeSumExp ] "f"(LowerRangeSumExp), [ UpperRangeSumExp ] "f"(UpperRangeSumExp),
              [ RoundingBias ] "f"(RoundingBias), [ Log2Reciprocal ] "f"(Log2Reciprocal), [ Log2High ] "f"(Log2High),
              [ Log2Low ] "f"(Log2Low), [ poly_0 ] "f"(poly_0), [ poly_1 ] "f"(poly_1), [ poly_2 ] "f"(poly_2),
              [ poly_3 ] "f"(poly_3), [ poly_4 ] "f"(poly_4), [ poly_56 ] "f"(poly_56),
              [ MaximumExponent ] "r"(MaximumExponent)
            : "cc", "s1", "s2", "t0", "t3");
    } else {
        __asm__ volatile(

            "mv                   t3, %[LEN]                                  \n\t"
            "mv                   s1, %[SRC]                                  \n\t"
            "mv                   s2, %[DST]                                  \n\t"

            /* 2.0 Compute exp() and accumulate and store to cache_buffer */
            "vsetvli              t0, zero, e32, m4                           \n\t"
            "vxor.vv              v16, v8, v8                                 \n\t"
            "vxor.vv              v0, v8, v8                                  \n\t"

            ".align 4                                                         \n\t"
            "_ACCNST_LEN_LPST:                                                \n\t"
            "vsetvli              t0, t3, e32, m4                             \n\t"

            "vle32.v              v0, (s1)                                    \n\t"
            "add                  s1, s1, 128                                 \n\t"

            /* 2.1 START exp()  */
            "vfadd.vf             v0, v0, %[NEG_MAX]                          \n\t"  // v4 = x - max

            // Ensure that q = RN(x/log(2)) >= e_min, so that 2^q can be computed
            // safely with a simple shift into the exponent field. xmin =
            // round(-126.5 * log(2), single, RU) ~ -87.68311309814453125 const
            // float xmin = -0x1.5ebb82p6;
            "vfmax.vf             v0, v0, %[LowerRangeSumExp]                 \n\t"

            // 2.1.0. Reduction x = s * q ln(2)
            // const float r_ln2f = 0x1.715476p0f;  // single(1/log(2));
            // const float l2uf = 0x1.62e4p-1f;     // round(log(2), 24-8, RN);
            // const float l2lf = 0x1.7f7d1cp-20f;  // round(log(2) - l2uf, single,
            // RN);
            "vfmv.v.f             v4, %[RoundingBias]                         \n\t"
            "vfmacc.vf            v4, %[Log2Reciprocal], v0                   \n\t"  // biased in mlas
            "vfsub.vf             v8, v4, %[RoundingBias]                     \n\t"  // v12_a = float(x - n);

            // Use Cody-Waite range reduction method (note two constants to
            // represent log(2)) to improve accuracy.
            "vfmacc.vf            v0, %[Log2High], v8                         \n\t"
            "vfmacc.vf            v0, %[Log2Low], v8                          \n\t"
            "vfcvt.x.f.V          v8, v4                                      \n\t"

            // 2.1.1. Approximate e^s by degree-6 polynomial approximation
            "vfmv.v.f             v4, %[poly_0]                               \n\t"
            "vfmv.v.f             v12, %[poly_1]                              \n\t"
            "vfmadd.vv            v4, v0, v12                                 \n\t"
            "vfmv.v.f             v12, %[poly_2]                              \n\t"
            "vfmadd.vv            v4, v0, v12                                 \n\t"
            "vfmv.v.f             v12, %[poly_3]                              \n\t"
            "vfmadd.vv            v4, v0, v12                                 \n\t"
            "vfmv.v.f             v12, %[poly_4]                              \n\t"
            "vfmadd.vv            v4, v0, v12                                 \n\t"
            "vfmv.v.f             v12, %[poly_56]                             \n\t"
            "vfmadd.vv            v4, v0, v12                                 \n\t"
            "vfmv.v.f             v12, %[poly_56]                             \n\t"
            "vfmadd.vv            v4, v0, v12                                 \n\t"  // v8 = poly(input - max)

            // 2.1.2. Reconstruction: compute u = u*2^q
            // const int16_t p = (24 - 1);
            // const int16_t bias = (128 - 1);
            "vsll.vi              v8, v8, 23                                  \n\t"
            "vadd.vx              v8, v8, %[MaximumExponent]                  \n\t"
            //"vfcvt.f.x.v          v12, v8                                   \n\t"

            "vfmul.vv             v0, v4, v8                                  \n\t"
            /* 2.1 END exp()  */

            "vfadd.vv             v16, v16, v0                                \n\t"
            "sub                  t3, t3, t0                                  \n\t"
            "bgtz                 t3, _ACCNST_LEN_LPST                        \n\t"

            "_ACCNST_LEN_LPND:                                                \n\t"

            "vsetvli              t0, zero, e32, m4                           \n\t"
            "vxor.vv              v24, v8, v8                                 \n\t"
            "vfredusum.vs         v24, v16, v24                               \n\t"
            "vfmv.f.s             %[RTN], v24                                 \n\t"  // ft2 = sum(exp( ))

            : [ RTN ] "=f"(Accumulator), [ SRC ] "+r"(src), [ DST ] "+r"(dst)
            : [ LEN ] "r"(D), [ NEG_MAX ] "f"(Neg_Max), [ LowerRange ] "f"(LowerRange), [ UpperRange ] "f"(UpperRange),
              [ LowerRangeSumExp ] "f"(LowerRangeSumExp), [ UpperRangeSumExp ] "f"(UpperRangeSumExp),
              [ RoundingBias ] "f"(RoundingBias), [ Log2Reciprocal ] "f"(Log2Reciprocal), [ Log2High ] "f"(Log2High),
              [ Log2Low ] "f"(Log2Low), [ poly_0 ] "f"(poly_0), [ poly_1 ] "f"(poly_1), [ poly_2 ] "f"(poly_2),
              [ poly_3 ] "f"(poly_3), [ poly_4 ] "f"(poly_4), [ poly_56 ] "f"(poly_56),
              [ MaximumExponent ] "r"(MaximumExponent)
            : "cc", "s1", "s2", "t0", "t3");
    }
    return Accumulator;
}
void
MlasComputeLogSoftmaxOutputF32Kernel_RVV(const float* Input, float* Output, size_t N, const float* Parameters)
{
    float* src = const_cast<float*>(reinterpret_cast<const float*>(Input));
    float* dst = reinterpret_cast<float*>(Output);
    size_t length = N;
    const float NegativeMaximum = Parameters[0];
    const float Logarithm = Parameters[1];

    __asm__ volatile(
        "mv                   t3, %[LEN]                                \n\t"
        "mv                   s1, %[SRC]                                \n\t"  // input
        "mv                   s2, %[DST]                                \n\t"  // output
        "fmv.s                ft1, %[NEG_MAX]                           \n\t"  // input
        "fmv.s                ft2, %[LOG]                               \n\t"  // output

        ".align 4                                                       \n\t"
        "_LOG_LEN_LPST:                                                 \n\t"
        "vsetvli              t0, t3, e32, m8                           \n\t"
        "vle32.v              v0, (s1)                                  \n\t"
        "addi                 s1, s1, 256                               \n\t"
        "vfadd.vf             v16, v0, %[NEG_MAX]                       \n\t"
        "vfsub.vf             v24, v16, %[LOG]                          \n\t"
        "vse32.v              v24, (s2)                                 \n\t"
        "addi                 s2, s2, 256                               \n\t"
        "sub                  t3, t3, t0                                \n\t"
        "bgtz                 t3, _LOG_LEN_LPST                         \n\t"

        : [ SRC ] "+r"(src), [ DST ] "+r"(dst)
        : [ LEN ] "r"(length), [ NEG_MAX ] "f"(NegativeMaximum), [ LOG ] "f"(Logarithm)
        : "cc", "s1", "s2", "t0", "t1", "t2", "t3", "ft1", "ft2");
}

void
MlasComputeSoftmaxOutputF32Kernel_RVV(float* Output, size_t N, const float* Parameters)
{
    float* pOutput = reinterpret_cast<float*>(Output);
    size_t length = N;
    const float Scale = Parameters[0];

    __asm__ volatile(
        "mv                   t3, %[LEN]                                \n\t"
        "mv                   s1, %[SRC]                                \n\t"  // input
        "mv                   s2, %[SRC]                                \n\t"  // output

        ".align 4                                                       \n\t"
        "_MUL_LEN_LPST:                                                 \n\t"
        "vsetvli              t0, t3, e32, m8                           \n\t"
        "vle32.v              v0, (s1)                                  \n\t"
        "addi                 s1, s1, 256                               \n\t"
        "vfmul.vf             v16, v0, %[SCL]                           \n\t"
        "vse32.v              v16, (s2)                                 \n\t"
        "addi                 s2, s2, 256                               \n\t"

        "sub                  t3, t3, t0                                \n\t"
        "bgtz                 t3, _MUL_LEN_LPST                         \n\t"

        : [ SRC ] "+r"(pOutput)
        : [ LEN ] "r"(length), [ SCL ] "f"(Scale)
        : "cc", "s1", "s2", "t0", "t1", "t2", "t3", "ft1");
}
