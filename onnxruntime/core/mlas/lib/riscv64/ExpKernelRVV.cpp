// Copyright (c) 2023 SpacemiT. All rights reserved.
// Licensed under the MIT License.

#include "mlasi.h"

void
MlasComputeExpF32Kernel_RVV(const float* Input, float* Output, const size_t N)
{
    float LowerRange = -103.9720840454f;
    float UpperRange = 88.7762626647950f;
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
    int32_t MinimumExponent = int32_t(0xC1000000);
    int32_t MaximumExponent = int32_t(0x3F800000);

    __asm__ volatile(
        "addi                 t2, %[N], 0                   \n\t"
        "LOOP_N%=:                                          \n\t"
        "vsetvli              t0, t2, e32, m8               \n\t"
        "vle32.v              v0, (%[SRC])                  \n\t"
        "sh2add               %[SRC], t0, %[SRC]            \n\t"
        "vfmv.v.f             v8, %[RoundingBias]           \n\t"
        "vfmv.v.f             v24, %[poly_0]                \n\t"
        "sub                  t2, t2, t0                    \n\t"
        "vfmax.vf             v0, v0, %[LowerRange]         \n\t"
        "vfmin.vf             v0, v0, %[UpperRange]         \n\t"
        "vfmacc.vf            v8, %[Log2Reciprocal], v0     \n\t"
        "vfsub.vf             v16, v8, %[RoundingBias]      \n\t"
        "vfmacc.vf            v0, %[Log2High], v16          \n\t"
        "vfmacc.vf            v0, %[Log2Low], v16           \n\t"

        "vfmv.v.f             v16, %[poly_1]                \n\t"
        "vfmadd.vv            v24, v0, v16                  \n\t"
        "vfmv.v.f             v16, %[poly_2]                \n\t"
        "vfmadd.vv            v24, v0, v16                  \n\t"
        "vfmv.v.f             v16, %[poly_3]                \n\t"
        "vfmadd.vv            v24, v0, v16                  \n\t"
        "vfmv.v.f             v16, %[poly_4]                \n\t"
        "vfmadd.vv            v24, v0, v16                  \n\t"
        "vfmv.v.f             v16, %[poly_56]               \n\t"
        "vfmadd.vv            v24, v0, v16                  \n\t"

        "vsll.vi              v8, v8, 23                    \n\t"
        "vmin.vx              v16, v8, %[MaximumExponent]   \n\t"
        "vmax.vx              v16, v16, %[MinimumExponent]  \n\t"
        "vsub.vv              v8, v8, v16                   \n\t"
        "vadd.vx              v8, v8, %[MaximumExponent]    \n\t"
        "vadd.vx              v16, v16, %[MaximumExponent]  \n\t"
        "vfmul.vv             v0, v0, v8                    \n\t"
        "vfmadd.vv            v24, v0, v8                   \n\t"
        "vfmul.vv             v24, v24, v16                 \n\t"
        "vse32.v              v24, (%[DST])                 \n\t"
        "sh2add               %[DST], t0, %[DST]            \n\t"
        "bnez                 t2, LOOP_N%=                  \n\t"

        : [ SRC ] "+r"(Input), [ DST ] "+r"(Output)
        : [ N ] "r"(N), [ LowerRange ] "f"(LowerRange), [ UpperRange ] "f"(UpperRange),
          [ RoundingBias ] "f"(RoundingBias), [ Log2Reciprocal ] "f"(Log2Reciprocal), [ Log2High ] "f"(Log2High),
          [ Log2Low ] "f"(Log2Low), [ MaximumExponent ] "r"(MaximumExponent), [ MinimumExponent ] "r"(MinimumExponent),
          [ poly_0 ] "f"(poly_0), [ poly_1 ] "f"(poly_1), [ poly_2 ] "f"(poly_2), [ poly_3 ] "f"(poly_3),
          [ poly_4 ] "f"(poly_4), [ poly_56 ] "f"(poly_56)
        : "cc", "t2", "t0");
}
