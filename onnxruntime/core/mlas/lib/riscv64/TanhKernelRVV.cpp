// Copyright (c) 2023 SpacemiT. All rights reserved.
// Licensed under the MIT License.

#include "mlasi.h"

struct {
    float LowerRange;
    float UpperRange;
    float alpha_13;
    float alpha_11;
    float alpha_9;
    float alpha_7;
    float alpha_5;
    float alpha_3;
    float alpha_1;
    float beta_6;
    float beta_4;
    float beta_2;
    float beta_0;
} MlasTanhConstants_RVV = {
    -9.0f,
    9.0f,
    -2.76076847742355e-16f,
    2.00018790482477e-13f,
    -8.60467152213735e-11f,
    5.12229709037114e-08f,
    1.48572235717979e-05f,
    6.37261928875436e-04f,
    4.89352455891786e-03f,
    1.19825839466702e-06f,
    1.18534705686654e-04f,
    2.26843463243900e-03f,
    4.89352518554385e-03f,
};

void
MlasTanhKernel_RVV(const float* Input, float* Output, size_t N)
{
    float* src = const_cast<float*>(reinterpret_cast<const float*>(Input));
    float* dst = Output;

    __asm__ volatile(

        "mv                   t3, %[LEN]                                  \n\t"
        "mv                   s1, %[SRC]                                  \n\t"
        "mv                   s2, %[DST]                                  \n\t"

        ".align 4                                                         \n\t"
        "_TANH_LEN_LPST:                                                  \n\t"

        "vsetvli              t0, t3, e32, m8                             \n\t"

        "vle32.v              v0, (s1)                                    \n\t"
        "sh2add               s1, t0, s1                                  \n\t"

        "vfmax.vf             v0, v0, %[LowerRange]                       \n\t"
        "vfmin.vf             v0, v0, %[UpperRange]                       \n\t"

        "vfmul.vv             v8, v0, v0                                  \n\t"

        "vfmv.v.f             v16, %[alpha_13]                            \n\t"
        "vfmv.v.f             v24, %[alpha_11]                            \n\t"
        "vfmadd.vv            v16, v8, v24                                \n\t"
        "vfmv.v.f             v24, %[alpha_9]                             \n\t"
        "vfmadd.vv            v16, v8, v24                                \n\t"
        "vfmv.v.f             v24, %[alpha_7]                             \n\t"
        "vfmadd.vv            v16, v8, v24                                \n\t"
        "vfmv.v.f             v24, %[alpha_5]                             \n\t"
        "vfmadd.vv            v16, v8, v24                                \n\t"
        "vfmv.v.f             v24, %[alpha_3]                             \n\t"
        "vfmadd.vv            v16, v8, v24                                \n\t"
        "vfmv.v.f             v24, %[alpha_1]                             \n\t"
        "vfmadd.vv            v16, v8, v24                                \n\t"
        "vfmul.vv             v16, v16, v0                                \n\t"  // p = v16

        "vfmv.v.f             v0, %[beta_6]                               \n\t"
        "vfmv.v.f             v24, %[beta_4]                              \n\t"
        "vfmadd.vv            v0, v8, v24                                 \n\t"
        "vfmv.v.f             v24, %[beta_2]                              \n\t"
        "vfmadd.vv            v0, v8, v24                                 \n\t"
        "vfmv.v.f             v24, %[beta_0]                              \n\t"
        "vfmadd.vv            v0, v8, v24                                 \n\t"

        // Softmax in ep has a div routine which might be better
        "vfdiv.vv            v16, v16, v0                                 \n\t"  // v16=v16/v0

        "vse32.v              v16, (s2)                                   \n\t"
        "sh2add               s2, t0, s2                                  \n\t"

        "sub                  t3, t3, t0                                  \n\t"
        "bgtz                 t3, _TANH_LEN_LPST                          \n\t"

        "_TANH_LEN_LPND:                                                  \n\t"

        : [ SRC ] "+r"(src), [ DST ] "+r"(dst)
        : [ LEN ] "r"(N), [ LowerRange ] "f"(MlasTanhConstants_RVV.LowerRange),
          [ UpperRange ] "f"(MlasTanhConstants_RVV.UpperRange), [ alpha_13 ] "f"(MlasTanhConstants_RVV.alpha_13),
          [ alpha_11 ] "f"(MlasTanhConstants_RVV.alpha_11), [ alpha_9 ] "f"(MlasTanhConstants_RVV.alpha_9),
          [ alpha_7 ] "f"(MlasTanhConstants_RVV.alpha_7), [ alpha_5 ] "f"(MlasTanhConstants_RVV.alpha_5),
          [ alpha_3 ] "f"(MlasTanhConstants_RVV.alpha_3), [ alpha_1 ] "f"(MlasTanhConstants_RVV.alpha_1),
          [ beta_6 ] "f"(MlasTanhConstants_RVV.beta_6), [ beta_4 ] "f"(MlasTanhConstants_RVV.beta_4),
          [ beta_2 ] "f"(MlasTanhConstants_RVV.beta_2), [ beta_0 ] "f"(MlasTanhConstants_RVV.beta_0)
        : "cc", "s1", "s2", "t0", "t3");
}
