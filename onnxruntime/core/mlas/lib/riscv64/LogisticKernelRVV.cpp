// Copyright (c) 2023 SpacemiT. All rights reserved.
// Licensed under the MIT License.

#include "mlasi.h"

MLAS_INTERNAL_DATA const struct {
    float LowerRange;
    float UpperRange;
    float alpha_9;
    float alpha_7;
    float alpha_5;
    float alpha_3;
    float alpha_1;
    float beta_10;
    float beta_8;
    float beta_6;
    float beta_4;
    float beta_2;
    float beta_0;
    float one_half;
} MlasLogisticConstantsRVV = {
    -18.0f,
    18.0f,
    4.37031012579801e-11f,
    1.15627324459942e-07f,
    6.08574864600143e-05f,
    8.51377133304701e-03f,
    2.48287947061529e-01f,
    6.10247389755681e-13f,
    5.76102136993427e-09f,
    6.29106785017040e-06f,
    1.70198817374094e-03f,
    1.16817656904453e-01f,
    9.93151921023180e-01f,
    0.5f,
};

void MLASCALL
MlasLogisticKernel_RVV(const float* Input, float* Output, size_t N)
/*++

Routine Description:

    This routine implements the generic kernel for the logistic function.

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    N - Supplies the number of elements to process.

Return Value:

    None.

--*/
{
    __asm__ volatile(
        "LOOP%=:                                \t\n"
        "vsetvli  t0,       %[n],     e32,      m1\t\n"
        "sub      %[n],     %[n],     t0        \t\n"
        "slli     t0,       t0,       2         \t\n"

        "vfmv.v.f v20,      %[b0]               \t\n"
        "vfmv.v.f v21,      %[a1]               \t\n"
        "vfmv.v.f v22,      %[b2]               \t\n"
        "vfmv.v.f v23,      %[a3]               \t\n"
        "vfmv.v.f v24,      %[b4]               \t\n"
        "vfmv.v.f v25,      %[a5]               \t\n"
        "vfmv.v.f v26,      %[b6]               \t\n"
        "vfmv.v.f v27,      %[a7]               \t\n"
        "vfmv.v.f v28,      %[b8]               \t\n"

        "vle32.v  v0,       (%[x])              \t\n"
        "add      %[x],     %[x],     t0        \t\n"

        // TODO: qemu riscv64 NaN Propagation, If one operand is a quiet NaN and the other is not a NaN, the result is
        // the non-NaN operand.
        "vfmax.vf v0,       v0,       %[lr]     \t\n"
        "vfmin.vf v0,       v0,       %[ur]     \t\n"  // v

        "vfmul.vv v4,       v0,       v0        \t\n"  // sq

        "vmv.v.v   v8,       v4                  \t\n"
        "vfmadd.vf v8,      %[a9],    v27       \t\n"  // p

        "vfmadd.vv v8,      v4,       v25       \t\n"  // p

        "vfmadd.vv v8,      v4,       v23       \t\n"  // p

        "vfmadd.vv v8,      v4,       v21       \t\n"  // p

        "vfmul.vv v8,       v8,       v0        \t\n"

        "vmv.v.v   v12,      v4                  \t\n"
        "vfmadd.vf v12,     %[b10],   v28       \t\n"  // q

        "vfmadd.vv v12,     v4,       v26       \t\n"  // q

        "vfmadd.vv v12,     v4,       v24       \t\n"  // q

        "vfmadd.vv v12,     v4,       v22       \t\n"  // q

        "vfmadd.vv v12,     v4,       v20       \t\n"  // q

        "vfdiv.vv v12,      v8,       v12       \t\n"

        "vfadd.vf v12,      v12,      %[onehalf]\t\n"

        "vse32.v  v12,      (%[y])              \t\n"
        "add      %[y],     %[y],     t0        \t\n"
        "bnez     %[n],     LOOP%=              \t\n"
        : [ n ] "+r"(N), [ x ] "+r"(Input), [ y ] "+r"(Output)
        : [ lr ] "f"(MlasLogisticConstantsRVV.LowerRange), [ ur ] "f"(MlasLogisticConstantsRVV.UpperRange),
          [ a1 ] "f"(MlasLogisticConstantsRVV.alpha_1), [ a3 ] "f"(MlasLogisticConstantsRVV.alpha_3),
          [ a5 ] "f"(MlasLogisticConstantsRVV.alpha_5), [ a7 ] "f"(MlasLogisticConstantsRVV.alpha_7),
          [ a9 ] "f"(MlasLogisticConstantsRVV.alpha_9), [ b0 ] "f"(MlasLogisticConstantsRVV.beta_0),
          [ b2 ] "f"(MlasLogisticConstantsRVV.beta_2), [ b4 ] "f"(MlasLogisticConstantsRVV.beta_4),
          [ b6 ] "f"(MlasLogisticConstantsRVV.beta_6), [ b8 ] "f"(MlasLogisticConstantsRVV.beta_8),
          [ b10 ] "f"(MlasLogisticConstantsRVV.beta_10), [ onehalf ] "f"(MlasLogisticConstantsRVV.one_half)
        : "cc", "t0");
}
