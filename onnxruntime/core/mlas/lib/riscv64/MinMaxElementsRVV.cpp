// Copyright (c) 2023 SpacemiT. All rights reserved.
// Licensed under the MIT License.

#include "../mlasi.h"

void MLASCALL
MlasReduceMinimumMaximumF32Kernel_RVV(const float* Input, float* Min, float* Max, size_t D)
{
    float* src = const_cast<float*>(Input);
    float Maximum = std::numeric_limits<float>::lowest();
    float Minimum = std::numeric_limits<float>::max();
    int64_t n = D;
    size_t gvl = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t vmaxf = __riscv_vfmv_v_f_f32m4(Maximum, gvl);
    vfloat32m4_t vminf = __riscv_vfmv_v_f_f32m4(Minimum, gvl);
    for (; n > 0; n -= gvl, src += gvl) {
        gvl = __riscv_vsetvl_e32m4(n);
        vfloat32m4_t vsrc = __riscv_vle32_v_f32m4(src, gvl);
        vmaxf = __riscv_vfmax_vv_f32m4(vmaxf, vsrc, gvl);
        vminf = __riscv_vfmin_vv_f32m4(vminf, vsrc, gvl);
    }
    gvl = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t vmaxf_init =
        __riscv_vfmax_vv_f32m1(__riscv_vget_v_f32m4_f32m1(vmaxf, 0), __riscv_vget_v_f32m4_f32m1(vmaxf, 1), gvl);
    vmaxf_init = __riscv_vfmax_vv_f32m1(vmaxf_init, __riscv_vget_v_f32m4_f32m1(vmaxf, 2), gvl);
    vmaxf_init = __riscv_vfmax_vv_f32m1(vmaxf_init, __riscv_vget_v_f32m4_f32m1(vmaxf, 3), gvl);

    vfloat32m1_t vminf_init =
        __riscv_vfmin_vv_f32m1(__riscv_vget_v_f32m4_f32m1(vminf, 0), __riscv_vget_v_f32m4_f32m1(vminf, 1), gvl);
    vminf_init = __riscv_vfmin_vv_f32m1(vminf_init, __riscv_vget_v_f32m4_f32m1(vminf, 2), gvl);
    vminf_init = __riscv_vfmin_vv_f32m1(vminf_init, __riscv_vget_v_f32m4_f32m1(vminf, 3), gvl);

    vfloat32m1_t vmaxf_val = __riscv_vfmv_v_f_f32m1(Maximum, gvl);
    vfloat32m1_t vminf_val = __riscv_vfmv_v_f_f32m1(Minimum, gvl);
    vmaxf_init = __riscv_vfredmax_vs_f32m1_f32m1(vmaxf_init, vmaxf_val, gvl);
    vminf_init = __riscv_vfredmin_vs_f32m1_f32m1(vminf_init, vminf_val, gvl);

    *Min = __riscv_vfmv_f_s_f32m1_f32(vminf_init);
    *Max = __riscv_vfmv_f_s_f32m1_f32(vmaxf_init);
}
