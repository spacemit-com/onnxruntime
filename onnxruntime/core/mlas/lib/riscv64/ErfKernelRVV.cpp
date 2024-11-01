// Copyright (c) 2023 SpacemiT. All rights reserved.
// Licensed under the MIT License.

#include "../mlasi.h"

MLAS_INTERNAL_DATA const struct {
    float ErfUpperAbsRange;
    float ErfSplitBoundary;
    float ErfSMALL_P0;
    float ErfSMALL_P1;
    float ErfSMALL_P2;
    float ErfSMALL_P3;
    float ErfSMALL_P4;
    float ErfSMALL_P5_Minus_One;
    float ErfReserved0;
    float ErfBIG_P0;
    float ErfBIG_P1;
    float ErfBIG_P2;
    float ErfBIG_P3;
    float ErfBIG_P4;
    float ErfBIG_P5;
    float ErfBIG_P6_Minus_One;
    float ErfNegZero;
    float ErfOne;
    float Exp_UpperRange;
    float Exp_LowerRange;
    float Exp_Log2Reciprocal;
    float Exp_log2_hi;
    float Exp_log2_lo;
    float Exp_P0;
    float Exp_P1;
    float Exp_P2;
    float Exp_P3;
    float Exp_P4;
    float Exp_P5;
    float Exp_P6;
    float Exp_C;
    int32_t Exp_X7F;
} MlasErfConstants_RVV = {
    3.925f,
    0.921875f,
    -5.99104969e-4f,
    4.99339588e-3f,
    -2.67667342e-2f,
    1.12818025e-1f,
    -3.76124859e-1f,
    1.28379151e-1f,
    0.0f,
    1.72948930e-5f,
    -3.83208680e-4f,
    3.88393435e-3f,
    -2.42545605e-2f,
    1.06777847e-1f,
    6.34846687e-1f,
    1.28717512e-1f,
    -0.0f,
    1.0f,
    // Independent parameters to calculate Exp for Erff()
    88.3762626647950f,
    -88.3762626647949f,
    1.44269504088896341f,
    -6.93145752e-1f,
    -1.42860677e-6f,
    1.38319808e-3f,
    8.37550033e-3f,
    4.16689515e-2f,
    1.66664466e-1f,
    4.99999851e-1f,
    1.00000000e+0f,
    1.00000000e+0f,
    1.25829120e+7f,
    127,
};
void
MlasErfKernel_RVV(const float* Input, float* Output, size_t N)
{
    float* input_ptr = const_cast<float*>(reinterpret_cast<const float*>(Input));
    float* output_ptr = reinterpret_cast<float*>(Output);
    int64_t batch_size = (int64_t)N;
    for (size_t vl; batch_size > 0; batch_size -= vl, input_ptr += vl, output_ptr += vl) {
        vl = __riscv_vsetvl_e32m4(batch_size);
        vfloat32m4_t vec_f32_src = __riscv_vle32_v_f32m4(input_ptr, vl);
        vfloat32m4_t fNegZero = __riscv_vfmv_v_f_f32m4(MlasErfConstants_RVV.ErfNegZero, vl);

        // float to int
        vint32m4_t dNegZero = __riscv_vreinterpret_v_f32m4_i32m4(fNegZero);
        vint32m4_t vec_i32_src = __riscv_vreinterpret_v_f32m4_i32m4(vec_f32_src);
        vint32m4_t SignMask = __riscv_vand_vv_i32m4(vec_i32_src, dNegZero, vl);
        vfloat32m4_t AbsValue = __riscv_vfabs_v_f32m4(vec_f32_src, vl);
        vfloat32m4_t SquareValue = __riscv_vfmul_vv_f32m4(AbsValue, AbsValue, vl);
        vfloat32m4_t r_small = __riscv_vfmv_v_f_f32m4(MlasErfConstants_RVV.ErfSMALL_P0, vl);
        vfloat32m4_t para = __riscv_vfmv_v_f_f32m4(MlasErfConstants_RVV.ErfSMALL_P1, vl);
        r_small = __riscv_vfmadd_vv_f32m4(r_small, SquareValue, para, vl);
        para = __riscv_vfmv_v_f_f32m4(MlasErfConstants_RVV.ErfSMALL_P2, vl);
        r_small = __riscv_vfmadd_vv_f32m4(r_small, SquareValue, para, vl);
        para = __riscv_vfmv_v_f_f32m4(MlasErfConstants_RVV.ErfSMALL_P3, vl);
        r_small = __riscv_vfmadd_vv_f32m4(r_small, SquareValue, para, vl);
        para = __riscv_vfmv_v_f_f32m4(MlasErfConstants_RVV.ErfSMALL_P4, vl);
        r_small = __riscv_vfmadd_vv_f32m4(r_small, SquareValue, para, vl);
        para = __riscv_vfmv_v_f_f32m4(MlasErfConstants_RVV.ErfSMALL_P5_Minus_One, vl);
        r_small = __riscv_vfmadd_vv_f32m4(r_small, SquareValue, para, vl);
        r_small = __riscv_vfmadd_vv_f32m4(r_small, AbsValue, AbsValue, vl);
        vbool8_t split_mask = __riscv_vmfgt_vf_f32m4_b8(AbsValue, MlasErfConstants_RVV.ErfSplitBoundary, vl);
        AbsValue = __riscv_vfmin_vf_f32m4(AbsValue, MlasErfConstants_RVV.ErfUpperAbsRange, vl);
        vfloat32m4_t r_big = __riscv_vfmv_v_f_f32m4(MlasErfConstants_RVV.ErfBIG_P0, vl);
        para = __riscv_vfmv_v_f_f32m4(MlasErfConstants_RVV.ErfBIG_P1, vl);
        r_big = __riscv_vfmadd_vv_f32m4(r_big, AbsValue, para, vl);
        para = __riscv_vfmv_v_f_f32m4(MlasErfConstants_RVV.ErfBIG_P2, vl);
        r_big = __riscv_vfmadd_vv_f32m4(r_big, AbsValue, para, vl);
        para = __riscv_vfmv_v_f_f32m4(MlasErfConstants_RVV.ErfBIG_P3, vl);
        r_big = __riscv_vfmadd_vv_f32m4(r_big, AbsValue, para, vl);
        para = __riscv_vfmv_v_f_f32m4(MlasErfConstants_RVV.ErfBIG_P4, vl);
        r_big = __riscv_vfmadd_vv_f32m4(r_big, AbsValue, para, vl);
        para = __riscv_vfmv_v_f_f32m4(MlasErfConstants_RVV.ErfBIG_P5, vl);
        r_big = __riscv_vfmadd_vv_f32m4(r_big, AbsValue, para, vl);
        para = __riscv_vfmv_v_f_f32m4(MlasErfConstants_RVV.ErfBIG_P6_Minus_One, vl);
        r_big = __riscv_vfmadd_vv_f32m4(r_big, AbsValue, para, vl);
        r_big = __riscv_vfmadd_vv_f32m4(r_big, AbsValue, AbsValue, vl);

        // 1.0 - exp(-r_big), no need to do min()
        float_t Const_M1 = -1.0f;
        vfloat32m4_t NEG_r_big = __riscv_vfmul_vf_f32m4(r_big, Const_M1, vl);
        r_big = __riscv_vfmax_vf_f32m4(NEG_r_big, MlasErfConstants_RVV.Exp_LowerRange, vl);
        vfloat32m4_t para_ExpC = __riscv_vfmv_v_f_f32m4(MlasErfConstants_RVV.Exp_C, vl);
        vfloat32m4_t r = __riscv_vfmv_v_f_f32m4(MlasErfConstants_RVV.Exp_Log2Reciprocal, vl);
        r = __riscv_vfmadd_vv_f32m4(r, r_big, para_ExpC, vl);
        r = __riscv_vfsub_vv_f32m4(r, para_ExpC, vl);
        para = __riscv_vfmv_v_f_f32m4(MlasErfConstants_RVV.Exp_log2_hi, vl);
        vfloat32m4_t fx = __riscv_vfmadd_vv_f32m4(r, para, r_big, vl);
        para = __riscv_vfmv_v_f_f32m4(MlasErfConstants_RVV.Exp_log2_lo, vl);
        fx = __riscv_vfmadd_vv_f32m4(r, para, fx, vl);

        // y = exp(fx)
        vfloat32m4_t y = __riscv_vfmv_v_f_f32m4(MlasErfConstants_RVV.Exp_P0, vl);
        para = __riscv_vfmv_v_f_f32m4(MlasErfConstants_RVV.Exp_P1, vl);
        y = __riscv_vfmadd_vv_f32m4(y, fx, para, vl);
        para = __riscv_vfmv_v_f_f32m4(MlasErfConstants_RVV.Exp_P2, vl);
        y = __riscv_vfmadd_vv_f32m4(y, fx, para, vl);
        para = __riscv_vfmv_v_f_f32m4(MlasErfConstants_RVV.Exp_P3, vl);
        y = __riscv_vfmadd_vv_f32m4(y, fx, para, vl);
        para = __riscv_vfmv_v_f_f32m4(MlasErfConstants_RVV.Exp_P4, vl);
        y = __riscv_vfmadd_vv_f32m4(y, fx, para, vl);
        para = __riscv_vfmv_v_f_f32m4(MlasErfConstants_RVV.Exp_P5, vl);
        y = __riscv_vfmadd_vv_f32m4(y, fx, para, vl);
        para = __riscv_vfmv_v_f_f32m4(MlasErfConstants_RVV.Exp_P6, vl);
        y = __riscv_vfmadd_vv_f32m4(y, fx, para, vl);

        // 1.0 - exp(fx) * 2^INT(r)
        vint32m4_t emm0 = __riscv_vfcvt_x_f_v_i32m4(r, vl);
        size_t const_127 = 127;
        emm0 = __riscv_vadd_vx_i32m4(emm0, const_127, vl);
        size_t const_23 = 23;
        emm0 = __riscv_vsll_vx_i32m4(emm0, const_23, vl);
        vfloat32m4_t r_PowerOf2 = __riscv_vreinterpret_v_i32m4_f32m4(emm0);
        y = __riscv_vfmul_vv_f32m4(y, r_PowerOf2, vl);
        para = __riscv_vfmv_v_f_f32m4(MlasErfConstants_RVV.ErfOne, vl);
        y = __riscv_vfsub_vv_f32m4(para, y, vl);

        // merge r_small and r_big
        y = __riscv_vmerge_vvm_f32m4(r_small, y, split_mask, vl);
        vint32m4_t int32_y = __riscv_vreinterpret_v_f32m4_i32m4(y);
        int32_y = __riscv_vor_vv_i32m4(int32_y, SignMask, vl);
        y = __riscv_vreinterpret_v_i32m4_f32m4(int32_y);
        __riscv_vse32_v_f32m4(output_ptr, y, vl);
    }  // End of For
}  // End of FUNC
