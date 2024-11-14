// Copyright (c) 2023 SpacemiT. All rights reserved.
// Licensed under the MIT License.

#include "mlasi.h"

template <typename T, bool simplified, bool has_skip_output>
void MLASCALL
MlasSkipLayerNormalizationPerTask(const T* input,
                                  const T* skip_input,
                                  T* output,
                                  T* skip_output,
                                  const T* gamma_data,
                                  const T* beta_data,
                                  const T* bias_data,
                                  int64_t hidden_size,
                                  ptrdiff_t task_idx,
                                  float epsilon,
                                  size_t skip_size,
                                  float* mean_out,
                                  float* mean_square_out)
{
    auto offset = task_idx * hidden_size;
    auto* p_input = const_cast<T*>(input + offset);
    auto* p_skip = const_cast<T*>(skip_input + offset % skip_size);

    auto* p_output = output + offset;
    auto* p_skip_input_bias_add_output_data = p_output;
    auto* p_temp_output = p_output;
    auto* p_gamma_data = const_cast<T*>(gamma_data);
    auto* p_beta_data = const_cast<T*>(beta_data);
    auto* p_bias_data = const_cast<T*>(bias_data);

    if constexpr (has_skip_output) {
        p_skip_input_bias_add_output_data = skip_output + offset;
        p_temp_output = p_skip_input_bias_add_output_data;
    }

    if constexpr (std::is_same<T, float>::value) {
        size_t gvl = __riscv_vsetvlmax_e32m4();
        vfloat32m4_t sum = __riscv_vfmv_v_f_f32m4(0.f, gvl);
        vfloat32m4_t sum_sq = __riscv_vfmv_v_f_f32m4(0.f, gvl);
        int64_t length = hidden_size;
        if (p_bias_data == nullptr) {
            while (length > 0) {
                gvl = __riscv_vsetvl_e32m4(length);
                // load data
                vfloat32m4_t src_data = __riscv_vle32_v_f32m4(p_input, gvl);
                vfloat32m4_t skip_data = __riscv_vle32_v_f32m4(p_skip, gvl);
                src_data = __riscv_vfadd_vv_f32m4(src_data, skip_data, gvl);

                if constexpr (!simplified) {
                    sum = __riscv_vfadd_vv_f32m4(sum, src_data, gvl);
                }
                sum_sq = __riscv_vfmacc_vv_f32m4(sum_sq, src_data, src_data, gvl);

                __riscv_vse32_v_f32m4(p_skip_input_bias_add_output_data, src_data, gvl);

                p_input += gvl;
                p_skip_input_bias_add_output_data += gvl;
                p_skip += gvl;
                length -= gvl;
            }
        } else {
            while (length > 0) {
                gvl = __riscv_vsetvl_e32m4(length);
                // load data
                vfloat32m4_t src_data = __riscv_vle32_v_f32m4(p_input, gvl);
                vfloat32m4_t skip_data = __riscv_vle32_v_f32m4(p_skip, gvl);
                src_data = __riscv_vfadd_vv_f32m4(src_data, skip_data, gvl);

                vfloat32m4_t bias_data_v = __riscv_vle32_v_f32m4(p_bias_data, gvl);
                src_data = __riscv_vfadd_vv_f32m4(src_data, bias_data_v, gvl);
                p_bias_data += gvl;

                if constexpr (!simplified) {
                    sum = __riscv_vfadd_vv_f32m4(sum, src_data, gvl);
                }
                sum_sq = __riscv_vfmacc_vv_f32m4(sum_sq, src_data, src_data, gvl);

                __riscv_vse32_v_f32m4(p_skip_input_bias_add_output_data, src_data, gvl);

                p_input += gvl;
                p_skip_input_bias_add_output_data += gvl;
                p_skip += gvl;
                length -= gvl;
            }
        }

        gvl = __riscv_vsetvlmax_e32m1();

        float mean = 0.f;
        vfloat32m1_t zero_v = __riscv_vfmv_v_f_f32m1(0.f, gvl);
        if constexpr (!simplified) {
            vfloat32m1_t mean_v =
                __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(sum, 0), __riscv_vget_v_f32m4_f32m1(sum, 1), gvl);
            mean_v = __riscv_vfadd_vv_f32m1(mean_v, __riscv_vget_v_f32m4_f32m1(sum, 2), gvl);
            mean_v = __riscv_vfadd_vv_f32m1(mean_v, __riscv_vget_v_f32m4_f32m1(sum, 3), gvl);
            mean_v = __riscv_vfredusum_vs_f32m1_f32m1(mean_v, zero_v, gvl);
            mean = __riscv_vfmv_f_s_f32m1_f32(mean_v);
            mean /= hidden_size;
        }

        *mean_out = mean;

        vfloat32m1_t mean_square_v =
            __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(sum_sq, 0), __riscv_vget_v_f32m4_f32m1(sum_sq, 1), gvl);
        mean_square_v = __riscv_vfadd_vv_f32m1(mean_square_v, __riscv_vget_v_f32m4_f32m1(sum_sq, 2), gvl);
        mean_square_v = __riscv_vfadd_vv_f32m1(mean_square_v, __riscv_vget_v_f32m4_f32m1(sum_sq, 3), gvl);
        mean_square_v = __riscv_vfredusum_vs_f32m1_f32m1(mean_square_v, zero_v, gvl);

        float mean_square = __riscv_vfmv_f_s_f32m1_f32(mean_square_v);
        mean_square /= hidden_size;

        if constexpr (simplified) {
            mean_square = sqrt(mean_square + epsilon);
        } else {
            mean_square = sqrt(mean_square - mean * mean + epsilon);
        }

        *mean_square_out = mean_square;

        mean_square = 1.0f / mean_square;
        length = hidden_size;

        if (p_beta_data == nullptr) {
            while (length > 0) {
                gvl = __riscv_vsetvl_e32m4(length);
                vfloat32m4_t src_data = __riscv_vle32_v_f32m4(p_temp_output, gvl);
                vfloat32m4_t gamma_data_v = __riscv_vle32_v_f32m4(p_gamma_data, gvl);
                if constexpr (!simplified) {
                    src_data = __riscv_vfsub_vf_f32m4(src_data, mean, gvl);
                }
                src_data = __riscv_vfmul_vf_f32m4(src_data, mean_square, gvl);
                src_data = __riscv_vfmul_vv_f32m4(src_data, gamma_data_v, gvl);
                __riscv_vse32_v_f32m4(p_output, src_data, gvl);
                p_temp_output += gvl;
                p_output += gvl;
                p_gamma_data += gvl;
                length -= gvl;
            }
        } else {
            while (length > 0) {
                gvl = __riscv_vsetvl_e32m4(length);
                vfloat32m4_t src_data = __riscv_vle32_v_f32m4(p_temp_output, gvl);
                vfloat32m4_t gamma_data_v = __riscv_vle32_v_f32m4(p_gamma_data, gvl);
                if constexpr (!simplified) {
                    src_data = __riscv_vfsub_vf_f32m4(src_data, mean, gvl);
                }
                src_data = __riscv_vfmul_vf_f32m4(src_data, mean_square, gvl);
                src_data = __riscv_vfmul_vv_f32m4(src_data, gamma_data_v, gvl);
                vfloat32m4_t beta_data_v = __riscv_vle32_v_f32m4(p_beta_data, gvl);
                src_data = __riscv_vfadd_vv_f32m4(src_data, beta_data_v, gvl);
                p_beta_data += gvl;
                __riscv_vse32_v_f32m4(p_output, src_data, gvl);
                p_temp_output += gvl;
                p_output += gvl;
                p_gamma_data += gvl;
                length -= gvl;
            }
        }
    } else {
        throw std::runtime_error("MlasSkipLayerNormalizationPerTask unsupported data type");
    }
}

template void MLASCALL
MlasSkipLayerNormalizationPerTask<float, true, true>(const float* input,
                                                     const float* skip_input,
                                                     float* output,
                                                     float* skip_output,
                                                     const float* gamma_data,
                                                     const float* beta_data,
                                                     const float* bias_data,
                                                     int64_t hidden_size,
                                                     ptrdiff_t task_idx,
                                                     float epsilon,
                                                     size_t skip_size,
                                                     float* mean_out,
                                                     float* mean_square_out);

template void MLASCALL
MlasSkipLayerNormalizationPerTask<float, true, false>(const float* input,
                                                      const float* skip_input,
                                                      float* output,
                                                      float* skip_output,
                                                      const float* gamma_data,
                                                      const float* beta_data,
                                                      const float* bias_data,
                                                      int64_t hidden_size,
                                                      ptrdiff_t task_idx,
                                                      float epsilon,
                                                      size_t skip_size,
                                                      float* mean_out,
                                                      float* mean_square_out);

template void MLASCALL
MlasSkipLayerNormalizationPerTask<float, false, true>(const float* input,
                                                      const float* skip_input,
                                                      float* output,
                                                      float* skip_output,
                                                      const float* gamma_data,
                                                      const float* beta_data,
                                                      const float* bias_data,
                                                      int64_t hidden_size,
                                                      ptrdiff_t task_idx,
                                                      float epsilon,
                                                      size_t skip_size,
                                                      float* mean_out,
                                                      float* mean_square_out);

template void MLASCALL
MlasSkipLayerNormalizationPerTask<float, false, false>(const float* input,
                                                       const float* skip_input,
                                                       float* output,
                                                       float* skip_output,
                                                       const float* gamma_data,
                                                       const float* beta_data,
                                                       const float* bias_data,
                                                       int64_t hidden_size,
                                                       ptrdiff_t task_idx,
                                                       float epsilon,
                                                       size_t skip_size,
                                                       float* mean_out,
                                                       float* mean_square_out);

template <typename T, bool simplified>
void
MlasLayerNormalizationPerTask(const T* input,
                              T* output,
                              const T* gamma_data,
                              const T* beta_data,
                              int64_t hidden_size,
                              ptrdiff_t task_idx,
                              float epsilon,
                              float* mean_out,
                              float* mean_square_out)
{
    auto offset = task_idx * hidden_size;
    auto* p_input = const_cast<T*>(input + offset);

    auto* p_output = output + offset;
    auto* p_temp_output = p_output;
    auto* p_gamma_data = const_cast<T*>(gamma_data);
    auto* p_beta_data = const_cast<T*>(beta_data);

    if constexpr (std::is_same<T, float>::value) {
        size_t gvl = __riscv_vsetvlmax_e32m4();
        vfloat32m4_t sum = __riscv_vfmv_v_f_f32m4(0.f, gvl);
        vfloat32m4_t sum_sq = __riscv_vfmv_v_f_f32m4(0.f, gvl);
        int64_t length = hidden_size;
        while (length > 0) {
            gvl = __riscv_vsetvl_e32m4(length);
            // load data
            vfloat32m4_t src_data = __riscv_vle32_v_f32m4(p_input, gvl);

            if constexpr (!simplified) {
                sum = __riscv_vfadd_vv_f32m4(sum, src_data, gvl);
            }
            sum_sq = __riscv_vfmacc_vv_f32m4(sum_sq, src_data, src_data, gvl);

            __riscv_vse32_v_f32m4(p_temp_output, src_data, gvl);

            p_input += gvl;
            p_temp_output += gvl;
            length -= gvl;
        }

        gvl = __riscv_vsetvlmax_e32m1();

        float mean = 0.f;
        vfloat32m1_t zero_v = __riscv_vfmv_v_f_f32m1(0.f, gvl);
        if constexpr (!simplified) {
            vfloat32m1_t mean_v =
                __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(sum, 0), __riscv_vget_v_f32m4_f32m1(sum, 1), gvl);
            mean_v = __riscv_vfadd_vv_f32m1(mean_v, __riscv_vget_v_f32m4_f32m1(sum, 2), gvl);
            mean_v = __riscv_vfadd_vv_f32m1(mean_v, __riscv_vget_v_f32m4_f32m1(sum, 3), gvl);
            mean_v = __riscv_vfredusum_vs_f32m1_f32m1(mean_v, zero_v, gvl);
            mean = __riscv_vfmv_f_s_f32m1_f32(mean_v);
            mean /= hidden_size;
        }

        *mean_out = mean;

        vfloat32m1_t mean_square_v =
            __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(sum_sq, 0), __riscv_vget_v_f32m4_f32m1(sum_sq, 1), gvl);
        mean_square_v = __riscv_vfadd_vv_f32m1(mean_square_v, __riscv_vget_v_f32m4_f32m1(sum_sq, 2), gvl);
        mean_square_v = __riscv_vfadd_vv_f32m1(mean_square_v, __riscv_vget_v_f32m4_f32m1(sum_sq, 3), gvl);
        mean_square_v = __riscv_vfredusum_vs_f32m1_f32m1(mean_square_v, zero_v, gvl);

        float mean_square = __riscv_vfmv_f_s_f32m1_f32(mean_square_v);
        mean_square /= hidden_size;

        if constexpr (simplified) {
            mean_square = sqrt(mean_square + epsilon);
        } else {
            mean_square = sqrt(mean_square - mean * mean + epsilon);
        }

        *mean_square_out = mean_square;

        mean_square = 1.0f / mean_square;
        length = hidden_size;
        p_temp_output = p_output;

        if (p_gamma_data == nullptr && p_beta_data == nullptr) {
            while (length > 0) {
                gvl = __riscv_vsetvl_e32m4(length);
                vfloat32m4_t src_data = __riscv_vle32_v_f32m4(p_temp_output, gvl);
                if constexpr (!simplified) {
                    src_data = __riscv_vfsub_vf_f32m4(src_data, mean, gvl);
                }
                src_data = __riscv_vfmul_vf_f32m4(src_data, mean_square, gvl);
                __riscv_vse32_v_f32m4(p_output, src_data, gvl);
                p_temp_output += gvl;
                p_output += gvl;
                length -= gvl;
            }
        } else if (p_beta_data == nullptr) {
            while (length > 0) {
                gvl = __riscv_vsetvl_e32m4(length);
                vfloat32m4_t src_data = __riscv_vle32_v_f32m4(p_temp_output, gvl);
                vfloat32m4_t gamma_data_v = __riscv_vle32_v_f32m4(p_gamma_data, gvl);
                if constexpr (!simplified) {
                    src_data = __riscv_vfsub_vf_f32m4(src_data, mean, gvl);
                }
                src_data = __riscv_vfmul_vf_f32m4(src_data, mean_square, gvl);
                src_data = __riscv_vfmul_vv_f32m4(src_data, gamma_data_v, gvl);
                __riscv_vse32_v_f32m4(p_output, src_data, gvl);
                p_temp_output += gvl;
                p_output += gvl;
                p_gamma_data += gvl;
                length -= gvl;
            }
        } else if (p_gamma_data != nullptr) {
            while (length > 0) {
                gvl = __riscv_vsetvl_e32m4(length);
                vfloat32m4_t src_data = __riscv_vle32_v_f32m4(p_temp_output, gvl);
                vfloat32m4_t gamma_data_v = __riscv_vle32_v_f32m4(p_gamma_data, gvl);
                if constexpr (!simplified) {
                    src_data = __riscv_vfsub_vf_f32m4(src_data, mean, gvl);
                }
                src_data = __riscv_vfmul_vf_f32m4(src_data, mean_square, gvl);
                src_data = __riscv_vfmul_vv_f32m4(src_data, gamma_data_v, gvl);
                vfloat32m4_t beta_data_v = __riscv_vle32_v_f32m4(p_beta_data, gvl);
                src_data = __riscv_vfadd_vv_f32m4(src_data, beta_data_v, gvl);
                p_beta_data += gvl;
                __riscv_vse32_v_f32m4(p_output, src_data, gvl);
                p_temp_output += gvl;
                p_output += gvl;
                p_gamma_data += gvl;
                length -= gvl;
            }
        } else {
            throw std::runtime_error("MlasLayerNormalizationPerTask unsupported scale nullptr");
        }
    } else {
        throw std::runtime_error("MlasLayerNormalizationPerTask unsupported data type");
    }
}

template void MLASCALL
MlasLayerNormalizationPerTask<float, true>(const float* input,
                                           float* output,
                                           const float* gamma_data,
                                           const float* beta_data,
                                           int64_t hidden_size,
                                           ptrdiff_t task_idx,
                                           float epsilon,
                                           float* mean_out,
                                           float* mean_square_out);

template void MLASCALL
MlasLayerNormalizationPerTask<float, false>(const float* input,
                                            float* output,
                                            const float* gamma_data,
                                            const float* beta_data,
                                            int64_t hidden_size,
                                            ptrdiff_t task_idx,
                                            float epsilon,
                                            float* mean_out,
                                            float* mean_square_out);