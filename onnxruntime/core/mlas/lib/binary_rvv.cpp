// Copyright (c) 2023 SpacemiT. All rights reserved.
// Licensed under the MIT License.

#include "mlasi.h"

template <typename T, bool IsScalarB>
void MLASCALL
MlasAdd(const T* A, const T* B, T* C, size_t count)
{
    if constexpr (IsScalarB) {
        if constexpr (std::is_same<T, float>::value) {
            auto rhs = B[0];
            __asm__ volatile(
                "LOOP%=:                                  \t\n"
                "vsetvli  t0,       %[n],     e32,      m4\t\n"
                "sub      %[n],     %[n],     t0          \t\n"
                "slli     t0,       t0,       2           \t\n"
                "vle32.v  v0,       (%[lhs])              \t\n"
                "add      %[lhs],   %[lhs],   t0          \t\n"
                "vfadd.vf v0,       v0,       %[rhs]      \t\n"
                "vse32.v  v0,      (%[y])                 \t\n"
                "add      %[y],     %[y],     t0          \t\n"
                "bnez     %[n],     LOOP%=                \t\n"
                : [ n ] "+r"(count), [ lhs ] "+r"(A), [ y ] "+r"(C)
                : [ rhs ] "f"(rhs)
                : "cc", "t0");
        }
    } else {
        if constexpr (std::is_same<T, float>::value) {
            __asm__ volatile(
                "LOOP%=:                                  \t\n"
                "vsetvli  t0,       %[n],     e32,      m4\t\n"
                "sub      %[n],     %[n],     t0          \t\n"
                "slli     t0,       t0,       2           \t\n"
                "vle32.v  v0,       (%[lhs])              \t\n"
                "add      %[lhs],   %[lhs],   t0          \t\n"
                "vle32.v  v8,       (%[rhs])              \t\n"
                "add      %[rhs],   %[rhs],   t0          \t\n"
                "vfadd.vv v0,       v0,       v8          \t\n"
                "vse32.v  v0,      (%[y])                 \t\n"
                "add      %[y],     %[y],     t0          \t\n"
                "bnez     %[n],     LOOP%=                \t\n"
                : [ n ] "+r"(count), [ lhs ] "+r"(A), [ rhs ] "+r"(B), [ y ] "+r"(C)
                :
                : "cc", "t0");
        }
    }
}

template <typename T, bool IsScalarB>
void MLASCALL
MlasMul(const T* A, const T* B, T* C, size_t count)
{
    if constexpr (IsScalarB) {
        if constexpr (std::is_same<T, float>::value) {
            auto rhs = B[0];
            __asm__ volatile(
                "LOOP%=:                                  \t\n"
                "vsetvli  t0,       %[n],     e32,      m4\t\n"
                "sub      %[n],     %[n],     t0          \t\n"
                "slli     t0,       t0,       2           \t\n"
                "vle32.v  v0,       (%[lhs])              \t\n"
                "add      %[lhs],   %[lhs],   t0          \t\n"
                "vfmul.vf v0,       v0,       %[rhs]      \t\n"
                "vse32.v  v0,      (%[y])                 \t\n"
                "add      %[y],     %[y],     t0          \t\n"
                "bnez     %[n],     LOOP%=                \t\n"
                : [ n ] "+r"(count), [ lhs ] "+r"(A), [ y ] "+r"(C)
                : [ rhs ] "f"(rhs)
                : "cc", "t0");
        }
    } else {
        if constexpr (std::is_same<T, float>::value) {
            __asm__ volatile(
                "LOOP%=:                                  \t\n"
                "vsetvli  t0,       %[n],     e32,      m4\t\n"
                "sub      %[n],     %[n],     t0          \t\n"
                "slli     t0,       t0,       2           \t\n"
                "vle32.v  v0,       (%[lhs])              \t\n"
                "add      %[lhs],   %[lhs],   t0          \t\n"
                "vle32.v  v8,       (%[rhs])              \t\n"
                "add      %[rhs],   %[rhs],   t0          \t\n"
                "vfmul.vv v0,       v0,       v8          \t\n"
                "vse32.v  v0,      (%[y])                 \t\n"
                "add      %[y],     %[y],     t0          \t\n"
                "bnez     %[n],     LOOP%=                \t\n"
                : [ n ] "+r"(count), [ lhs ] "+r"(A), [ rhs ] "+r"(B), [ y ] "+r"(C)
                :
                : "cc", "t0");
        }
    }
}

template void MLASCALL
MlasAdd<float, true>(const float*, const float*, float*, size_t);

template void MLASCALL
MlasAdd<float, false>(const float*, const float*, float*, size_t);

template void MLASCALL
MlasMul<float, true>(const float*, const float*, float*, size_t);

template void MLASCALL
MlasMul<float, false>(const float*, const float*, float*, size_t);