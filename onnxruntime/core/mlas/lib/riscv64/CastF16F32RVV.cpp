// Copyright (c) 2023 SpacemiT. All rights reserved.
// Licensed under the MIT License.

#include "mlasi.h"

void
MlasCastF32ToF16Kernel_RVV(const float *in, unsigned short *out, size_t N)
{
    asm volatile(
        "srli           t1, %[LEN], 5       \n\t"
        "blez           t1, MAIN_F2H_TAIL   \n\t"

        "MAIN_F2H:                          \n\t"
        "vsetvli        t0, zero, e32, m4   \n\t"
        "vle32.v        v0, (%[IN])         \n\t"
        "vsetvli        t0, zero, e16, m2   \n\t"
        "vfncvt.f.f.w   v4, v0              \n\t"
        "vse16.v        v4, (%[DST])        \n\t"
        "addi           %[IN], %[IN], 128   \n\t"
        "addi           %[DST], %[DST], 64  \n\t"
        "addi           t1, t1, -1          \n\t"
        "bgtz           t1, MAIN_F2H        \n\t"

        "MAIN_F2H_TAIL:                     \n\t"
        "andi           t1, %[LEN], 31      \n\t"
        "blez           t1, MAIN_F2H_OUT    \n\t"

        "vsetvli        t0, t1, e32, m4     \n\t"
        "vle32.v        v0, (%[IN])         \n\t"
        "vsetvli        t0, zero, e16, m2   \n\t"
        "vfncvt.f.f.w   v4, v0              \n\t"
        "vse16.v        v4, (%[DST])        \n\t"

        "MAIN_F2H_OUT:                      \n\t"

        : [ IN ] "+r"(in), [ DST ] "+r"(out)
        : [ LEN ] "r"(N)
        : "t0", "t1", "v0", "v1", "v2", "v3", "v4", "v5");
}

void
MlasCastF16ToF32Kernel_RVV(const unsigned short *in, float *out, size_t N)
{
    asm volatile(
        "srli           t1, %[LEN], 5       \n\t"
        "blez           t1, MAIN_H2F_TAIL   \n\t"

        "MAIN_H2F:                          \n\t"
        "vsetvli        t0, zero, e16, m2   \n\t"
        "vle16.v        v0, (%[IN])         \n\t"
        "vsetvli        t0, zero, e32, m4   \n\t"
        "vfncvt.f.f.w   v4, v0              \n\t"
        "vse32.v        v4, (%[DST])        \n\t"
        "addi           %[IN], %[IN], 64    \n\t"
        "addi           %[DST], %[DST], 128 \n\t"
        "addi           t1, t1, -1          \n\t"
        "bgtz           t1, MAIN_H2F        \n\t"

        "MAIN_H2F_TAIL:                     \n\t"
        "andi           t1, %[LEN], 31      \n\t"
        "blez           t1, MAIN_H2F_OUT    \n\t"

        "vsetvli        t0, t1, e32, m4     \n\t"
        "vle32.v        v0, (%[IN])         \n\t"
        "vsetvli        t0, zero, e16, m2   \n\t"
        "vfncvt.f.f.w   v4, v0              \n\t"
        "vse16.v        v4, (%[DST])        \n\t"

        "MAIN_H2F_OUT:                      \n\t"

        : [ IN ] "+r"(in), [ DST ] "+r"(out)
        : [ LEN ] "r"(N)
        : "t0", "t1", "v0", "v1", "v4", "v5", "v6", "v7");
}