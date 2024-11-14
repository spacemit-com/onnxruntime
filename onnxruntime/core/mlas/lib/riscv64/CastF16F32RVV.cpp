// Copyright (c) 2023 SpacemiT. All rights reserved.
// Licensed under the MIT License.

#include "mlasi.h"

void
MlasCastF32ToF16Kernel_RVV(const float *in, unsigned short *out, size_t N)
{
#if 0
    __fp16 *dst = reinterpret_cast<__fp16 *>(out);
    for (size_t i = 0; i < N; i++) {
        dst[i] = static_cast<__fp16>(in[i]);
    }
#else
    __asm__ volatile(
        "LOOP%=:                            \t\n"
        "vsetvli        t0, %[n], e32, m4   \t\n"
        "slli           t1, t0, 1           \t\n"
        "slli           t2, t0, 2           \t\n"
        "vle32.v        v0, (%[IN])         \t\n"
        "add            %[IN], %[IN], t2    \t\n"
        "vsetvli        t0, %[n], e16, m2   \t\n"
        "vfncvt.f.f.w   v4, v0              \n\t"
        "vse16.v        v4, (%[DST])        \n\t"
        "add            %[DST], %[DST], t1  \t\n"
        "sub            %[n],  %[n], t0     \t\n"
        "bnez           %[n], LOOP%=        \t\n"

        : [ IN ] "+r"(in), [ DST ] "+r"(out), [ n ] "+r"(N)
        :
        : "cc", "t0", "t1", "t2");
#endif
}

void
MlasCastF16ToF32Kernel_RVV(const unsigned short *in, float *out, size_t N)
{
#if 0
    const __fp16 *src = reinterpret_cast<const __fp16 *>(in);
    for (size_t i = 0; i < N; i++) {
        out[i] = static_cast<float>(src[i]);
    }
#else
    __asm__ volatile(
        "LOOP%=:                            \t\n"
        "vsetvli        t0, %[n], e16, m2   \t\n"
        "slli           t1, t0, 2           \t\n"
        "slli           t2, t0, 1           \t\n"
        "vle16.v        v0, (%[IN])         \t\n"
        "add            %[IN], %[IN], t2    \t\n"
        "vfwcvt.f.f.v   v4, v0              \n\t"
        "vsetvli        t0, %[n], e32, m4   \n\t"
        "vse32.v        v4, (%[DST])        \n\t"
        "add            %[DST], %[DST], t1  \t\n"
        "sub            %[n],  %[n], t0     \t\n"
        "bnez           %[n], LOOP%=        \t\n"

        : [ IN ] "+r"(in), [ DST ] "+r"(out), [ n ] "+r"(N)
        :
        : "cc", "t0", "t1", "t2");
#endif
}