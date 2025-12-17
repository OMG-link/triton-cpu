#include "ggml_def.h"


void ggml_vec_dot_q4_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_K * GGML_RESTRICT x = static_cast<const block_q4_K*>(vx);
    const block_q8_K * GGML_RESTRICT y = static_cast<const block_q8_K*>(vy);

    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];


#if defined __riscv_v

    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    float sumf = 0;
    const int vector_length = __riscv_vlenb() * 8;

    switch (vector_length) {
    case 256:
        for (int i = 0; i < nb; ++i) {

            size_t vl = 8;

            const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
            const float dmin = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

            vint16mf2_t q8sums_0 = __riscv_vlse16_v_i16mf2(y[i].bsums, 4, vl);
            vint16mf2_t q8sums_1 = __riscv_vlse16_v_i16mf2(y[i].bsums+1, 4, vl);
            vint16mf2_t q8sums   = __riscv_vadd_vv_i16mf2(q8sums_0, q8sums_1, vl);

            memcpy(utmp, x[i].scales, 12);
            utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
            const uint32_t uaux = utmp[1] & kmask1;
            utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
            utmp[2] = uaux;
            utmp[0] &= kmask1;

            vuint8mf4_t mins8  = __riscv_vle8_v_u8mf4(mins, vl);
            vint16mf2_t v_mins = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vzext_vf2_u16mf2(mins8, vl));
            vint32m1_t  prod   = __riscv_vwmul_vv_i32m1(q8sums, v_mins, vl);

            vint32m1_t sumi = __riscv_vredsum_vs_i32m1_i32m1(prod, __riscv_vmv_v_x_i32m1(0, 1), vl);
            sumf -= dmin * __riscv_vmv_x_s_i32m1_i32(sumi);

            const uint8_t * GGML_RESTRICT q4 = x[i].qs;
            const int8_t  * GGML_RESTRICT q8 = y[i].qs;

            vl = 32;

            int32_t sum_1 = 0;
            int32_t sum_2 = 0;

            vint16m1_t vzero = __riscv_vmv_v_x_i16m1(0, 1);

            for (int j = 0; j < QK_K/64; ++j) {
                // load Q4
                vuint8m1_t q4_x = __riscv_vle8_v_u8m1(q4, vl);

                // load Q8 and multiply it with lower Q4 nibble
                vint8m1_t  q8_0 = __riscv_vle8_v_i8m1(q8, vl);
                vint8m1_t  q4_0 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(q4_x, 0x0F, vl));
                vint16m2_t qv_0 = __riscv_vwmul_vv_i16m2(q4_0, q8_0, vl);
                vint16m1_t vs_0 = __riscv_vredsum_vs_i16m2_i16m1(qv_0, vzero, vl);

                sum_1 += __riscv_vmv_x_s_i16m1_i16(vs_0) * scales[2*j+0];

                // load Q8 and multiply it with upper Q4 nibble
                vint8m1_t  q8_1 = __riscv_vle8_v_i8m1(q8+32, vl);
                vint8m1_t  q4_1 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(q4_x, 0x04, vl));
                vint16m2_t qv_1 = __riscv_vwmul_vv_i16m2(q4_1, q8_1, vl);
                vint16m1_t vs_1 = __riscv_vredsum_vs_i16m2_i16m1(qv_1, vzero, vl);

                sum_2 += __riscv_vmv_x_s_i16m1_i16(vs_1) * scales[2*j+1];

                q4 += 32;    q8 += 64;

            }

            sumf += d*(sum_1 + sum_2);

        }
        break;
    case 128:
        for (int i = 0; i < nb; ++i) {
            const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
            const float dmin = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

            float ftmp, ft2;
            const uint8_t * GGML_RESTRICT q40;
            const uint8_t * GGML_RESTRICT q41;
            const uint8_t * GGML_RESTRICT q42;
            const uint8_t * GGML_RESTRICT q43;
            const int8_t  * GGML_RESTRICT q80;
            const int8_t  * GGML_RESTRICT q81;
            const int8_t  * GGML_RESTRICT q82;
            const int8_t  * GGML_RESTRICT q83;
            int s0, s1, s2, s3;

            __asm__ __volatile__(
                "li %[s1], 8\n\t"
                "vsetivli zero, 4, e32, m1, ta, ma\n\t"
                "vle32.v v1, (%[s6b])\n\t"
                "vslide1down.vx v1, v1, zero\n\t"
                "vmv.v.x v16, zero\n\t"
                "vslidedown.vi v2, v1, 2\n\t"
                "vmv1r.v v3, v2\n\t"
                "vslideup.vi v2, v3, 1\n\t" // {aux[2], aux[2]}
                "vsetivli zero, 2, e32, m1, ta, ma\n\t"
                "vmv.v.i v4, 4\n\t"
                "vand.vx v8, v1, %[kmask1]\n\t"
                "vslide1up.vx v5, v4, zero\n\t" // {0, 4}
                "vsrl.vi v6, v1, 6\n\t"
                "vsrl.vv v7, v2, v5\n\t"
                "vsse32.v v8, (%[utmp]), %[s1]\n\t"
                "vand.vx v0, v6, %[kmask3]\n\t"
                "vand.vx v2, v7, %[kmask2]\n\t"
                "vsll.vi v6, v0, 4\n\t"
                "addi %[s0], %[utmp], 4\n\t"
                "vor.vv v1, v6, v2\n\t"
                "vsse32.v v1, (%[s0]), %[s1]\n\t"
                "vsetivli zero, 8, e16, m1, ta, ma\n\t"
                "vle32.v v2, (%[bsums])\n\t"
                "vnsrl.wi v0, v2, 0\n\t"
                "vnsrl.wi v1, v2, 16\n\t"
                "vadd.vv v2, v0, v1\n\t"
                "vle8.v v3, (%[mins])\n\t"
                "vzext.vf2 v4, v3\n\t"
                "vwmul.vv v6, v4, v2\n\t"
                "vsetivli zero, 4, e32, m1, ta, ma\n\t"
                "vredsum.vs v0, v6, v16\n\t"
                "vredsum.vs v0, v7, v0\n\t"
                "vfcvt.f.x.v v0, v0\n\t"
                "vfmv.f.s %[ftmp], v0\n\t"
                "vsetivli zero, 16, e8, m1, ta, ma\n\t"
                "vle8.v v0, (%[xs])\n\t"
                "fnmsub.s %[sumf], %[dmin], %[ftmp], %[sumf]\n\t"
                "addi %[q40], %[xs], 64\n\t"
                "addi %[q41], %[xs], 16\n\t"
                "addi %[q42], %[xs], 32\n\t"
                "addi %[q43], %[xs], 48\n\t"
                "addi %[q80], %[ys], 64\n\t"
                "vle8.v v1, (%[q41])\n\t"
                "vle8.v v2, (%[q42])\n\t"
                "addi %[q81], %[ys], 16\n\t"
                "addi %[q41], %[q41], 64\n\t"
                "addi %[q82], %[ys], 32\n\t"
                "vle8.v v3, (%[q43])\n\t"
                "vle8.v v8, (%[ys])\n\t"
                "addi %[q42], %[q42], 64\n\t"
                "addi %[q83], %[ys], 48\n\t"
                "addi %[q43], %[q43], 64\n\t"
                "vsrl.vi v4, v0, 4\n\t"
                "vle8.v v9, (%[q81])\n\t"
                "vle8.v v10, (%[q82])\n\t"
                "vand.vi v0, v0, 0xF\n\t"
                "addi %[q81], %[q81], 64\n\t"
                "vsrl.vi v5, v1, 4\n\t"
                "addi %[q82], %[q82], 64\n\t"
                "vle8.v v11, (%[q83])\n\t"
                "vle8.v v12, (%[q80])\n\t"
                "vand.vi v1, v1, 0xF\n\t"
                "addi %[q83], %[q83], 64\n\t"
                "vsrl.vi v6, v2, 4\n\t"
                "addi %[q80], %[q80], 64\n\t"
                "vle8.v v13, (%[q81])\n\t"
                "vle8.v v14, (%[q82])\n\t"
                "vand.vi v2, v2, 0xF\n\t"
                "addi %[q81], %[q81], 64\n\t"
                "vsrl.vi v7, v3, 4\n\t"
                "addi %[q82], %[q82], 64\n\t"
                "vwmul.vv v16, v0, v8\n\t"
                "vle8.v v15, (%[q83])\n\t"
                "vle8.v v0, (%[q40])\n\t"
                "vand.vi v3, v3, 0xF\n\t"
                "addi %[q83], %[q83], 64\n\t"
                "vwmul.vv v24, v2, v12\n\t"
                "vwmul.vv v20, v4, v10\n\t"
                "vwmul.vv v28, v6, v14\n\t"
                "vwmacc.vv v16, v1, v9\n\t"
                "vle8.v v1, (%[q41])\n\t"
                "vle8.v v2, (%[q42])\n\t"
                "vwmacc.vv v24, v3, v13\n\t"
                "vwmacc.vv v20, v5, v11\n\t"
                "vwmacc.vv v28, v7, v15\n\t"
                "addi %[q40], %[q80], 64\n\t"
                "addi %[q41], %[q81], 64\n\t"
                "vle8.v v3, (%[q43])\n\t"
                "vle8.v v8, (%[q80])\n\t"
                "addi %[q42], %[q82], 64\n\t"
                "addi %[q43], %[q83], 64\n\t"
                "vsrl.vi v4, v0, 4\n\t"
                "vle8.v v9, (%[q81])\n\t"
                "vle8.v v10, (%[q82])\n\t"
                "vand.vi v0, v0, 0xF\n\t"
                "vsrl.vi v5, v1, 4\n\t"
                "vsrl.vi v7, v3, 4\n\t"
                "vand.vi v3, v3, 0xF\n\t"
                "vle8.v v11, (%[q83])\n\t"
                "vle8.v v12, (%[q40])\n\t"
                "vand.vi v1, v1, 0xF\n\t"
                "vsrl.vi v6, v2, 4\n\t"
                "vand.vi v2, v2, 0xF\n\t"
                "vwmul.vv v18, v0, v8\n\t"
                "vle8.v v13, (%[q41])\n\t"
                "vle8.v v14, (%[q42])\n\t"
                "vwmul.vv v26, v2, v12\n\t"
                "vwmul.vv v22, v4, v10\n\t"
                "vwmul.vv v30, v6, v14\n\t"
                "vwmacc.vv v18, v1, v9\n\t"
                "vle8.v v15, (%[q43])\n\t"
                "vwmacc.vv v26, v3, v13\n\t"
                "vwmacc.vv v22, v5, v11\n\t"
                "vwmacc.vv v30, v7, v15\n\t"
                "vmv.v.x v0, zero\n\t"
                "vsetivli zero, 16, e16, m2, ta, ma\n\t"
                "vwredsum.vs v4, v16, v0\n\t"
                "lbu %[s0], 0(%[scale])\n\t"
                "vwredsum.vs v5, v20, v0\n\t"
                "lbu %[s1], 1(%[scale])\n\t"
                "vwredsum.vs v6, v24, v0\n\t"
                "lbu %[s2], 2(%[scale])\n\t"
                "vwredsum.vs v7, v28, v0\n\t"
                "lbu %[s3], 3(%[scale])\n\t"
                "vwredsum.vs v8, v18, v0\n\t"
                "lbu %[q40], 4(%[scale])\n\t"
                "vwredsum.vs v9, v22, v0\n\t"
                "lbu %[q41], 5(%[scale])\n\t"
                "vwredsum.vs v10, v26, v0\n\t"
                "lbu %[q42], 6(%[scale])\n\t"
                "vwredsum.vs v11, v30, v0\n\t"
                "lbu %[q43], 7(%[scale])\n\t"
                "vsetivli zero, 4, e32, m1, ta, ma\n\t"
                "vmul.vx v0, v4, %[s0]\n\t"
                "vmul.vx v1, v8, %[q40]\n\t"
                "vmacc.vx v0, %[s1], v5\n\t"
                "vmacc.vx v1, %[q41], v9\n\t"
                "vmacc.vx v0, %[s2], v6\n\t"
                "vmacc.vx v1, %[q42], v10\n\t"
                "vmacc.vx v0, %[s3], v7\n\t"
                "vmacc.vx v1, %[q43], v11\n\t"
                "vfcvt.f.x.v v0, v0\n\t"
                "vfcvt.f.x.v v1, v1\n\t"
                "vfmv.f.s %[ft2], v0\n\t"
                "vfmv.f.s %[ftmp], v1\n\t"
                "fadd.s %[ft2], %[ft2], %[ftmp]\n\t"
                "fmadd.s %[sumf], %[d], %[ft2], %[sumf]"
                : [ftmp] "=&f" (ftmp), [sumf] "+&f" (sumf), [ft2] "=&f" (ft2)
                , [s0] "=&r" (s0), [s1] "=&r" (s1), [s2] "=&r" (s2), [s3] "=&r" (s3)
                , [q40] "=&r" (q40), [q41] "=&r" (q41), [q42] "=&r" (q42), [q43] "=&r" (q43)
                , [q80] "=&r" (q80), [q81] "=&r" (q81), [q82] "=&r" (q82), [q83] "=&r" (q83)
                : [d] "f" (d), [ys] "r" (y[i].qs), [xs] "r" (x[i].qs), [scale] "r" (scales)
                , [bsums] "r" (y[i].bsums), [mins] "r" (mins), [utmp] "r" (utmp)
                , [s6b] "r" (&x[i]), [kmask1] "r" (kmask1), [dmin] "f" (dmin)
                , [kmask2] "r" (kmask2), [kmask3] "r" (kmask3)
                : "memory"
                , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
                , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
                , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
            );
        }
        break;
    default:
        assert(false && "Unsupported vector length");
        break;
    }

    *s = sumf;

#else

    UNUSED(x);
    UNUSED(y);
    UNUSED(kmask1);
    UNUSED(kmask2);
    UNUSED(kmask3);
    UNUSED(nb);
    UNUSED(utmp);

    ggml_vec_dot_q4_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}
