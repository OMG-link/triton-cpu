#include "ggml_def.h"
// #include "riscv_subt.h"

using block_q4_Kx32 = block_q4_Kx<32>;
using block_q8_Kx12 = block_q8_Kx<12>;
using block_q8_Kx4 = block_q8_Kx<4>;

void ggml_gemm_q4_K_8x32_q8_K(int n, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx, const void *GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(blocklen);

    const block_q4_Kx32 *b_ptr_start = (const block_q4_Kx32 *)vx;
    const block_q8_Kx4 *a_ptr_start = (const block_q8_Kx4 *)vy;
    size_t vl = 32;

    uint32_t utmp[16];

    const uint8_t *scales = (const uint8_t *)&utmp[0];
    const uint8_t *mins = (const uint8_t *)&utmp[8];

    // int anr = nr - nr % 16;

    for (int64_t y = 0; y < nr / 4; y++) { // M

        const block_q8_Kx4 *a_ptr = a_ptr_start + (y * nb);
        for (int64_t x = 0; x < nc / 32; x++) { // N

            const block_q4_Kx32 *b_ptr = b_ptr_start + (x * nb);

            //行累加和
            vfloat32m4_t sum_rows0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t sum_rows1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t sum_rows2 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t sum_rows3 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t sum_min_rows0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t sum_min_rows1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t sum_min_rows2 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t sum_min_rows3 = __riscv_vfmv_v_f_f32m4(0.0f, vl);

            for (int64_t b = 0; b < nb; b++) { // K

                vint32m4_t sum_block0 = __riscv_vmv_v_x_i32m4(0, vl); // 所有元素置零
                vint32m4_t sum_min_block0 = __riscv_vmv_v_x_i32m4(0, vl);
                vint32m4_t sum_block1 = __riscv_vmv_v_x_i32m4(0, vl); // 所有元素置零
                vint32m4_t sum_min_block1 = __riscv_vmv_v_x_i32m4(0, vl);
                vint32m4_t sum_block2 = __riscv_vmv_v_x_i32m4(0, vl); // 所有元素置零
                vint32m4_t sum_min_block2 = __riscv_vmv_v_x_i32m4(0, vl);
                vint32m4_t sum_block3 = __riscv_vmv_v_x_i32m4(0, vl); // 所有元素置零
                vint32m4_t sum_min_block3 = __riscv_vmv_v_x_i32m4(0, vl);

                // Loop to iterate over the eight sub blocks of a super block - two sub blocks are processed per iteration
                for (int sb = 0; sb < QK_K / 64; sb++) { // QK_K / 64=4,一次处理src0的64列  //
                    vint16m2_t suml_32_0 = __riscv_vmv_v_x_i16m2(0, vl);
                    vint16m2_t suml_32_1 = __riscv_vmv_v_x_i16m2(0, vl);
                    vint16m2_t suml_32_2 = __riscv_vmv_v_x_i16m2(0, vl);
                    vint16m2_t suml_32_3 = __riscv_vmv_v_x_i16m2(0, vl);
                    vint16m2_t suml_64_0 = __riscv_vmv_v_x_i16m2(0, vl);
                    vint16m2_t suml_64_1 = __riscv_vmv_v_x_i16m2(0, vl);
                    vint16m2_t suml_64_2 = __riscv_vmv_v_x_i16m2(0, vl);
                    vint16m2_t suml_64_3 = __riscv_vmv_v_x_i16m2(0, vl);
                    
                    for (int z = 0; z < 32; z++) {

                        //权重矩阵的0-3列和32-35列
                        vuint8m1_t src0_0 = __riscv_vle8_v_u8m1(b_ptr[b].qs + sb * 1024 + z * 32, vl);
                        vint8m1_t low4bits_0 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(src0_0, 0x0F, vl));
                        vint8m1_t high4bits_0 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(src0_0, 4, vl));
                        //输出矩阵4行的一个位置的低32个数和32-64个数
                        const int64_t ssb = sb * 256 + z * 4;
                        suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[b].qs[ssb], low4bits_0, vl);
                        suml_32_1 = __riscv_vwmacc_vx_i16m2(suml_32_1, a_ptr[b].qs[ssb + 1], low4bits_0, vl);
                        suml_32_2 = __riscv_vwmacc_vx_i16m2(suml_32_2, a_ptr[b].qs[ssb + 2], low4bits_0, vl);
                        suml_32_3 = __riscv_vwmacc_vx_i16m2(suml_32_3, a_ptr[b].qs[ssb + 3], low4bits_0, vl);
                        suml_64_0 = __riscv_vwmacc_vx_i16m2(suml_64_0, a_ptr[b].qs[ssb + 128], high4bits_0, vl);
                        suml_64_1 = __riscv_vwmacc_vx_i16m2(suml_64_1, a_ptr[b].qs[ssb + 128 + 1], high4bits_0, vl);
                        suml_64_2 = __riscv_vwmacc_vx_i16m2(suml_64_2, a_ptr[b].qs[ssb + 128 + 2], high4bits_0, vl);
                        suml_64_3 = __riscv_vwmacc_vx_i16m2(suml_64_3, a_ptr[b].qs[ssb + 128 + 3], high4bits_0, vl);
                    }

                    // 处理scales和mins 
                    // 是否可以向量化实现 
                    memcpy(utmp, b_ptr[b].scales + 96 * sb, 48);
                    utmp[15] = ((utmp[11] >> 4) & kmask2) | (((utmp[7] >> 6) & kmask3) << 4);
                    utmp[14] = ((utmp[10] >> 4) & kmask2) | (((utmp[6] >> 6) & kmask3) << 4);
                    utmp[13] = ((utmp[9] >> 4) & kmask2) | (((utmp[5] >> 6) & kmask3) << 4);
                    utmp[12] = ((utmp[8] >> 4) & kmask2) | (((utmp[4] >> 6) & kmask3) << 4);
                    uint32_t uaux = utmp[4] & kmask1;
                    uint32_t uaux1 = utmp[5] & kmask1;
                    uint32_t uaux2 = utmp[6] & kmask1;
                    uint32_t uaux3 = utmp[7] & kmask1;
                    utmp[4] = (utmp[8] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
                    utmp[5] = (utmp[9] & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
                    utmp[6] = (utmp[10] & kmask2) | (((utmp[2] >> 6) & kmask3) << 4);
                    utmp[7] = (utmp[11] & kmask2) | (((utmp[3] >> 6) & kmask3) << 4);
                    utmp[8] = uaux;
                    utmp[9] = uaux1;
                    utmp[10] = uaux2;
                    utmp[11] = uaux3;
                    utmp[0] &= kmask1;
                    utmp[1] &= kmask1;
                    utmp[2] &= kmask1;
                    utmp[3] &= kmask1;

                    vint16m2_t vec_scales_32 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(__riscv_vle8_v_u8m1(scales, vl), vl));
                    vint16m2_t vec_mins_32 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(__riscv_vle8_v_u8m1(mins, vl), vl));

                    memcpy(utmp, b_ptr[b].scales + 48 + 96 * sb, 48);
                    utmp[15] = ((utmp[11] >> 4) & kmask2) | (((utmp[7] >> 6) & kmask3) << 4);
                    utmp[14] = ((utmp[10] >> 4) & kmask2) | (((utmp[6] >> 6) & kmask3) << 4);
                    utmp[13] = ((utmp[9] >> 4) & kmask2) | (((utmp[5] >> 6) & kmask3) << 4);
                    utmp[12] = ((utmp[8] >> 4) & kmask2) | (((utmp[4] >> 6) & kmask3) << 4);
                    uaux = utmp[4] & kmask1;
                    uaux1 = utmp[5] & kmask1;
                    uaux2 = utmp[6] & kmask1;
                    uaux3 = utmp[7] & kmask1;
                    utmp[4] = (utmp[8] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
                    utmp[5] = (utmp[9] & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
                    utmp[6] = (utmp[10] & kmask2) | (((utmp[2] >> 6) & kmask3) << 4);
                    utmp[7] = (utmp[11] & kmask2) | (((utmp[3] >> 6) & kmask3) << 4);
                    utmp[8] = uaux;
                    utmp[9] = uaux1;
                    utmp[10] = uaux2;
                    utmp[11] = uaux3;
                    utmp[0] &= kmask1;
                    utmp[1] &= kmask1;
                    utmp[2] &= kmask1;
                    utmp[3] &= kmask1;

                    vint16m2_t vec_scales_64 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(__riscv_vle8_v_u8m1(scales, vl), vl));
                    vint16m2_t vec_mins_64 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(__riscv_vle8_v_u8m1(mins, vl), vl));

                    sum_block0 = __riscv_vwmacc_vv_i32m4(sum_block0, suml_32_0, vec_scales_32, vl);
                    sum_block1 = __riscv_vwmacc_vv_i32m4(sum_block1, suml_32_1, vec_scales_32, vl);
                    sum_block2 = __riscv_vwmacc_vv_i32m4(sum_block2, suml_32_2, vec_scales_32, vl);
                    sum_block3 = __riscv_vwmacc_vv_i32m4(sum_block3, suml_32_3, vec_scales_32, vl);
                    sum_block0 = __riscv_vwmacc_vv_i32m4(sum_block0, suml_64_0, vec_scales_64, vl);
                    sum_block1 = __riscv_vwmacc_vv_i32m4(sum_block1, suml_64_1, vec_scales_64, vl);
                    sum_block2 = __riscv_vwmacc_vv_i32m4(sum_block2, suml_64_2, vec_scales_64, vl);
                    sum_block3 = __riscv_vwmacc_vv_i32m4(sum_block3, suml_64_3, vec_scales_64, vl);

                    // mins* qsums 0
                    sum_min_block0 = __riscv_vadd_vv_i32m4(
                        sum_min_block0,
                        __riscv_vadd_vv_i32m4(__riscv_vwmul_vx_i32m4(vec_mins_32, a_ptr[b].bsums[0 + sb * 8], vl), __riscv_vwmul_vx_i32m4(vec_mins_64, a_ptr[b].bsums[4 + sb * 8], vl), vl), vl);
                    sum_min_block1 = __riscv_vadd_vv_i32m4(
                        sum_min_block1,
                        __riscv_vadd_vv_i32m4(__riscv_vwmul_vx_i32m4(vec_mins_32, a_ptr[b].bsums[1 + sb * 8], vl), __riscv_vwmul_vx_i32m4(vec_mins_64, a_ptr[b].bsums[5 + sb * 8], vl), vl), vl);
                    sum_min_block2 = __riscv_vadd_vv_i32m4(
                        sum_min_block2,
                        __riscv_vadd_vv_i32m4(__riscv_vwmul_vx_i32m4(vec_mins_32, a_ptr[b].bsums[2 + sb * 8], vl), __riscv_vwmul_vx_i32m4(vec_mins_64, a_ptr[b].bsums[6 + sb * 8], vl), vl), vl);
                    sum_min_block3 = __riscv_vadd_vv_i32m4(
                        sum_min_block3,
                        __riscv_vadd_vv_i32m4(__riscv_vwmul_vx_i32m4(vec_mins_32, a_ptr[b].bsums[3 + sb * 8], vl), __riscv_vwmul_vx_i32m4(vec_mins_64, a_ptr[b].bsums[7 + sb * 8], vl), vl), vl);
                }

                float src1_d0 = a_ptr[b].d[0];
                float src1_d1 = a_ptr[b].d[1];
                float src1_d2 = a_ptr[b].d[2];
                float src1_d3 = a_ptr[b].d[3];

                vfloat32m4_t src0_d = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vreinterpret_v_u16m2_f16m2(__riscv_vle16_v_u16m2(b_ptr[b].d, vl)), vl);
                vfloat32m4_t src0_dmin = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vreinterpret_v_u16m2_f16m2(__riscv_vle16_v_u16m2(b_ptr[b].dmin, vl)), vl);

                sum_rows0 = __riscv_vfmacc_vv_f32m4(sum_rows0, __riscv_vfcvt_f_x_v_f32m4(sum_block0, vl), __riscv_vfmul_vf_f32m4(src0_d, src1_d0, vl), vl);
                sum_min_rows0 = __riscv_vfmacc_vv_f32m4(sum_min_rows0, __riscv_vfcvt_f_x_v_f32m4(sum_min_block0, vl), __riscv_vfmul_vf_f32m4(src0_dmin, src1_d0, vl), vl);
                sum_rows1 = __riscv_vfmacc_vv_f32m4(sum_rows1, __riscv_vfcvt_f_x_v_f32m4(sum_block1, vl), __riscv_vfmul_vf_f32m4(src0_d, src1_d1, vl), vl);
                sum_min_rows1 = __riscv_vfmacc_vv_f32m4(sum_min_rows1, __riscv_vfcvt_f_x_v_f32m4(sum_min_block1, vl), __riscv_vfmul_vf_f32m4(src0_dmin, src1_d1, vl), vl);
                sum_rows2 = __riscv_vfmacc_vv_f32m4(sum_rows2, __riscv_vfcvt_f_x_v_f32m4(sum_block2, vl), __riscv_vfmul_vf_f32m4(src0_d, src1_d2, vl), vl);
                sum_min_rows2 = __riscv_vfmacc_vv_f32m4(sum_min_rows2, __riscv_vfcvt_f_x_v_f32m4(sum_min_block2, vl), __riscv_vfmul_vf_f32m4(src0_dmin, src1_d2, vl), vl);
                sum_rows3 = __riscv_vfmacc_vv_f32m4(sum_rows3, __riscv_vfcvt_f_x_v_f32m4(sum_block3, vl), __riscv_vfmul_vf_f32m4(src0_d, src1_d3, vl), vl);
                sum_min_rows3 = __riscv_vfmacc_vv_f32m4(sum_min_rows3, __riscv_vfcvt_f_x_v_f32m4(sum_min_block3, vl), __riscv_vfmul_vf_f32m4(src0_dmin, src1_d3, vl), vl);
            }
            __riscv_vse32_v_f32m4((float *)(s + ((y * 4 + 0) * bs + x * 32)), __riscv_vfsub_vv_f32m4(sum_rows0, sum_min_rows0, vl), vl);
            __riscv_vse32_v_f32m4((float *)(s + ((y * 4 + 1) * bs + x * 32)), __riscv_vfsub_vv_f32m4(sum_rows1, sum_min_rows1, vl), vl);
            __riscv_vse32_v_f32m4((float *)(s + ((y * 4 + 2) * bs + x * 32)), __riscv_vfsub_vv_f32m4(sum_rows2, sum_min_rows2, vl), vl);
            __riscv_vse32_v_f32m4((float *)(s + ((y * 4 + 3) * bs + x * 32)), __riscv_vfsub_vv_f32m4(sum_rows3, sum_min_rows3, vl), vl);
        }
    }
}