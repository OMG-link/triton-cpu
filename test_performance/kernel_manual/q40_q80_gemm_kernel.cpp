#include "ggml_def.h"
#include <cstdio>
#include <algorithm>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

//基于gemm1和gemm4修改， 8 * 32的内核,正确的版本
//边界处理使用4*32计算，好处是核心计算可以展开很大，缺点是中间结果需要占用数组，增加向量store和load操作。
// n : k
// m: m
// n: n
void ggml_gemm_q4_0_8x32_q8_0(int k, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx,
                              const void *GGML_RESTRICT vy, int m, int n) {
    const int nb = k / QK4_0;
    const int ncols_interleaved = 32;

    assert(k % QK4_0 == 0);
    assert(m % 8 == 0);
    assert(n % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(n);
    UNUSED(n);
    UNUSED(nb);
    // UNUSED(ncols_interleaved);

    // int64_t start_time = ggml_time_us();

    // riscv，一次处理输入矩阵的4行*参数矩阵的8行
    const block_q4_0x32 *b_ptr_start = (const block_q4_0x32 *)vx;
    const block_q8_0x8 *a_ptr_start = (const block_q8_0x8 *)vy;
    // size_t vl = 32;   //vl = VLEN/8

    // int vl = __riscv_vsetvl_e32m1(__riscv_vlenb() / sizeof(int8_t));
    int vl = __riscv_vlenb(); // 字节数=VLEN/8

    const int mrr = 8;
    const int nrr = 32;
    int anr = m - m % mrr; // m 整数倍
    const int mnr = mrr * nrr;

    for (int y = 0; y < anr / mrr; y++) { // M 

        const block_q8_0x8 *a_ptr = a_ptr_start + (y * nb);
        
        for (int x = 0; x < n / nrr; x++) { // N

            const block_q4_0x32 *b_ptr = b_ptr_start + (x * nb);

            float sum_row[mnr] = {0.0};

            // 行累加和
            for (int64_t b = 0; b < nb; b++) { // K

                vuint8m1_t src0_0, src0_0_prefetch;
                vint16m2_t suml_32_0, suml_32_1, suml_32_2, suml_32_3, suml_32_4, suml_32_5;
                vint16m2_t suml_32_6, suml_32_7;
                // ✅ 外积布局核心 8（n 维度） * 32（k 维度）的权重 8 个子块连续 
                // ✅ 每个子块 32 个 4-bit 量化值，打包存储在 16 字节中 
                // ✅ 超块按照 QK_K/QK_SB_K = 8 * 32 在 k 维度横向堆叠 

                // 内核贡献 1.5  
                {    const int z = 0;
                    // b 权重，qs 存储的是 4-bit 量化值，两个值打包在一个字节中 
                    // 这里需要将 4-bit 量化值拆分出来 
                    // m 个 subblock 大小，32 * 16B (32 * 4 bit)= 512 字节 
                    src0_0 = __riscv_vle8_v_u8m1(b_ptr[b].qs + z * nrr, vl);
                    src0_0_prefetch = __riscv_vle8_v_u8m1(b_ptr[b].qs + (z + 1) * nrr, vl);
                    vint8m1_t low4bits_0 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(src0_0, 0x0F, vl));
                    suml_32_0 = __riscv_vwmul_vx_i16m2(low4bits_0, a_ptr[b].qs[z * 16 + 0], vl);
                    suml_32_1 = __riscv_vwmul_vx_i16m2(low4bits_0, a_ptr[b].qs[z * 16 + 1], vl);
                    suml_32_2 = __riscv_vwmul_vx_i16m2(low4bits_0, a_ptr[b].qs[z * 16 + 2], vl);
                    suml_32_3 = __riscv_vwmul_vx_i16m2(low4bits_0, a_ptr[b].qs[z * 16 + 3], vl);
                    suml_32_4 = __riscv_vwmul_vx_i16m2(low4bits_0, a_ptr[b].qs[z * 16 + 4], vl);
                    suml_32_5 = __riscv_vwmul_vx_i16m2(low4bits_0, a_ptr[b].qs[z * 16 + 5], vl);
                    suml_32_6 = __riscv_vwmul_vx_i16m2(low4bits_0, a_ptr[b].qs[z * 16 + 6], vl);
                    suml_32_7 = __riscv_vwmul_vx_i16m2(low4bits_0, a_ptr[b].qs[z * 16 + 7], vl);
                    
                    low4bits_0 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(src0_0, 4, vl));
                    suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[b].qs[z * 16 + 8], low4bits_0, vl);
                    suml_32_1 = __riscv_vwmacc_vx_i16m2(suml_32_1, a_ptr[b].qs[z * 16 + 9], low4bits_0, vl);
                    suml_32_2 = __riscv_vwmacc_vx_i16m2(suml_32_2, a_ptr[b].qs[z * 16 + 10], low4bits_0, vl);
                    suml_32_3 = __riscv_vwmacc_vx_i16m2(suml_32_3, a_ptr[b].qs[z * 16 + 11], low4bits_0, vl);
                    suml_32_4 = __riscv_vwmacc_vx_i16m2(suml_32_4, a_ptr[b].qs[z * 16 + 12], low4bits_0, vl);
                    suml_32_5 = __riscv_vwmacc_vx_i16m2(suml_32_5, a_ptr[b].qs[z * 16 + 13], low4bits_0, vl);
                    suml_32_6 = __riscv_vwmacc_vx_i16m2(suml_32_6, a_ptr[b].qs[z * 16 + 14], low4bits_0, vl);
                    suml_32_7 = __riscv_vwmacc_vx_i16m2(suml_32_7, a_ptr[b].qs[z * 16 + 15], low4bits_0, vl);
                }
                for (int z = 1; z <= 14; z++) {
                    src0_0 = src0_0_prefetch;
                    src0_0_prefetch = __riscv_vle8_v_u8m1(b_ptr[b].qs + (z + 1) * nrr, vl);
                    vint8m1_t low4bits_0 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(src0_0, 0x0F, vl));
                    suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[b].qs[z * 16 + 0], low4bits_0, vl);
                    suml_32_1 = __riscv_vwmacc_vx_i16m2(suml_32_1, a_ptr[b].qs[z * 16 + 1], low4bits_0, vl);
                    suml_32_2 = __riscv_vwmacc_vx_i16m2(suml_32_2, a_ptr[b].qs[z * 16 + 2], low4bits_0, vl);
                    suml_32_3 = __riscv_vwmacc_vx_i16m2(suml_32_3, a_ptr[b].qs[z * 16 + 3], low4bits_0, vl);
                    suml_32_4 = __riscv_vwmacc_vx_i16m2(suml_32_4, a_ptr[b].qs[z * 16 + 4], low4bits_0, vl);
                    suml_32_5 = __riscv_vwmacc_vx_i16m2(suml_32_5, a_ptr[b].qs[z * 16 + 5], low4bits_0, vl);
                    suml_32_6 = __riscv_vwmacc_vx_i16m2(suml_32_6, a_ptr[b].qs[z * 16 + 6], low4bits_0, vl);
                    suml_32_7 = __riscv_vwmacc_vx_i16m2(suml_32_7, a_ptr[b].qs[z * 16 + 7], low4bits_0, vl);
                    low4bits_0 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(src0_0, 4, vl));
                    suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[b].qs[z * 16 + 8], low4bits_0, vl);
                    suml_32_1 = __riscv_vwmacc_vx_i16m2(suml_32_1, a_ptr[b].qs[z * 16 + 9], low4bits_0, vl);
                    suml_32_2 = __riscv_vwmacc_vx_i16m2(suml_32_2, a_ptr[b].qs[z * 16 + 10], low4bits_0, vl);
                    suml_32_3 = __riscv_vwmacc_vx_i16m2(suml_32_3, a_ptr[b].qs[z * 16 + 11], low4bits_0, vl);
                    suml_32_4 = __riscv_vwmacc_vx_i16m2(suml_32_4, a_ptr[b].qs[z * 16 + 12], low4bits_0, vl);
                    suml_32_5 = __riscv_vwmacc_vx_i16m2(suml_32_5, a_ptr[b].qs[z * 16 + 13], low4bits_0, vl);
                    suml_32_6 = __riscv_vwmacc_vx_i16m2(suml_32_6, a_ptr[b].qs[z * 16 + 14], low4bits_0, vl);
                    suml_32_7 = __riscv_vwmacc_vx_i16m2(suml_32_7, a_ptr[b].qs[z * 16 + 15], low4bits_0, vl);
                }
                {     const int z = 15;
                    src0_0 = src0_0_prefetch;
                    vint8m1_t low4bits_0 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(src0_0, 0x0F, vl));
                    suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[b].qs[z * 16 + 0], low4bits_0, vl);
                    suml_32_1 = __riscv_vwmacc_vx_i16m2(suml_32_1, a_ptr[b].qs[z * 16 + 1], low4bits_0, vl);
                    suml_32_2 = __riscv_vwmacc_vx_i16m2(suml_32_2, a_ptr[b].qs[z * 16 + 2], low4bits_0, vl);
                    suml_32_3 = __riscv_vwmacc_vx_i16m2(suml_32_3, a_ptr[b].qs[z * 16 + 3], low4bits_0, vl);
                    suml_32_4 = __riscv_vwmacc_vx_i16m2(suml_32_4, a_ptr[b].qs[z * 16 + 4], low4bits_0, vl);
                    suml_32_5 = __riscv_vwmacc_vx_i16m2(suml_32_5, a_ptr[b].qs[z * 16 + 5], low4bits_0, vl);
                    suml_32_6 = __riscv_vwmacc_vx_i16m2(suml_32_6, a_ptr[b].qs[z * 16 + 6], low4bits_0, vl);
                    suml_32_7 = __riscv_vwmacc_vx_i16m2(suml_32_7, a_ptr[b].qs[z * 16 + 7], low4bits_0, vl);
                    low4bits_0 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(src0_0, 4, vl));
                    suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[b].qs[z * 16 + 8], low4bits_0, vl);
                    suml_32_1 = __riscv_vwmacc_vx_i16m2(suml_32_1, a_ptr[b].qs[z * 16 + 9], low4bits_0, vl);
                    suml_32_2 = __riscv_vwmacc_vx_i16m2(suml_32_2, a_ptr[b].qs[z * 16 + 10], low4bits_0, vl);
                    suml_32_3 = __riscv_vwmacc_vx_i16m2(suml_32_3, a_ptr[b].qs[z * 16 + 11], low4bits_0, vl);
                    suml_32_4 = __riscv_vwmacc_vx_i16m2(suml_32_4, a_ptr[b].qs[z * 16 + 12], low4bits_0, vl);
                    suml_32_5 = __riscv_vwmacc_vx_i16m2(suml_32_5, a_ptr[b].qs[z * 16 + 13], low4bits_0, vl);
                    suml_32_6 = __riscv_vwmacc_vx_i16m2(suml_32_6, a_ptr[b].qs[z * 16 + 14], low4bits_0, vl);
                    suml_32_7 = __riscv_vwmacc_vx_i16m2(suml_32_7, a_ptr[b].qs[z * 16 + 15], low4bits_0, vl);
                }

                // printf("m-y: %d, n-x: %d, k-b: %d\n", x, y, b);
                vfloat32m4_t vec_scales_32 = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vreinterpret_v_u16m2_f16m2(__riscv_vle16_v_u16m2(b_ptr[b].d, vl)), vl);
            
                //先将sum结果与q4的d相乘，再与q8的d相乘
                __riscv_vse32_v_f32m4(sum_row,
                    __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row, vl), 
                        __riscv_vfwcvt_f_x_v_f32m4(suml_32_0, vl), 
                        __riscv_vfmul_vf_f32m4(vec_scales_32, GGML_FP16_TO_FP32(a_ptr[b].d[0]), vl), 
                    vl),
                vl);
                __riscv_vse32_v_f32m4(sum_row + nrr,
                    __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nrr, vl),
                    __riscv_vfwcvt_f_x_v_f32m4(suml_32_1, vl), 
                        __riscv_vfmul_vf_f32m4(vec_scales_32, GGML_FP16_TO_FP32(a_ptr[b].d[1]), vl), 
                    vl),
                vl);
                __riscv_vse32_v_f32m4(sum_row + nrr * 2,
                    __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nrr * 2, vl),
                    __riscv_vfwcvt_f_x_v_f32m4(suml_32_2, vl), 
                        __riscv_vfmul_vf_f32m4(vec_scales_32, GGML_FP16_TO_FP32(a_ptr[b].d[2]), vl), 
                    vl),                   
                vl);
                __riscv_vse32_v_f32m4(sum_row + nrr * 3,
                    __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nrr * 3, vl),
                    __riscv_vfwcvt_f_x_v_f32m4(suml_32_3, vl), 
                        __riscv_vfmul_vf_f32m4(vec_scales_32, GGML_FP16_TO_FP32(a_ptr[b].d[3]), vl), 
                    vl),                 
                vl);
                __riscv_vse32_v_f32m4(sum_row + nrr * 4,
                    __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nrr * 4, vl),
                    __riscv_vfwcvt_f_x_v_f32m4(suml_32_4, vl), 
                        __riscv_vfmul_vf_f32m4(vec_scales_32, GGML_FP16_TO_FP32(a_ptr[b].d[4]), vl), 
                    vl),                   
                 vl);
                __riscv_vse32_v_f32m4(sum_row + nrr * 5,
                    __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nrr * 5, vl),
                    __riscv_vfwcvt_f_x_v_f32m4(suml_32_5, vl), 
                        __riscv_vfmul_vf_f32m4(vec_scales_32, GGML_FP16_TO_FP32(a_ptr[b].d[5]), vl), 
                    vl),                 
                 vl);
                __riscv_vse32_v_f32m4(sum_row + nrr * 6,
                    __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nrr * 6, vl),
                    __riscv_vfwcvt_f_x_v_f32m4(suml_32_6, vl), 
                        __riscv_vfmul_vf_f32m4(vec_scales_32, GGML_FP16_TO_FP32(a_ptr[b].d[6]), vl), 
                    vl),                  
                vl);
                __riscv_vse32_v_f32m4(sum_row + nrr * 7,
                    __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nrr * 7, vl),
                    __riscv_vfwcvt_f_x_v_f32m4(suml_32_7, vl), 
                        __riscv_vfmul_vf_f32m4(vec_scales_32, GGML_FP16_TO_FP32(a_ptr[b].d[7]), vl), 
                    vl),                  
                vl);
            }
                
            __riscv_vse32_v_f32m4((float *)(s + (y * 8 * bs + x * nrr)), __riscv_vle32_v_f32m4(sum_row, vl), vl);
            __riscv_vse32_v_f32m4((float *)(s + ((y * 8 + 1) * bs + x * nrr)), __riscv_vle32_v_f32m4(sum_row + nrr, vl), vl);
            __riscv_vse32_v_f32m4((float *)(s + ((y * 8 + 2) * bs + x * nrr)), __riscv_vle32_v_f32m4(sum_row + nrr * 2, vl), vl);
            __riscv_vse32_v_f32m4((float *)(s + ((y * 8 + 3) * bs + x * nrr)), __riscv_vle32_v_f32m4(sum_row + nrr * 3, vl), vl);
            __riscv_vse32_v_f32m4((float *)(s + ((y * 8 + 4) * bs + x * nrr)), __riscv_vle32_v_f32m4(sum_row + nrr * 4, vl), vl);
            __riscv_vse32_v_f32m4((float *)(s + ((y * 8 + 5) * bs + x * nrr)), __riscv_vle32_v_f32m4(sum_row + nrr * 5, vl), vl);
            __riscv_vse32_v_f32m4((float *)(s + ((y * 8 + 6) * bs + x * nrr)), __riscv_vle32_v_f32m4(sum_row + nrr * 6, vl), vl);
            __riscv_vse32_v_f32m4((float *)(s + ((y * 8 + 7) * bs + x * nrr)), __riscv_vle32_v_f32m4(sum_row + nrr * 7, vl), vl);
        }
    }

}