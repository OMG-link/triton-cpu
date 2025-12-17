#include "ggml_def.h"
#include <cstdio>
#include <algorithm>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

//基于gemm1和gemm4修改， 12 * 32的内核,正确的版本
//边界处理使用4*32计算，好处是核心计算可以展开很大，缺点是中间结果需要占用数组，增加向量store和load操作。
// n : k
// m: m
// n: n
void ggml_gemm_q4_0_12x32_q8_0(int k, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx,
                              const void *GGML_RESTRICT vy, int m, int n) {
    const int nb = k / QK4_0;
    const int ncols_interleaved = 32;
    const int blocklen = 12;

    // const int mr = 12;
    // const int nr = 32;

    printf("m = %d, n = %d, k = %d\n", m, n, k);


    assert(k % QK4_0 == 0);
    assert(m % 12 == 0);
    assert(n % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(n);
    UNUSED(n);
    UNUSED(nb);
    // UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    // int64_t start_time = ggml_time_us();

    // riscv，一次处理输入矩阵的4行*参数矩阵的8行
    const block_q4_0x32 *b_ptr_start = (const block_q4_0x32 *)vx;
    const block_q8_0x12 *a_ptr_start = (const block_q8_0x12 *)vy;
    // size_t vl = 32;   //vl = VLEN/8

    // int vl = __riscv_vsetvl_e32m1(__riscv_vlenb() / sizeof(int8_t));
    int vl = __riscv_vlenb(); // 字节数=VLEN/8

    const int mrr = 12;
    const int nrr = 32;
    int anr = m - m % mrr; // m 整数倍
    const int mnr = mrr * nrr;

    for (int y = 0; y < anr / mrr; y++) { // M 

        const block_q8_0x12 *a_ptr = a_ptr_start + (y * nb);
        
        for (int x = 0; x < n / nrr; x++) { // N

            const block_q4_0x32 *b_ptr = b_ptr_start + (x * nb);

            float sum_row[mnr] = {0.0};

            // 行累加和
            for (int64_t b = 0; b < nb; b++) { // K

                vuint8m1_t src0_0, src0_0_prefetch;
                vint16m2_t suml_32_0, suml_32_1, suml_32_2, suml_32_3, suml_32_4, suml_32_5;
                vint16m2_t suml_32_6, suml_32_7, suml_32_8, suml_32_9, suml_32_10, suml_32_11;
                // ✅ 外积布局核心 12（n 维度） * 32（k 维度）的权重 12 个子块连续 
                // ✅ 每个子块 32 个 4-bit 量化值，打包存储在 16 字节中 
                // ✅ 超块按照 QK_K/QK_SB_K = 8 * 32 在 k 维度横向堆叠 
                const uint64_t *a_ptr_temp = static_cast<const uint64_t *>((void *)&a_ptr[b].qs[0]); // mr * QK_SB_K = 12 * 32 = 384  

                // 内核贡献 1.5  
                {    const int z = 0;
                    // b 权重，qs 存储的是 4-bit 量化值，两个值打包在一个字节中 
                    // 这里需要将 4-bit 量化值拆分出来 
                    // m 个 subblock 大小，32 * 16B (32 * 4 bit)= 512 字节 
                    src0_0 = __riscv_vle8_v_u8m1(b_ptr[b].qs + z * nrr, vl);
                    src0_0_prefetch = __riscv_vle8_v_u8m1(b_ptr[b].qs + (z + 1) * nrr, vl);
                    vint8m1_t low4bits_0 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(src0_0, 0x0F, vl));
                    uint64_t a_0_7 = *a_ptr_temp++;
                    suml_32_0 = __riscv_vwmul_vx_i16m2(low4bits_0, (a_0_7 >> (0 * 8)) & 0xff, vl); // mul 指令是在干嘛
                    suml_32_1 = __riscv_vwmul_vx_i16m2(low4bits_0, (a_0_7 >> (1 * 8)) & 0xff, vl); // 
                    suml_32_2 = __riscv_vwmul_vx_i16m2(low4bits_0, (a_0_7 >> (2 * 8)) & 0xff, vl);
                    suml_32_3 = __riscv_vwmul_vx_i16m2(low4bits_0, (a_0_7 >> (3 * 8)) & 0xff, vl);
                    suml_32_4 = __riscv_vwmul_vx_i16m2(low4bits_0, (a_0_7 >> (4 * 8)) & 0xff, vl);
                    suml_32_5 = __riscv_vwmul_vx_i16m2(low4bits_0, (a_0_7 >> (5 * 8)) & 0xff, vl);
                    suml_32_6 = __riscv_vwmul_vx_i16m2(low4bits_0, (a_0_7 >> (6 * 8)) & 0xff, vl);
                    suml_32_7 = __riscv_vwmul_vx_i16m2(low4bits_0, (a_0_7 >> (7 * 8)) & 0xff, vl);
                    
                    uint64_t a_8_15 = *a_ptr_temp++;
                    suml_32_8 = __riscv_vwmul_vx_i16m2(low4bits_0, (a_8_15 >> (0 * 8)) & 0xff, vl);
                    suml_32_9 = __riscv_vwmul_vx_i16m2(low4bits_0, (a_8_15 >> (1 * 8)) & 0xff, vl);
                    suml_32_10 = __riscv_vwmul_vx_i16m2(low4bits_0, (a_8_15 >> (2 * 8)) & 0xff, vl);
                    suml_32_11 = __riscv_vwmul_vx_i16m2(low4bits_0, (a_8_15 >> (3 * 8)) & 0xff, vl);
                    
                    low4bits_0 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(src0_0, 4, vl));
                    suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, (a_8_15 >> (4 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_1 = __riscv_vwmacc_vx_i16m2(suml_32_1, (a_8_15 >> (5 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_2 = __riscv_vwmacc_vx_i16m2(suml_32_2, (a_8_15 >> (6 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_3 = __riscv_vwmacc_vx_i16m2(suml_32_3, (a_8_15 >> (7 * 8)) & 0xff, low4bits_0, vl);
                    uint64_t a_16_23 = *a_ptr_temp++;
                    suml_32_4 = __riscv_vwmacc_vx_i16m2(suml_32_4, (a_16_23 >> (0 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_5 = __riscv_vwmacc_vx_i16m2(suml_32_5, (a_16_23 >> (1 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_6 = __riscv_vwmacc_vx_i16m2(suml_32_6, (a_16_23 >> (2 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_7 = __riscv_vwmacc_vx_i16m2(suml_32_7, (a_16_23 >> (3 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_8 = __riscv_vwmacc_vx_i16m2(suml_32_8, (a_16_23 >> (4 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_9 = __riscv_vwmacc_vx_i16m2(suml_32_9, (a_16_23 >> (5 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_10 = __riscv_vwmacc_vx_i16m2(suml_32_10, (a_16_23 >> (6 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_11 = __riscv_vwmacc_vx_i16m2(suml_32_11, (a_16_23 >> (7 * 8)) & 0xff, low4bits_0, vl);
                }
                for (int z = 1; z <= 14; z++) {
                    src0_0 = src0_0_prefetch;
                    src0_0_prefetch = __riscv_vle8_v_u8m1(b_ptr[b].qs + (z + 1) * nrr, vl);
                    vint8m1_t low4bits_0 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(src0_0, 0x0F, vl));
                    uint64_t a_0_7 = *a_ptr_temp++;
                    suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, (a_0_7 >> (0 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_1 = __riscv_vwmacc_vx_i16m2(suml_32_1, (a_0_7 >> (1 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_2 = __riscv_vwmacc_vx_i16m2(suml_32_2, (a_0_7 >> (2 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_3 = __riscv_vwmacc_vx_i16m2(suml_32_3, (a_0_7 >> (3 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_4 = __riscv_vwmacc_vx_i16m2(suml_32_4, (a_0_7 >> (4 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_5 = __riscv_vwmacc_vx_i16m2(suml_32_5, (a_0_7 >> (5 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_6 = __riscv_vwmacc_vx_i16m2(suml_32_6, (a_0_7 >> (6 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_7 = __riscv_vwmacc_vx_i16m2(suml_32_7, (a_0_7 >> (7 * 8)) & 0xff, low4bits_0, vl);
                    uint64_t a_8_15 = *a_ptr_temp++;
                    suml_32_8 = __riscv_vwmacc_vx_i16m2(suml_32_8, (a_8_15 >> (0 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_9 = __riscv_vwmacc_vx_i16m2(suml_32_9, (a_8_15 >> (1 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_10 = __riscv_vwmacc_vx_i16m2(suml_32_10, (a_8_15 >> (2 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_11 = __riscv_vwmacc_vx_i16m2(suml_32_11, (a_8_15 >> (3 * 8)) & 0xff, low4bits_0, vl);
                    low4bits_0 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(src0_0, 4, vl));
                    suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, (a_8_15 >> (4 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_1 = __riscv_vwmacc_vx_i16m2(suml_32_1, (a_8_15 >> (5 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_2 = __riscv_vwmacc_vx_i16m2(suml_32_2, (a_8_15 >> (6 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_3 = __riscv_vwmacc_vx_i16m2(suml_32_3, (a_8_15 >> (7 * 8)) & 0xff, low4bits_0, vl);
                    uint64_t a_16_23 = *a_ptr_temp++;
                    suml_32_4 = __riscv_vwmacc_vx_i16m2(suml_32_4, (a_16_23 >> (0 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_5 = __riscv_vwmacc_vx_i16m2(suml_32_5, (a_16_23 >> (1 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_6 = __riscv_vwmacc_vx_i16m2(suml_32_6, (a_16_23 >> (2 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_7 = __riscv_vwmacc_vx_i16m2(suml_32_7, (a_16_23 >> (3 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_8 = __riscv_vwmacc_vx_i16m2(suml_32_8, (a_16_23 >> (4 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_9 = __riscv_vwmacc_vx_i16m2(suml_32_9, (a_16_23 >> (5 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_10 = __riscv_vwmacc_vx_i16m2(suml_32_10, (a_16_23 >> (6 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_11 = __riscv_vwmacc_vx_i16m2(suml_32_11, (a_16_23 >> (7 * 8)) & 0xff, low4bits_0, vl);
                }
                { // z = 15 
                    src0_0 = src0_0_prefetch;
                    vint8m1_t low4bits_0 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(src0_0, 0x0F, vl));
                    uint64_t a_0_7 = *a_ptr_temp++;
                    suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, (a_0_7 >> (0 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_1 = __riscv_vwmacc_vx_i16m2(suml_32_1, (a_0_7 >> (1 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_2 = __riscv_vwmacc_vx_i16m2(suml_32_2, (a_0_7 >> (2 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_3 = __riscv_vwmacc_vx_i16m2(suml_32_3, (a_0_7 >> (3 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_4 = __riscv_vwmacc_vx_i16m2(suml_32_4, (a_0_7 >> (4 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_5 = __riscv_vwmacc_vx_i16m2(suml_32_5, (a_0_7 >> (5 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_6 = __riscv_vwmacc_vx_i16m2(suml_32_6, (a_0_7 >> (6 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_7 = __riscv_vwmacc_vx_i16m2(suml_32_7, (a_0_7 >> (7 * 8)) & 0xff, low4bits_0, vl);
                    uint64_t a_8_15 = *a_ptr_temp++;
                    suml_32_8 = __riscv_vwmacc_vx_i16m2(suml_32_8, (a_8_15 >> (0 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_9 = __riscv_vwmacc_vx_i16m2(suml_32_9, (a_8_15 >> (1 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_10 = __riscv_vwmacc_vx_i16m2(suml_32_10, (a_8_15 >> (2 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_11 = __riscv_vwmacc_vx_i16m2(suml_32_11, (a_8_15 >> (3 * 8)) & 0xff, low4bits_0, vl);

                    
                    low4bits_0 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(src0_0, 4, vl));
                    suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, (a_8_15 >> (4 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_1 = __riscv_vwmacc_vx_i16m2(suml_32_1, (a_8_15 >> (5 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_2 = __riscv_vwmacc_vx_i16m2(suml_32_2, (a_8_15 >> (6 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_3 = __riscv_vwmacc_vx_i16m2(suml_32_3, (a_8_15 >> (7 * 8)) & 0xff, low4bits_0, vl);
                    uint64_t a_16_23 = *a_ptr_temp++;
                    suml_32_4 = __riscv_vwmacc_vx_i16m2(suml_32_4, (a_16_23 >> (0 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_5 = __riscv_vwmacc_vx_i16m2(suml_32_5, (a_16_23 >> (1 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_6 = __riscv_vwmacc_vx_i16m2(suml_32_6, (a_16_23 >> (2 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_7 = __riscv_vwmacc_vx_i16m2(suml_32_7, (a_16_23 >> (3 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_8 = __riscv_vwmacc_vx_i16m2(suml_32_8, (a_16_23 >> (4 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_9 = __riscv_vwmacc_vx_i16m2(suml_32_9, (a_16_23 >> (5 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_10 = __riscv_vwmacc_vx_i16m2(suml_32_10, (a_16_23 >> (6 * 8)) & 0xff, low4bits_0, vl);
                    suml_32_11 = __riscv_vwmacc_vx_i16m2(suml_32_11, (a_16_23 >> (7 * 8)) & 0xff, low4bits_0, vl);
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
                __riscv_vse32_v_f32m4(sum_row + nrr * 8,
                    __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nrr * 8, vl),
                    __riscv_vfwcvt_f_x_v_f32m4(suml_32_8, vl), 
                        __riscv_vfmul_vf_f32m4(vec_scales_32, GGML_FP16_TO_FP32(a_ptr[b].d[8]), vl), 
                    vl),                 
                vl);
                __riscv_vse32_v_f32m4(sum_row + nrr * 9,
                    __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nrr * 9, vl),
                    __riscv_vfwcvt_f_x_v_f32m4(suml_32_9, vl), 
                        __riscv_vfmul_vf_f32m4(vec_scales_32, GGML_FP16_TO_FP32(a_ptr[b].d[9]), vl), 
                    vl),                  
                vl);
                __riscv_vse32_v_f32m4(sum_row + nrr * 10,
                    __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nrr * 10, vl),
                    __riscv_vfwcvt_f_x_v_f32m4(suml_32_10, vl), 
                        __riscv_vfmul_vf_f32m4(vec_scales_32, GGML_FP16_TO_FP32(a_ptr[b].d[10]), vl), 
                    vl),                  
                vl);
                __riscv_vse32_v_f32m4(sum_row + nrr * 11,
                    __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nrr * 11, vl),
                    __riscv_vfwcvt_f_x_v_f32m4(suml_32_11, vl), 
                        __riscv_vfmul_vf_f32m4(vec_scales_32, GGML_FP16_TO_FP32(a_ptr[b].d[11]), vl), 
                    vl),                 
                vl);
            }
                
            __riscv_vse32_v_f32m4((float *)(s + (y * 12 * bs + x * nrr)), __riscv_vle32_v_f32m4(sum_row, vl), vl);
            __riscv_vse32_v_f32m4((float *)(s + ((y * 12 + 1) * bs + x * nrr)), __riscv_vle32_v_f32m4(sum_row + nrr, vl), vl);
            __riscv_vse32_v_f32m4((float *)(s + ((y * 12 + 2) * bs + x * nrr)), __riscv_vle32_v_f32m4(sum_row + nrr * 2, vl), vl);
            __riscv_vse32_v_f32m4((float *)(s + ((y * 12 + 3) * bs + x * nrr)), __riscv_vle32_v_f32m4(sum_row + nrr * 3, vl), vl);
            __riscv_vse32_v_f32m4((float *)(s + ((y * 12 + 4) * bs + x * nrr)), __riscv_vle32_v_f32m4(sum_row + nrr * 4, vl), vl);
            __riscv_vse32_v_f32m4((float *)(s + ((y * 12 + 5) * bs + x * nrr)), __riscv_vle32_v_f32m4(sum_row + nrr * 5, vl), vl);
            __riscv_vse32_v_f32m4((float *)(s + ((y * 12 + 6) * bs + x * nrr)), __riscv_vle32_v_f32m4(sum_row + nrr * 6, vl), vl);
            __riscv_vse32_v_f32m4((float *)(s + ((y * 12 + 7) * bs + x * nrr)), __riscv_vle32_v_f32m4(sum_row + nrr * 7, vl), vl);
            __riscv_vse32_v_f32m4((float *)(s + ((y * 12 + 8) * bs + x * nrr)), __riscv_vle32_v_f32m4(sum_row + nrr * 8, vl), vl);
            __riscv_vse32_v_f32m4((float *)(s + ((y * 12 + 9) * bs + x * nrr)), __riscv_vle32_v_f32m4(sum_row + nrr * 9, vl), vl);
            __riscv_vse32_v_f32m4((float *)(s + ((y * 12 + 10) * bs + x * nrr)), __riscv_vle32_v_f32m4(sum_row + nrr * 10, vl), vl);
            __riscv_vse32_v_f32m4((float *)(s + ((y * 12 + 11) * bs + x * nrr)), __riscv_vle32_v_f32m4(sum_row + nrr * 11, vl), vl);
        }
    }

}