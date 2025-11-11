#include "ggml_def.h"
#include <cstdio>
#include <algorithm>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif


using block_q4_Kx32 = block_q4_Kx<32>;
using block_q8_Kx12 = block_q8_Kx<12>;

void ggml_gemm_q4_K_8x32_q8_K(int k, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx, const void *GGML_RESTRICT vy, int m, int n) {
    int vl = __riscv_vlenb(); // 字节数=VLEN/8

    const int mr = 12;
    const int nr = 32;

    const int mc = mr * 4;
    const int nc = nr * 4;

    assert(vl == nr);
    assert(k % QK_K == 0);
    assert(m % mr == 0);
    assert(n % nr == 0);

    auto b_ptr_start = (const block_q4_Kx32 *)vx;
    auto a_ptr_start = (const block_q8_Kx12 *)vy;

#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (int i_nc = 0; i_nc < n; i_nc += nc) {     // Nc 
        for (int i_mc = 0; i_mc < m; i_mc += mc) { // Mc 
            assert(i_mc < m && i_mc % mr == 0);
            assert(i_nc < n && i_nc % nr == 0);

            // printf("Task i_mc=%d, i_nc=%d, thread=%d\n", i_mc, i_nc, omp_get_thread_num());

            int up_n = std::min(i_nc + nc, n);
            int up_m = std::min(i_mc + mc, m);
            for (int i_mr = i_mc; i_mr < up_m; i_mr += mr) {     // Mr 12 
                for (int i_nr = i_nc; i_nr < up_n; i_nr += nr) { // Nr 32 
                    ASSERT_MSG(i_nr < n && i_nr + nr <= n,
                               "i_nr out of bounds: i_nr=%d, n=%d, nr=%d",
                               i_nr, n, nr);
                    
                    ASSERT_MSG(i_mr < m && i_mr + mr <= m,
                               "i_mr out of bounds: i_mr=%d, m=%d, mr=%d",
                               i_mr, m, mr);

                    auto a_ptr = a_ptr_start + (i_mr * k) / (mr * QK_K); 
                    auto b_ptr = b_ptr_start + (i_nr * k) / (nr * QK_K); 
                    float sum_row[mr * nr] = {0.0};
                    for (int b = 0; b < k / QK_K; b++) { // K: superblock 
                        int sum_block[mr * nr];
                        for (int sb = 0; sb < QK_K / QK_SB_K; sb++) { // K: subblock 寄存器内核 循环展开 
                            vuint8m1_t src0_0, src0_0_prefetch;
                            vint16m2_t suml_32_0, suml_32_1, suml_32_2, suml_32_3, suml_32_4, suml_32_5;
                            vint16m2_t suml_32_6, suml_32_7, suml_32_8, suml_32_9, suml_32_10, suml_32_11;
                            const uint64_t *a_ptr_temp = static_cast<const uint64_t *>((void *)&a_ptr[b].qs[sb * 384]); // mr * QK_SB_K = 12 * 32 = 384  
                            
                            // 内核贡献 1.5 
                            {    const int z = 0;
                                // b 权重，qs 存储的是 4-bit 量化值，两个值打包在一个字节中
                                // 这里需要将 4-bit 量化值拆分出来
                                // nr 个 subblock 大小，32 * 16B (32 * 4 bit)= 512 字节
                                src0_0 = __riscv_vle8_v_u8m1(b_ptr[b].qs + sb * 512 + z * nr, vl);
                                src0_0_prefetch = __riscv_vle8_v_u8m1(b_ptr[b].qs + sb * 512 + (z + 1) * nr, vl);
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
                                src0_0_prefetch = __riscv_vle8_v_u8m1(b_ptr[b].qs + sb * 512 + (z + 1) * nr, vl);
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

                            // 累加到 sum_block 贡献 0.36
                            vint16m2_t vec_scales_32 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(__riscv_vle8_v_u8m1(b_ptr[b].scales + sb * nr, vl), vl));

                            if (sb == 0) {
                                __riscv_vse32_v_i32m4(sum_block + 0 * nr, __riscv_vwmul_vv_i32m4(suml_32_0, vec_scales_32, vl), vl);
                                __riscv_vse32_v_i32m4(sum_block + 1 * nr, __riscv_vwmul_vv_i32m4(suml_32_1, vec_scales_32, vl), vl);
                                __riscv_vse32_v_i32m4(sum_block + 2 * nr, __riscv_vwmul_vv_i32m4(suml_32_2, vec_scales_32, vl), vl);
                                __riscv_vse32_v_i32m4(sum_block + 3 * nr, __riscv_vwmul_vv_i32m4(suml_32_3, vec_scales_32, vl), vl);
                                __riscv_vse32_v_i32m4(sum_block + 4 * nr, __riscv_vwmul_vv_i32m4(suml_32_4, vec_scales_32, vl), vl);
                                __riscv_vse32_v_i32m4(sum_block + 5 * nr, __riscv_vwmul_vv_i32m4(suml_32_5, vec_scales_32, vl), vl);
                                __riscv_vse32_v_i32m4(sum_block + 6 * nr, __riscv_vwmul_vv_i32m4(suml_32_6, vec_scales_32, vl), vl);
                                __riscv_vse32_v_i32m4(sum_block + 7 * nr, __riscv_vwmul_vv_i32m4(suml_32_7, vec_scales_32, vl), vl);
                                __riscv_vse32_v_i32m4(sum_block + 8 * nr, __riscv_vwmul_vv_i32m4(suml_32_8, vec_scales_32, vl), vl);
                                __riscv_vse32_v_i32m4(sum_block + 9 * nr, __riscv_vwmul_vv_i32m4(suml_32_9, vec_scales_32, vl), vl);
                                __riscv_vse32_v_i32m4(sum_block + 10 * nr, __riscv_vwmul_vv_i32m4(suml_32_10, vec_scales_32, vl), vl);
                                __riscv_vse32_v_i32m4(sum_block + 11 * nr, __riscv_vwmul_vv_i32m4(suml_32_11, vec_scales_32, vl), vl);
                            } else {
                                __riscv_vse32_v_i32m4(sum_block + nr * 0, __riscv_vwmacc_vv_i32m4(__riscv_vle32_v_i32m4(sum_block + nr * 0, vl), suml_32_0, vec_scales_32, vl), vl);
                                __riscv_vse32_v_i32m4(sum_block + nr * 1, __riscv_vwmacc_vv_i32m4(__riscv_vle32_v_i32m4(sum_block + nr * 1, vl), suml_32_1, vec_scales_32, vl), vl);
                                __riscv_vse32_v_i32m4(sum_block + nr * 2, __riscv_vwmacc_vv_i32m4(__riscv_vle32_v_i32m4(sum_block + nr * 2, vl), suml_32_2, vec_scales_32, vl), vl);
                                __riscv_vse32_v_i32m4(sum_block + nr * 3, __riscv_vwmacc_vv_i32m4(__riscv_vle32_v_i32m4(sum_block + nr * 3, vl), suml_32_3, vec_scales_32, vl), vl);
                                __riscv_vse32_v_i32m4(sum_block + nr * 4, __riscv_vwmacc_vv_i32m4(__riscv_vle32_v_i32m4(sum_block + nr * 4, vl), suml_32_4, vec_scales_32, vl), vl);
                                __riscv_vse32_v_i32m4(sum_block + nr * 5, __riscv_vwmacc_vv_i32m4(__riscv_vle32_v_i32m4(sum_block + nr * 5, vl), suml_32_5, vec_scales_32, vl), vl);
                                __riscv_vse32_v_i32m4(sum_block + nr * 6, __riscv_vwmacc_vv_i32m4(__riscv_vle32_v_i32m4(sum_block + nr * 6, vl), suml_32_6, vec_scales_32, vl), vl);
                                __riscv_vse32_v_i32m4(sum_block + nr * 7, __riscv_vwmacc_vv_i32m4(__riscv_vle32_v_i32m4(sum_block + nr * 7, vl), suml_32_7, vec_scales_32, vl), vl);
                                __riscv_vse32_v_i32m4(sum_block + nr * 8, __riscv_vwmacc_vv_i32m4(__riscv_vle32_v_i32m4(sum_block + nr * 8, vl), suml_32_8, vec_scales_32, vl), vl);
                                __riscv_vse32_v_i32m4(sum_block + nr * 9, __riscv_vwmacc_vv_i32m4(__riscv_vle32_v_i32m4(sum_block + nr * 9, vl), suml_32_9, vec_scales_32, vl), vl);
                                __riscv_vse32_v_i32m4(sum_block + nr * 10, __riscv_vwmacc_vv_i32m4(__riscv_vle32_v_i32m4(sum_block + nr * 10, vl), suml_32_10, vec_scales_32, vl), vl);
                                __riscv_vse32_v_i32m4(sum_block + nr * 11, __riscv_vwmacc_vv_i32m4(__riscv_vle32_v_i32m4(sum_block + nr * 11, vl), suml_32_11, vec_scales_32, vl), vl);
                            }
                        }

                        // 累加到 sum_row 贡献 0.1
                        vfloat32m4_t src0_d = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vreinterpret_v_u16m2_f16m2(__riscv_vle16_v_u16m2(b_ptr[b].d, vl)), vl);

                        __riscv_vse32_v_f32m4(sum_row + nr * 0,
                                              __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row, vl), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_block, vl), vl),
                                                                      __riscv_vfmul_vf_f32m4(src0_d, a_ptr[b].d[0], vl), vl),
                                              vl);
                        __riscv_vse32_v_f32m4(sum_row + nr * 1,
                                              __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr, vl), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_block + nr, vl), vl),
                                                                      __riscv_vfmul_vf_f32m4(src0_d, a_ptr[b].d[1], vl), vl),
                                              vl);
                        __riscv_vse32_v_f32m4(sum_row + nr * 2,
                                              __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr * 2, vl), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_block + nr * 2, vl), vl),
                                                                      __riscv_vfmul_vf_f32m4(src0_d, a_ptr[b].d[2], vl), vl),
                                              vl);
                        __riscv_vse32_v_f32m4(sum_row + nr * 3,
                                              __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr * 3, vl), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_block + nr * 3, vl), vl),
                                                                      __riscv_vfmul_vf_f32m4(src0_d, a_ptr[b].d[3], vl), vl),
                                              vl);
                        __riscv_vse32_v_f32m4(sum_row + nr * 4,
                                              __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr * 4, vl), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_block + nr * 4, vl), vl),
                                                                      __riscv_vfmul_vf_f32m4(src0_d, a_ptr[b].d[4], vl), vl),
                                              vl);
                        __riscv_vse32_v_f32m4(sum_row + nr * 5,
                                              __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr * 5, vl), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_block + nr * 5, vl), vl),
                                                                      __riscv_vfmul_vf_f32m4(src0_d, a_ptr[b].d[5], vl), vl),
                                              vl);
                        __riscv_vse32_v_f32m4(sum_row + nr * 6,
                                              __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr * 6, vl), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_block + nr * 6, vl), vl),
                                                                      __riscv_vfmul_vf_f32m4(src0_d, a_ptr[b].d[6], vl), vl),
                                              vl);
                        __riscv_vse32_v_f32m4(sum_row + nr * 7,
                                              __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr * 7, vl), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_block + nr * 7, vl), vl),
                                                                      __riscv_vfmul_vf_f32m4(src0_d, a_ptr[b].d[7], vl), vl),
                                              vl);
                        __riscv_vse32_v_f32m4(sum_row + nr * 8,
                                              __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr * 8, vl), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_block + nr * 8, vl), vl),
                                                                      __riscv_vfmul_vf_f32m4(src0_d, a_ptr[b].d[8], vl), vl),
                                              vl);
                        __riscv_vse32_v_f32m4(sum_row + nr * 9,
                                              __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr * 9, vl), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_block + nr * 9, vl), vl),
                                                                      __riscv_vfmul_vf_f32m4(src0_d, a_ptr[b].d[9], vl), vl),
                                              vl);
                        __riscv_vse32_v_f32m4(sum_row + nr * 10,
                                              __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr * 10, vl), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_block + nr * 10, vl), vl),
                                                                      __riscv_vfmul_vf_f32m4(src0_d, a_ptr[b].d[10], vl), vl),
                                              vl);
                        __riscv_vse32_v_f32m4(sum_row + nr * 11,
                                              __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr * 11, vl), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_block + nr * 11, vl), vl),
                                                                      __riscv_vfmul_vf_f32m4(src0_d, a_ptr[b].d[11], vl), vl),
                                              vl);
                    }
                    // 计算 sum_min_row 贡献 0.16 
                    float sum_min_row[mr * nr] = {0.0};
                    
                    
                    for (int64_t b = 0; b < k / QK_K; b++) { // K: block(min)
                        vint16m2_t vec_mins_0 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(__riscv_vle8_v_u8m1(b_ptr[b].scales + nr * 8, vl), vl));
                        vint16m2_t vec_mins_1 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(__riscv_vle8_v_u8m1(b_ptr[b].scales + nr * 9, vl), vl));
                        vint16m2_t vec_mins_2 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(__riscv_vle8_v_u8m1(b_ptr[b].scales + nr * 10, vl), vl));
                        vint16m2_t vec_mins_3 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(__riscv_vle8_v_u8m1(b_ptr[b].scales + nr * 11, vl), vl));
                        vint16m2_t vec_mins_4 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(__riscv_vle8_v_u8m1(b_ptr[b].scales + nr * 12, vl), vl));
                        vint16m2_t vec_mins_5 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(__riscv_vle8_v_u8m1(b_ptr[b].scales + nr * 13, vl), vl));
                        vint16m2_t vec_mins_6 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(__riscv_vle8_v_u8m1(b_ptr[b].scales + nr * 14, vl), vl));
                        vint16m2_t vec_mins_7 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(__riscv_vle8_v_u8m1(b_ptr[b].scales + nr * 15, vl), vl));
                        vfloat32m4_t src0_dmin = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vreinterpret_v_u16m2_f16m2(__riscv_vle16_v_u16m2(b_ptr[b].dmin, vl)), vl);
                        for (int i = 0; i < mr; i++) {
                            vint32m4_t sum_min_block = __riscv_vwmul_vx_i32m4(vec_mins_0, a_ptr[b].bsums[i + 0 * mr], vl);
                            sum_min_block = __riscv_vwmacc_vx_i32m4(sum_min_block, a_ptr[b].bsums[i + 1 * mr], vec_mins_1, vl);
                            sum_min_block = __riscv_vwmacc_vx_i32m4(sum_min_block, a_ptr[b].bsums[i + 2 * mr], vec_mins_2, vl);
                            sum_min_block = __riscv_vwmacc_vx_i32m4(sum_min_block, a_ptr[b].bsums[i + 3 * mr], vec_mins_3, vl);
                            sum_min_block = __riscv_vwmacc_vx_i32m4(sum_min_block, a_ptr[b].bsums[i + 4 * mr], vec_mins_4, vl);
                            sum_min_block = __riscv_vwmacc_vx_i32m4(sum_min_block, a_ptr[b].bsums[i + 5 * mr], vec_mins_5, vl);
                            sum_min_block = __riscv_vwmacc_vx_i32m4(sum_min_block, a_ptr[b].bsums[i + 6 * mr], vec_mins_6, vl);
                            sum_min_block = __riscv_vwmacc_vx_i32m4(sum_min_block, a_ptr[b].bsums[i + 7 * mr], vec_mins_7, vl);
                            __riscv_vse32_v_f32m4(sum_min_row + nr * i,
                                                  __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_min_row + nr * i, vl), __riscv_vfcvt_f_x_v_f32m4(sum_min_block, vl),
                                                                          __riscv_vfmul_vf_f32m4(src0_dmin, a_ptr[b].d[i], vl), vl),
                                                  vl);
                        }
                    }
                    // 累加答案到 C 贡献 0.05
                    __riscv_vse32_v_f32m4(static_cast<float *>(s + ((i_mr + 0) * bs + (i_nr))),
                                          __riscv_vfsub_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr * 0, vl), __riscv_vle32_v_f32m4(sum_min_row + nr * 0, vl), vl), vl);
                    __riscv_vse32_v_f32m4(static_cast<float *>(s + ((i_mr + 1) * bs + (i_nr))),
                                          __riscv_vfsub_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr * 1, vl), __riscv_vle32_v_f32m4(sum_min_row + nr * 1, vl), vl), vl);
                    __riscv_vse32_v_f32m4(static_cast<float *>(s + ((i_mr + 2) * bs + (i_nr))),
                                          __riscv_vfsub_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr * 2, vl), __riscv_vle32_v_f32m4(sum_min_row + nr * 2, vl), vl), vl);
                    __riscv_vse32_v_f32m4(static_cast<float *>(s + ((i_mr + 3) * bs + (i_nr))),
                                          __riscv_vfsub_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr * 3, vl), __riscv_vle32_v_f32m4(sum_min_row + nr * 3, vl), vl), vl);
                    __riscv_vse32_v_f32m4(static_cast<float *>(s + ((i_mr + 4) * bs + (i_nr))),
                                          __riscv_vfsub_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr * 4, vl), __riscv_vle32_v_f32m4(sum_min_row + nr * 4, vl), vl), vl);
                    __riscv_vse32_v_f32m4(static_cast<float *>(s + ((i_mr + 5) * bs + (i_nr))),
                                          __riscv_vfsub_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr * 5, vl), __riscv_vle32_v_f32m4(sum_min_row + nr * 5, vl), vl), vl);
                    __riscv_vse32_v_f32m4(static_cast<float *>(s + ((i_mr + 6) * bs + (i_nr))),
                                          __riscv_vfsub_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr * 6, vl), __riscv_vle32_v_f32m4(sum_min_row + nr * 6, vl), vl), vl);
                    __riscv_vse32_v_f32m4(static_cast<float *>(s + ((i_mr + 7) * bs + (i_nr))),
                                          __riscv_vfsub_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr * 7, vl), __riscv_vle32_v_f32m4(sum_min_row + nr * 7, vl), vl), vl);
                    __riscv_vse32_v_f32m4(static_cast<float *>(s + ((i_mr + 8) * bs + (i_nr))),
                                          __riscv_vfsub_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr * 8, vl), __riscv_vle32_v_f32m4(sum_min_row + nr * 8, vl), vl), vl);
                    __riscv_vse32_v_f32m4(static_cast<float *>(s + ((i_mr + 9) * bs + (i_nr))),
                                          __riscv_vfsub_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr * 9, vl), __riscv_vle32_v_f32m4(sum_min_row + nr * 9, vl), vl), vl);
                    __riscv_vse32_v_f32m4(static_cast<float *>(s + ((i_mr + 10) * bs + (i_nr))),
                                          __riscv_vfsub_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr * 10, vl), __riscv_vle32_v_f32m4(sum_min_row + nr * 10, vl), vl), vl);
                    __riscv_vse32_v_f32m4(static_cast<float *>(s + ((i_mr + 11) * bs + (i_nr))),
                                          __riscv_vfsub_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr * 11, vl), __riscv_vle32_v_f32m4(sum_min_row + nr * 11, vl), vl), vl);
                }
            }
        }
    }
}
