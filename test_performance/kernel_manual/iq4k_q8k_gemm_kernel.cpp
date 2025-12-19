// #include "ggml_def.h"
#include <cstdio>
#include <algorithm>
#include <string>
#include <riscv_vector.h>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#ifdef __cplusplus
// restrict not standard in C++
#if defined(__GNUC__)
#define GGML_RESTRICT __restrict__
#elif defined(__clang__)
#define GGML_RESTRICT __restrict
#elif defined(_MSC_VER)
#define GGML_RESTRICT __restrict
#else
#define GGML_RESTRICT
#endif
#else
#if defined(_MSC_VER) && (__STDC_VERSION__ < 201112L)
#define GGML_RESTRICT __restrict
#else
#define GGML_RESTRICT restrict
#endif
#endif

#define UNUSED(x) (void)(x)

typedef uint16_t ggml_half;

#define QK_K 256
#define QK_SB_K 32


// 自定义带消息的断言宏
#ifndef NDEBUG
    #define ASSERT_MSG(cond, fmt, ...) \
        do { \
            if (!(cond)) { \
                fprintf(stderr, "[ASSERT] %s:%d: " fmt "\n", \
                        __FILE__, __LINE__, ##__VA_ARGS__); \
                fprintf(stderr, "  Failed: %s\n", #cond); \
                abort(); \
            } \
        } while(0)
#else
    #define ASSERT_MSG(cond, fmt, ...) ((void)0)
#endif



// VL * QK
template <int VL> struct block_iq4_Kx { 
	ggml_half d[VL];                           // super-block scale for quantized scales
	uint16_t extra[VL];                        // extra information
	uint8_t scales[VL * (QK_K / QK_SB_K) * 2]; // scales and mins, quantized with 6 bits, but stored with 8 bits
	uint8_t qs[VL * QK_K / 2];                 // 4--bit quants
};

template <int VL> struct block_q8_Kx { 
	float d[VL];                         // delta
	uint8_t qs[VL * QK_K];               // quants
	uint16_t bsums[VL * QK_K / QK_SB_K]; // sum of quants in groups of 32
};

const int8_t iq4k_values[32] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    -123, -100, -79, -61, -45, -31, -18,  -6, 5, 17, 29, 42, 57, 73, 93, 117
};


using block_iq4_Kx32 = block_iq4_Kx<32>;
using block_q8_Kx12 = block_q8_Kx<12>;



void ggml_gemm_iq4_K_12x32_q8_K(int k, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT iq4k, const void *GGML_RESTRICT q8k, int m, int n) {
	int vl = __riscv_vlenb(); // 字节数=VLEN/8

	const int mr = 12;
	const int nr = 32;

	const int mc = mr * 4;
	const int nc = nr * 4;

	assert(vl == nr);
	assert(k % QK_K == 0);
	assert(m % mr == 0);
	assert(n % nr == 0);

	// 加载整个 32 元素的 iq4k_values 表到向量寄存器
	vint8m1_t iq4k_table_all = __riscv_vle8_v_i8m1((const int8_t *)iq4k_values, vl);

	auto iq4k_ptr_start = (const block_iq4_Kx32 *)iq4k;
	auto q8k_ptr_start = (const block_q8_Kx12 *)q8k;


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
					auto iq4k_ptr = iq4k_ptr_start + (i_nr * k) / (nr * QK_K); 
					auto q8k_ptr = q8k_ptr_start + (i_mr * k) / (mr * QK_K); // 激活

					float sum_row[mr * nr] = {0.0};
					
					for (int b = 0; b < k / QK_K; b++) { // K: superblock 
						int sum_block[mr * nr]; // int32
						vuint16m2_t extra_vec = __riscv_vle16_v_u16m2(
							(const uint16_t *)&iq4k_ptr[b].extra, vl);
						for (int sb = 0; sb < QK_K / QK_SB_K; sb++) { // K: 处理整个 superblock，每个 subblock 寄存器内核 循环展开 
							vuint8m1_t src0_0, src0_0_prefetch;
							vint16m2_t suml_32_0, suml_32_1, suml_32_2, suml_32_3, suml_32_4, suml_32_5;
							vint16m2_t suml_32_6, suml_32_7, suml_32_8, suml_32_9, suml_32_10, suml_32_11;
							// ✅ 外积布局核心 12（n 维度） * 32（k 维度）的权重 12 个子块连续 
							// ✅ 每个子块 32 个 4-bit 量化值，打包存储在 16 字节中 
							// ✅ 超块按照 QK_K/QK_SB_K = 8 * 32 在 k 维度横向堆叠 
							const uint64_t *q8k_ptr_temp = static_cast<const uint64_t *>((void *)&q8k_ptr[b].qs[sb * 384]); // mr * QK_SB_K = 12 * 32 = 384  
							vuint16m2_t extra_vec_shift = __riscv_vsrl_vx_u16m2(extra_vec, 2 * sb, vl); // [NR] @ uint16
							// 窄化为 uint8 类型，截断高位字节
							vuint8m1_t extra_vec_shift_sub_block = __riscv_vncvt_x_x_w_u8m1(extra_vec_shift, vl);
							// 提取 block 位信息，1 和 2
							vuint8m1_t tlb_start_idx_1_8 = __riscv_vand_vx_u8m1(extra_vec_shift_sub_block, 0x01, vl); // [NR]
							vuint8m1_t tlb_start_idx_2_8 = __riscv_vand_vx_u8m1(extra_vec_shift_sub_block, 0x02, vl); // [NR]
							// 左移 4 bit
							vuint8m1_t tlb_start_idx_1_8_shift = __riscv_vsll_vx_u8m1(tlb_start_idx_1_8, 4, vl);
							vuint8m1_t tlb_start_idx_2_8_shift = __riscv_vsll_vx_u8m1(tlb_start_idx_2_8, 4, vl);

							// QK_SB_K = 32 迭代循环 
							// 内核贡献 1.5 
							{    const int z = 0;
								// b 权重，qs 存储的是 4-bit 量化值，两个值打包在一个字节中 
								// 这里需要将 4-bit 量化值拆分出来 
								// nr 个 subblock 大小，32 * 16B (32 * 4 bit)= 512 字节 
								src0_0 = __riscv_vle8_v_u8m1(iq4k_ptr[b].qs + sb * 512 + z * nr, vl);
								src0_0_prefetch = __riscv_vle8_v_u8m1(iq4k_ptr[b].qs + sb * 512 + (z + 1) * nr, vl);
								vuint8m1_t src0_0_and = __riscv_vand_vx_u8m1(src0_0, 0x0F, vl);
								// 拼接 extra 位 与 src0_0_and
								vuint8m1_t src0_0_and_extra = __riscv_vor_vv_u8m1(src0_0_and, tlb_start_idx_1_8_shift, vl);
								vint8m1_t low4bits_0 = __riscv_vrgather_vv_i8m1(iq4k_table_all, src0_0_and_extra, vl);
								uint64_t a_0_7 = *q8k_ptr_temp++;
								suml_32_0 = __riscv_vwmul_vx_i16m2(low4bits_0, (a_0_7 >> (0 * 8)) & 0xff, vl); 
								suml_32_1 = __riscv_vwmul_vx_i16m2(low4bits_0, (a_0_7 >> (1 * 8)) & 0xff, vl);
								suml_32_2 = __riscv_vwmul_vx_i16m2(low4bits_0, (a_0_7 >> (2 * 8)) & 0xff, vl);
								suml_32_3 = __riscv_vwmul_vx_i16m2(low4bits_0, (a_0_7 >> (3 * 8)) & 0xff, vl);
								suml_32_4 = __riscv_vwmul_vx_i16m2(low4bits_0, (a_0_7 >> (4 * 8)) & 0xff, vl);
								suml_32_5 = __riscv_vwmul_vx_i16m2(low4bits_0, (a_0_7 >> (5 * 8)) & 0xff, vl);
								suml_32_6 = __riscv_vwmul_vx_i16m2(low4bits_0, (a_0_7 >> (6 * 8)) & 0xff, vl);
								suml_32_7 = __riscv_vwmul_vx_i16m2(low4bits_0, (a_0_7 >> (7 * 8)) & 0xff, vl);
								uint64_t a_8_15 = *q8k_ptr_temp++;
								suml_32_8 = __riscv_vwmul_vx_i16m2(low4bits_0, (a_8_15 >> (0 * 8)) & 0xff, vl);
								suml_32_9 = __riscv_vwmul_vx_i16m2(low4bits_0, (a_8_15 >> (1 * 8)) & 0xff, vl);
								suml_32_10 = __riscv_vwmul_vx_i16m2(low4bits_0, (a_8_15 >> (2 * 8)) & 0xff, vl);
								suml_32_11 = __riscv_vwmul_vx_i16m2(low4bits_0, (a_8_15 >> (3 * 8)) & 0xff, vl);
								
								// src0_0 右移 4bit
								vuint8m1_t src0_0_shift = __riscv_vsrl_vx_u8m1(src0_0, 4, vl);
								src0_0_and_extra = __riscv_vor_vv_u8m1(src0_0_shift, tlb_start_idx_2_8_shift, vl);
								low4bits_0 = __riscv_vrgather_vv_i8m1(iq4k_table_all, src0_0_and_extra, vl);
								suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, (a_8_15 >> (4 * 8)) & 0xff, low4bits_0, vl);
								suml_32_1 = __riscv_vwmacc_vx_i16m2(suml_32_1, (a_8_15 >> (5 * 8)) & 0xff, low4bits_0, vl);
								suml_32_2 = __riscv_vwmacc_vx_i16m2(suml_32_2, (a_8_15 >> (6 * 8)) & 0xff, low4bits_0, vl);
								suml_32_3 = __riscv_vwmacc_vx_i16m2(suml_32_3, (a_8_15 >> (7 * 8)) & 0xff, low4bits_0, vl);
								uint64_t a_16_23 = *q8k_ptr_temp++;
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
									src0_0_prefetch = __riscv_vle8_v_u8m1(iq4k_ptr[b].qs + sb * 512 + (z + 1) * nr, vl);
									vuint8m1_t src0_0_and = __riscv_vand_vx_u8m1(src0_0, 0x0F, vl);
									vuint8m1_t src0_0_and_extra = __riscv_vor_vv_u8m1(src0_0_and, tlb_start_idx_1_8_shift, vl);
									vint8m1_t low4bits_0 = __riscv_vrgather_vv_i8m1(iq4k_table_all, src0_0_and_extra, vl);
									uint64_t a_0_7 = *q8k_ptr_temp++;
									suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, (a_0_7 >> (0 * 8)) & 0xff, low4bits_0, vl);
									suml_32_1 = __riscv_vwmacc_vx_i16m2(suml_32_1, (a_0_7 >> (1 * 8)) & 0xff, low4bits_0, vl);
									suml_32_2 = __riscv_vwmacc_vx_i16m2(suml_32_2, (a_0_7 >> (2 * 8)) & 0xff, low4bits_0, vl);
									suml_32_3 = __riscv_vwmacc_vx_i16m2(suml_32_3, (a_0_7 >> (3 * 8)) & 0xff, low4bits_0, vl);
									suml_32_4 = __riscv_vwmacc_vx_i16m2(suml_32_4, (a_0_7 >> (4 * 8)) & 0xff, low4bits_0, vl);
									suml_32_5 = __riscv_vwmacc_vx_i16m2(suml_32_5, (a_0_7 >> (5 * 8)) & 0xff, low4bits_0, vl);
									suml_32_6 = __riscv_vwmacc_vx_i16m2(suml_32_6, (a_0_7 >> (6 * 8)) & 0xff, low4bits_0, vl);
									suml_32_7 = __riscv_vwmacc_vx_i16m2(suml_32_7, (a_0_7 >> (7 * 8)) & 0xff, low4bits_0, vl);
									uint64_t a_8_15 = *q8k_ptr_temp++;
									suml_32_8 = __riscv_vwmacc_vx_i16m2(suml_32_8, (a_8_15 >> (0 * 8)) & 0xff, low4bits_0, vl);
									suml_32_9 = __riscv_vwmacc_vx_i16m2(suml_32_9, (a_8_15 >> (1 * 8)) & 0xff, low4bits_0, vl);
									suml_32_10 = __riscv_vwmacc_vx_i16m2(suml_32_10, (a_8_15 >> (2 * 8)) & 0xff, low4bits_0, vl);
									suml_32_11 = __riscv_vwmacc_vx_i16m2(suml_32_11, (a_8_15 >> (3 * 8)) & 0xff, low4bits_0, vl);
									
									vuint8m1_t src0_0_shift = __riscv_vsrl_vx_u8m1(src0_0, 4, vl);
									src0_0_and_extra = __riscv_vor_vv_u8m1(src0_0_shift, tlb_start_idx_2_8_shift, vl);
									low4bits_0 = __riscv_vrgather_vv_i8m1(iq4k_table_all, src0_0_and_extra, vl);
									suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, (a_8_15 >> (4 * 8)) & 0xff, low4bits_0, vl);
									suml_32_1 = __riscv_vwmacc_vx_i16m2(suml_32_1, (a_8_15 >> (5 * 8)) & 0xff, low4bits_0, vl);
									suml_32_2 = __riscv_vwmacc_vx_i16m2(suml_32_2, (a_8_15 >> (6 * 8)) & 0xff, low4bits_0, vl);
									suml_32_3 = __riscv_vwmacc_vx_i16m2(suml_32_3, (a_8_15 >> (7 * 8)) & 0xff, low4bits_0, vl);
									uint64_t a_16_23 = *q8k_ptr_temp++;
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
									vuint8m1_t src0_0_and = __riscv_vand_vx_u8m1(src0_0, 0x0F, vl);
									vuint8m1_t src0_0_and_extra = __riscv_vor_vv_u8m1(src0_0_and, tlb_start_idx_1_8_shift, vl);
									vint8m1_t low4bits_0 = __riscv_vrgather_vv_i8m1(iq4k_table_all, src0_0_and_extra, vl);
									uint64_t a_0_7 = *q8k_ptr_temp++;
									suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, (a_0_7 >> (0 * 8)) & 0xff, low4bits_0, vl);
									suml_32_1 = __riscv_vwmacc_vx_i16m2(suml_32_1, (a_0_7 >> (1 * 8)) & 0xff, low4bits_0, vl);
									suml_32_2 = __riscv_vwmacc_vx_i16m2(suml_32_2, (a_0_7 >> (2 * 8)) & 0xff, low4bits_0, vl);
									suml_32_3 = __riscv_vwmacc_vx_i16m2(suml_32_3, (a_0_7 >> (3 * 8)) & 0xff, low4bits_0, vl);
									suml_32_4 = __riscv_vwmacc_vx_i16m2(suml_32_4, (a_0_7 >> (4 * 8)) & 0xff, low4bits_0, vl);
									suml_32_5 = __riscv_vwmacc_vx_i16m2(suml_32_5, (a_0_7 >> (5 * 8)) & 0xff, low4bits_0, vl);
									suml_32_6 = __riscv_vwmacc_vx_i16m2(suml_32_6, (a_0_7 >> (6 * 8)) & 0xff, low4bits_0, vl);
									suml_32_7 = __riscv_vwmacc_vx_i16m2(suml_32_7, (a_0_7 >> (7 * 8)) & 0xff, low4bits_0, vl);
									uint64_t a_8_15 = *q8k_ptr_temp++;
									suml_32_8 = __riscv_vwmacc_vx_i16m2(suml_32_8, (a_8_15 >> (0 * 8)) & 0xff, low4bits_0, vl);
									suml_32_9 = __riscv_vwmacc_vx_i16m2(suml_32_9, (a_8_15 >> (1 * 8)) & 0xff, low4bits_0, vl);
									suml_32_10 = __riscv_vwmacc_vx_i16m2(suml_32_10, (a_8_15 >> (2 * 8)) & 0xff, low4bits_0, vl);
									suml_32_11 = __riscv_vwmacc_vx_i16m2(suml_32_11, (a_8_15 >> (3 * 8)) & 0xff, low4bits_0, vl);
									
									vuint8m1_t src0_0_shift = __riscv_vsrl_vx_u8m1(src0_0, 4, vl);
									src0_0_and_extra = __riscv_vor_vv_u8m1(src0_0_shift, tlb_start_idx_2_8_shift, vl);
									low4bits_0 = __riscv_vrgather_vv_i8m1(iq4k_table_all, src0_0_and_extra, vl);
									suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, (a_8_15 >> (4 * 8)) & 0xff, low4bits_0, vl);
									suml_32_1 = __riscv_vwmacc_vx_i16m2(suml_32_1, (a_8_15 >> (5 * 8)) & 0xff, low4bits_0, vl);
									suml_32_2 = __riscv_vwmacc_vx_i16m2(suml_32_2, (a_8_15 >> (6 * 8)) & 0xff, low4bits_0, vl);
									suml_32_3 = __riscv_vwmacc_vx_i16m2(suml_32_3, (a_8_15 >> (7 * 8)) & 0xff, low4bits_0, vl);
									uint64_t a_16_23 = *q8k_ptr_temp++;
									suml_32_4 = __riscv_vwmacc_vx_i16m2(suml_32_4, (a_16_23 >> (0 * 8)) & 0xff, low4bits_0, vl);
									suml_32_5 = __riscv_vwmacc_vx_i16m2(suml_32_5, (a_16_23 >> (1 * 8)) & 0xff, low4bits_0, vl);
									suml_32_6 = __riscv_vwmacc_vx_i16m2(suml_32_6, (a_16_23 >> (2 * 8)) & 0xff, low4bits_0, vl);
									suml_32_7 = __riscv_vwmacc_vx_i16m2(suml_32_7, (a_16_23 >> (3 * 8)) & 0xff, low4bits_0, vl);
									suml_32_8 = __riscv_vwmacc_vx_i16m2(suml_32_8, (a_16_23 >> (4 * 8)) & 0xff, low4bits_0, vl);
									suml_32_9 = __riscv_vwmacc_vx_i16m2(suml_32_9, (a_16_23 >> (5 * 8)) & 0xff, low4bits_0, vl);
									suml_32_10 = __riscv_vwmacc_vx_i16m2(suml_32_10, (a_16_23 >> (6 * 8)) & 0xff, low4bits_0, vl);
									suml_32_11 = __riscv_vwmacc_vx_i16m2(suml_32_11, (a_16_23 >> (7 * 8)) & 0xff, low4bits_0, vl);
							}
							

							// subblock scale
							// load subblock scale
							// 累加到 sum_block
							vint16m2_t vec_scales_32 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(__riscv_vle8_v_u8m1(iq4k_ptr[b].scales + sb * nr, vl), vl));

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

						// super block scale
						vfloat32m4_t iq4k_super_block_scale = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vreinterpret_v_u16m2_f16m2(__riscv_vle16_v_u16m2(iq4k_ptr[b].d, vl)), vl);
						__riscv_vse32_v_f32m4(sum_row + nr * 0,
							__riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row, vl), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_block, vl), vl),
																			__riscv_vfmul_vf_f32m4(iq4k_super_block_scale, q8k_ptr[b].d[0], vl), vl),
							vl);
						__riscv_vse32_v_f32m4(sum_row + nr * 1, __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr, vl), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_block + nr, vl), vl),
											__riscv_vfmul_vf_f32m4(iq4k_super_block_scale, q8k_ptr[b].d[1], vl), vl), vl);
						__riscv_vse32_v_f32m4(sum_row + nr * 2,__riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr * 2, vl), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_block + nr * 2, vl), vl),
											__riscv_vfmul_vf_f32m4(iq4k_super_block_scale, q8k_ptr[b].d[2], vl), vl), vl);
						__riscv_vse32_v_f32m4(sum_row + nr * 3, __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr * 3, vl), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_block + nr * 3, vl), vl),
											__riscv_vfmul_vf_f32m4(iq4k_super_block_scale, q8k_ptr[b].d[3], vl), vl), vl);
						__riscv_vse32_v_f32m4(sum_row + nr * 4, __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr * 4, vl), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_block + nr * 4, vl), vl),
												__riscv_vfmul_vf_f32m4(iq4k_super_block_scale, q8k_ptr[b].d[4], vl), vl),vl);
						__riscv_vse32_v_f32m4(sum_row + nr * 5,	__riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr * 5, vl), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_block + nr * 5, vl), vl),
													__riscv_vfmul_vf_f32m4(iq4k_super_block_scale, q8k_ptr[b].d[5], vl), vl), vl);
						__riscv_vse32_v_f32m4(sum_row + nr * 6, __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr * 6, vl), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_block + nr * 6, vl), vl),
													__riscv_vfmul_vf_f32m4(iq4k_super_block_scale, q8k_ptr[b].d[6], vl), vl),vl);
						__riscv_vse32_v_f32m4(sum_row + nr * 7, 	__riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr * 7, vl), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_block + nr * 7, vl), vl),
													__riscv_vfmul_vf_f32m4(iq4k_super_block_scale, q8k_ptr[b].d[7], vl), vl),	vl);
						__riscv_vse32_v_f32m4(sum_row + nr * 8,	__riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr * 8, vl), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_block + nr * 8, vl), vl),
													__riscv_vfmul_vf_f32m4(iq4k_super_block_scale, q8k_ptr[b].d[8], vl), vl),vl);
						__riscv_vse32_v_f32m4(sum_row + nr * 9,	__riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr * 9, vl), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_block + nr * 9, vl), vl),
													__riscv_vfmul_vf_f32m4(iq4k_super_block_scale, q8k_ptr[b].d[9], vl), vl),vl);
						__riscv_vse32_v_f32m4(sum_row + nr * 10,	__riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr * 10, vl), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_block + nr * 10, vl), vl),
														__riscv_vfmul_vf_f32m4(iq4k_super_block_scale, q8k_ptr[b].d[10], vl), vl),vl);
						__riscv_vse32_v_f32m4(sum_row + nr * 11,	__riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr * 11, vl), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_block + nr * 11, vl), vl),
													__riscv_vfmul_vf_f32m4(iq4k_super_block_scale, q8k_ptr[b].d[11], vl), vl),vl);
					}
				}
				
			}
		
		}
	
	}

	
}