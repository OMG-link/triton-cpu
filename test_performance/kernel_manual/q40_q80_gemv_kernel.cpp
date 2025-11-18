#include "ggml_def.h"
#ifdef _OPENMP
#include <omp.h>
#endif
//基于gemv2，将激活矩阵的d，fp16转为fp32提到循环外，decode 阶段达到5.52
#define QK4_0 32
// typedef struct { 
//     ggml_half d;           // delta 
//     uint8_t qs[QK4_0 / 2]; // nibbles / quants 
// } block_q4_0; 

// vx: q4_0 // 怎么个逻辑 block_q4_0x32 
// vy: q8_0 // 怎么个逻辑 block_q8_0 
// n: k 维度 
// nr: m 维度，这里是 1 ，GEMV 
// nc:  n 维度 

// q4_0 [N/nr][K/QK4_0][QK4_0][nr] 
// q8_0 [K/QK4_0][QK4_0] 

void ggml_gemv_q4_0_8x32_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    //矩阵乘实际上是1*n*nc 

    const int qk = QK4_0; // 32 
    const int nb = n / qk; // 多少个量化块 

    assert (n % qk == 0);
    // assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);


    size_t vl = 32;   //vl = 32

    const block_q4_0x32 * b_ptr_start = (const block_q4_0x32 *)vx; // 右操作数 权重 
    const block_q8_0 * a_ptr_start = (const block_q8_0 *)vy; // 左操作数 激活值 
    
    const block_q8_0 * a_ptr = a_ptr_start;

    float a_scales[nb];
    
    // n / qk  
    // 输入激活向量  
    for(int i = 0; i < nb; i++) // k 维度多少 block 
        a_scales[i] = float(a_ptr[i].d);


    // n 维度 
    // 32 由寄存器宽度决定 
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(8) 
    #endif

    for (int64_t x = 0; x < nc / 32; x++) {  // n 维度 

        // block_q4_0x32 32n * 32k 
        const block_q4_0x32 * b_ptr = b_ptr_start + (x * nb); // n_block, 每个跨整个k 的 n 维度列的初始位置 

        // 行累加和 
        vfloat32m4_t sum_rows = __riscv_vfmv_v_f_f32m4(0.0f, vl); // 浮点寄存器，4 倍寄存器宽度 

        for (int64_t b = 0; b < nb; b++) {  // k 维度，所有 k 的子块 每个 block k = 32 长度
            
            vint16m2_t suml_32_0 = __riscv_vmv_v_x_i16m2(0, vl); // int16 中间结果，两倍寄存器宽度 

            // 带宽 7.xx GB
            // 1.6 GHz
            // SIMD 256bit 32 byte
            // 1.6 * 32 / 2 = 25.7 GB/s 理论可以占用 带宽
            // 25.7 / 7 = 4 约等于 8 个时钟周期有一次发射即可
            
            #pragma unroll 8
            for(int z = 0; z < 8; z++) { // k 维度迭代
                // 32 个元素 
                vint8m1_t src0_0 = __riscv_vle8_v_i8m1(b_ptr[b].qs + z * 32, vl);
                // vint8m1_t src1_0 = __riscv_vle8_v_i8m1(b_ptr[b].qs + z * 32, vl);
                // 向量算术右移指令 Vector Shift Right Arithmetic  
                    // vint8m1_t __riscv_vsra_vx_i8m1(vint8m1_t vs2, size_t rs1, size_t vl) 
                    // vs2: 源向量寄存器 
                    // rs1: 标量移位量（shift amount） 
                    // vl: 向量长度返回值: 移位后的向量 
                    // __riscv_vsll_vx_i8m1 先左移4位，再算术右移4位，得到低4位 
            
                // 作用：提取并符号扩展低 4 位 
                vint8m1_t low4bits_0 = __riscv_vsra_vx_i8m1(__riscv_vsll_vx_i8m1(src0_0, 4, vl), 4, vl); 
                const int64_t sb_32 = z * 2; // A[0/2/4/6/8/10/12/14] 
                suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[b].qs[sb_32], low4bits_0, vl);

                // 高 4bit // B[1,3,5,7,9,11,13,15]
                low4bits_0 = __riscv_vsra_vx_i8m1(src0_0, 4, vl); 
                suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[b].qs[sb_32 + 1], low4bits_0, vl); 
            }
            
            vint32m4_t suml_32_00 = __riscv_vmv_v_x_i32m4(0, vl); 
            
            // 处理精度的问题 
            suml_32_00 = __riscv_vwadd_wv_i32m4(suml_32_00, suml_32_0, vl);

            suml_32_0 = __riscv_vmv_v_x_i16m2(0, vl);
            
            // k = 16 ～ 31 
            #pragma unroll 8
            for(int z = 8; z < 16; z++) { // k 维度
                vint8m1_t src0_0 = __riscv_vle8_v_i8m1(b_ptr[b].qs + z * 32, vl);
                
                // 低四位 
                vint8m1_t low4bits_0 = __riscv_vsra_vx_i8m1(__riscv_vsll_vx_i8m1(src0_0, 4, vl), 4, vl);
                const int64_t sb_32 = z * 2;
                suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[b].qs[sb_32], low4bits_0, vl);
                
                // 高四位 
                low4bits_0 = __riscv_vsra_vx_i8m1(src0_0, 4, vl);
                suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[b].qs[sb_32 + 1], low4bits_0, vl);
            }

            suml_32_00 = __riscv_vwadd_wv_i32m4(suml_32_00, suml_32_0, vl);
            
            // 整个 k 的所有子块 
            // float sum_rows[32] // n 维度，每个元素 代表了 [1,k] * [k * 1] 

            // 权重 scales 
            vfloat32m4_t vec_scales_32 = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vreinterpret_v_u16m2_f16m2(__riscv_vle16_v_u16m2(b_ptr[b].d, vl)), vl);
            // sum_rows[i] ← sum_rows[i] + ( float(suml_32_00[i]) ×  vec_scales_32[i] × a_scales[b]) 
            sum_rows = __riscv_vfmacc_vv_f32m4(sum_rows, __riscv_vfcvt_f_x_v_f32m4(suml_32_00, vl), __riscv_vfmul_vf_f32m4(vec_scales_32, a_scales[b], vl), vl);
        }

        // 每次求结果向量的 32 个值，每个值表示 [1,k] * [k * 1]的结果规约后情况 
        __riscv_vse32_v_f32m4(s + x * 32, sum_rows, vl); 
    }
}