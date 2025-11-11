//基于gemv2，将激活矩阵的d，fp16转为fp32提到循环外，decode 阶段达到5.52
#define QK4_0 32
// typedef struct {
//     ggml_half d;           // delta
//     uint8_t qs[QK4_0 / 2]; // nibbles / quants
// } block_q4_0;


static void ggml_gemv_q4_0_8x32_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    //矩阵乘实际上是1*n*nc

    const int qk = QK4_0; // 32
    const int nb = n / qk; // 多少个量化块
    // const int ncols_interleaved = 8;
    // const int blocklen = 8;

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
    

    // Pointers to LHS blocks of block_q8_K format 
    const block_q8_0 * a_ptr = a_ptr_start;

    float a_scales[nb];
    for(int i = 0; i < nb; i++)
        a_scales[i] = GGML_FP16_TO_FP32(a_ptr[i].d);

    // Take group of eight interleaved block_q4_K structures at each pass of the loop and perform dot product operation 
    for (int64_t x = 0; x < nc / 32; x++) {  // 每次处理 32 k 权重矩阵 

        // Pointers to RHS blocks 
        const block_q4_0x32 * b_ptr = b_ptr_start + (x * nb); // 32行的权重矩阵起始位置

        // 行累加和 
        vfloat32m4_t sum_rows = __riscv_vfmv_v_f_f32m4(0.0f, vl); 

        for (int64_t b = 0; b < nb; b++) {  //遍历6个量化块，每个块256*8 
            

            vint16m2_t suml_32_0 = __riscv_vmv_v_x_i16m2(0, vl); 
            for(int z = 0; z < 8; z++){

                //权重矩阵的0-3列和32-35列
                vint8m1_t src0_0 = __riscv_vle8_v_i8m1(b_ptr[b].qs + z * 32, vl);
                vint8m1_t low4bits_0 = __riscv_vsra_vx_i8m1(__riscv_vsll_vx_i8m1(src0_0, 4, vl), 4, vl); //位操作是不是用整形单元
                //输出矩阵一行的一个位置的低32个数和32-64个数
                const int64_t sb_32 = z * 2; 
                suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[b].qs[sb_32], low4bits_0, vl); 
                low4bits_0 = __riscv_vsra_vx_i8m1(src0_0, 4, vl); 
                suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[b].qs[sb_32 + 1], low4bits_0, vl); 

            }

            vint32m4_t suml_32_00 = __riscv_vmv_v_x_i32m4(0, vl);

            suml_32_00 = __riscv_vwadd_wv_i32m4(suml_32_00, suml_32_0, vl);

            suml_32_0 = __riscv_vmv_v_x_i16m2(0, vl);
            for(int z = 8; z < 16; z++){

                //权重矩阵的0-3列和32-35列
                vint8m1_t src0_0 = __riscv_vle8_v_i8m1(b_ptr[b].qs + z * 32, vl);
                vint8m1_t low4bits_0 = __riscv_vsra_vx_i8m1(__riscv_vsll_vx_i8m1(src0_0, 4, vl), 4, vl);//位操作是不是用整形单元
                //输出矩阵一行的一个位置的低32个数和32-64个数
                const int64_t sb_32 = z * 2;
                suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[b].qs[sb_32], low4bits_0, vl);
                low4bits_0 = __riscv_vsra_vx_i8m1(src0_0, 4, vl);
                suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[b].qs[sb_32 + 1], low4bits_0, vl);

            }

            suml_32_00 = __riscv_vwadd_wv_i32m4(suml_32_00, suml_32_0, vl);

            // vint16m2_t vec_scales_32 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vle16_v_u16m2(b_ptr[b].d, vl));
            // sum_rows = __riscv_vfmacc_vf_f32m4(sum_rows, GGML_FP16_TO_FP32(a_ptr[b].d), __riscv_vfcvt_f_x_v_f32m4(__riscv_vwmul_vv_i32m4(suml_32_0, vec_scales_32, vl), vl), vl);
            
            vfloat32m4_t vec_scales_32 = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vreinterpret_v_u16m2_f16m2(__riscv_vle16_v_u16m2(b_ptr[b].d, vl)), vl);
            
            sum_rows = __riscv_vfmacc_vv_f32m4(sum_rows, __riscv_vfcvt_f_x_v_f32m4(suml_32_00, vl), __riscv_vfmul_vf_f32m4(vec_scales_32, a_scales[b], vl), vl);

            // vint16m2_t vec_scales_32 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vle16_v_u16m2(b_ptr[b].d, vl));
            // sum_rows = __riscv_vfmacc_vv_f32m4(sum_rows, __riscv_vfcvt_f_x_v_f32m4(suml_32_00, vl), __riscv_vfcvt_f_x_v_f32m4(__riscv_vwmul_vx_i32m4(vec_scales_32, a_ptr[b].d, vl), vl), vl);

        }

        __riscv_vse32_v_f32m4(s + x * 32, sum_rows, vl);

    }

}