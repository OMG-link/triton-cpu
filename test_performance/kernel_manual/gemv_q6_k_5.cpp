static void ggml_gemv_q6_K_8x32_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    //矩阵乘实际上是1*n*nc

    const int qk = QK_K;    //256
    const int nb = n / qk;  //1536/256=6

    assert (n % qk == 0);
    // assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nb);
    size_t vl = 32;   //vl = 32

    const block_q6_Kx32 * b_ptr_start = (const block_q6_Kx32 *)vx; // 右操作数 权重
    const block_q8_K * a_ptr_start = (const block_q8_K *)vy; // 左操作数 激活值
    
    for (int64_t y = 0; y < nr; y++) {  //nr=1 

        // Pointers to LHS blocks of block_q8_K format 
        const block_q8_K * a_ptr = a_ptr_start + (y * nb); 

        // Take group of eight interleaved block_q4_K structures at each pass of the loop and perform dot product operation
        for (int64_t x = 0; x < nc / 32; x++) {  // 每次处理32行

            // Pointers to RHS blocks
            const block_q6_Kx32 * b_ptr = b_ptr_start + (x * nb);

            // 行累加和
            vfloat32m4_t sum_rows = __riscv_vfmv_v_f_f32m4(0.0f, vl);

            for (int64_t b = 0; b < nb; b++) {  //遍历6个量化块，每个块256*8
                // Load and convert to FP32 scale from block_q8_K
                //d填充到256位数据中
                // vfloat32m1_t src1_d = __riscv_vfmv_v_f_f32m1(a_ptr[b].d, 8);

                vint32m4_t sum_block = __riscv_vmv_v_x_i32m4(0, vl); // 所有元素置零

                for (int sb = 0; sb < QK_K / 16; sb++) { //QK_K / 64=4,一次处理src0的64列
                    // vint16m2_t suml_32_0 = __riscv_vmv_v_x_i16m2(0, vl);

                    const int sb_z = sb * 384;
                    vuint8m1_t src0_l = __riscv_vle8_v_u8m1(b_ptr[b].qlhl + sb_z, vl);
                    vuint8m1_t ql_4bits = __riscv_vand_vx_u8m1(src0_l, 0x0F, vl);
                    vuint8m1_t src0_h = __riscv_vle8_v_u8m1(b_ptr[b].qlhl + sb_z + 32, vl);
                    vuint8m1_t qh_2bits = __riscv_vand_vx_u8m1(src0_h, 0x03, vl);
                    vuint8m1_t uqs_6 = __riscv_vor_vv_u8m1(ql_4bits, __riscv_vsll_vx_u8m1(qh_2bits, 0x04, vl), vl);
                    vint8m1_t qs_6 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(uqs_6), 32, vl);
                    //输出矩阵一行的一个位置的低32个数和32-64个数
                    const int64_t sb_32 = sb * 16;
                    vint16m2_t suml_32_0 = __riscv_vwmul_vx_i16m2(qs_6, a_ptr[b].qs[sb_32], vl);

                    ql_4bits = __riscv_vsrl_vx_u8m1(src0_l, 0x04, vl);
                    qh_2bits = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(src0_h, 0x02, vl), 0x03, vl);
                    uqs_6 = __riscv_vor_vv_u8m1(ql_4bits, __riscv_vsll_vx_u8m1(qh_2bits, 0x04, vl), vl);
                    qs_6 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(uqs_6), 32, vl);
                    suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[b].qs[sb_32 + 1], qs_6, vl);

                    src0_l = __riscv_vle8_v_u8m1(b_ptr[b].qlhl + sb_z + 64, vl);
                    ql_4bits = __riscv_vand_vx_u8m1(src0_l, 0x0F, vl);
                    qh_2bits = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(src0_h, 0x04, vl), 0x03, vl);
                    uqs_6 = __riscv_vor_vv_u8m1(ql_4bits, __riscv_vsll_vx_u8m1(qh_2bits, 0x04, vl), vl);
                    qs_6 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(uqs_6), 32, vl);
                    suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[b].qs[sb_32 + 2], qs_6, vl);

                    ql_4bits = __riscv_vsrl_vx_u8m1(src0_l, 0x04, vl);
                    qh_2bits = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(src0_h, 0x06, vl), 0x03, vl);
                    uqs_6 = __riscv_vor_vv_u8m1(ql_4bits, __riscv_vsll_vx_u8m1(qh_2bits, 0x04, vl), vl);
                    qs_6 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(uqs_6), 32, vl);
                    suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[b].qs[sb_32 + 3], qs_6, vl);

                    for(int z = 1; z < 4; z++){
                        //权重矩阵的0-3列和32-35列
                        const int sb_z = sb * 384 + z * 96;
                        src0_l = __riscv_vle8_v_u8m1(b_ptr[b].qlhl + sb_z, vl);
                        ql_4bits = __riscv_vand_vx_u8m1(src0_l, 0x0F, vl);
                        src0_h = __riscv_vle8_v_u8m1(b_ptr[b].qlhl + sb_z + 32, vl);
                        qh_2bits = __riscv_vand_vx_u8m1(src0_h, 0x03, vl);
                        uqs_6 = __riscv_vor_vv_u8m1(ql_4bits, __riscv_vsll_vx_u8m1(qh_2bits, 0x04, vl), vl);
                        qs_6 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(uqs_6), 32, vl);
                        //输出矩阵一行的一个位置的低32个数和32-64个数
                        const int64_t sb_32 = sb * 16 + z * 4;
                        suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[b].qs[sb_32], qs_6, vl);
                        ql_4bits = __riscv_vsrl_vx_u8m1(src0_l, 0x04, vl);
                        qh_2bits = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(src0_h, 0x02, vl), 0x03, vl);
                        uqs_6 = __riscv_vor_vv_u8m1(ql_4bits, __riscv_vsll_vx_u8m1(qh_2bits, 0x04, vl), vl);
                        qs_6 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(uqs_6), 32, vl);
                        suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[b].qs[sb_32 + 1], qs_6, vl);

                        src0_l = __riscv_vle8_v_u8m1(b_ptr[b].qlhl + sb_z + 64, vl);
                        ql_4bits = __riscv_vand_vx_u8m1(src0_l, 0x0F, vl);
                        qh_2bits = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(src0_h, 0x04, vl), 0x03, vl);
                        uqs_6 = __riscv_vor_vv_u8m1(ql_4bits, __riscv_vsll_vx_u8m1(qh_2bits, 0x04, vl), vl);
                        qs_6 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(uqs_6), 32, vl);
                        suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[b].qs[sb_32 + 2], qs_6, vl);

                        ql_4bits = __riscv_vsrl_vx_u8m1(src0_l, 0x04, vl);
                        qh_2bits = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(src0_h, 0x06, vl), 0x03, vl);
                        uqs_6 = __riscv_vor_vv_u8m1(ql_4bits, __riscv_vsll_vx_u8m1(qh_2bits, 0x04, vl), vl);
                        qs_6 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(uqs_6), 32, vl);
                        suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[b].qs[sb_32 + 3], qs_6, vl);

                    }

                    vint16m2_t vec_scales_32 = __riscv_vsext_vf2_i16m2(__riscv_vle8_v_i8m1(b_ptr[b].scales + sb * 32, vl), vl);

                    sum_block = __riscv_vwmacc_vv_i32m4(sum_block, suml_32_0, vec_scales_32, vl);

                }


                float src1_d = a_ptr[b].d;

                vfloat32m4_t src0_d = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vreinterpret_v_u16m2_f16m2(__riscv_vle16_v_u16m2(b_ptr[b].d, vl)) ,vl);

                sum_rows = __riscv_vfmacc_vv_f32m4(sum_rows, __riscv_vfcvt_f_x_v_f32m4(sum_block, vl), __riscv_vfmul_vf_f32m4(src0_d, src1_d, vl), vl);
            }

            __riscv_vse32_v_f32m4(s + (y * nr + x * 32), sum_rows, vl);

        }
    }

}