//最内层不展开, 0和1相邻，减少寄存器使用，寄存器服用，A矩阵访存连续
static void ggml_gemv_q4_K_8x32_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    //矩阵乘实际上是1*n*nc

    const int qk = QK_K;    //256
    const int nb = n / qk;  //1536/256=6
    const int ncols_interleaved = 8;
    const int blocklen = 8;
    static const uint32_t kmask1 = 0x3f3f3f3f;  //32位
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    assert (n % qk == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);


    size_t vl = 32;   //vl = 32

    // int64_t b_nb = n / QK_K;

    const block_q4_Kx32 * b_ptr_start = (const block_q4_Kx32 *)vx;
    const block_q8_K * a_ptr_start = (const block_q8_K *)vy;
    // vint16m1_t vzero = __riscv_vmv_v_x_i16m1(0, 1);
    // uint32_t utmp[4];
    uint32_t utmp[16];

    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[8];

    // const uint8_t * scales2 = (const uint8_t*)&utmp2[0];
    // const uint8_t * mins2   = (const uint8_t*)&utmp2[2];
    
    for (int64_t y = 0; y < nr; y++) {  //nr=1

        // Pointers to LHS blocks of block_q8_K format
        const block_q8_K * a_ptr = a_ptr_start + (y * nb);

        // Take group of eight interleaved block_q4_K structures at each pass of the loop and perform dot product operation
        for (int64_t x = 0; x < nc / 32; x++) {  //每次处理32行

            // Pointers to RHS blocks
            const block_q4_Kx32 * b_ptr = b_ptr_start + (x * nb);

            // 行累加和
            vfloat32m4_t sum_rows = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t sum_min_rows = __riscv_vfmv_v_f_f32m4(0.0f, vl);

            for (int64_t b = 0; b < nb; b++) {  //遍历6个量化块，每个块256*8
                // Load and convert to FP32 scale from block_q8_K
                //d填充到256位数据中
                // vfloat32m1_t src1_d = __riscv_vfmv_v_f_f32m1(a_ptr[b].d, 8);

                vint16mf2_t q8sums_0 = __riscv_vlse16_v_i16mf2(a_ptr[b].bsums, 4, 8);
                vint16mf2_t q8sums_1 = __riscv_vlse16_v_i16mf2(a_ptr[b].bsums+1, 4, 8);
                vint16mf2_t q8sums   = __riscv_vadd_vv_i16mf2(q8sums_0, q8sums_1, 8);
                int16_t sum[8];
                __riscv_vse16_v_i16mf2(sum, q8sums, 8);
                
                vint32m4_t sum_block = __riscv_vmv_v_x_i32m4(0, vl); // 所有元素置零
                vint32m4_t sum_min_block = __riscv_vmv_v_x_i32m4(0, vl);

                for (int sb = 0; sb < QK_K / 32; sb++) { //QK_K / 64=4,一次处理src0的64列
                    vint16m2_t suml_32_0 = __riscv_vmv_v_x_i16m2(0, vl);
                    for(int z = 0; z < 16; z++){
  
                        //权重矩阵的0-3列和32-35列
                        vuint8m1_t src0_0 = __riscv_vle8_v_u8m1(b_ptr[b].qs + sb * 512 + z * 32, vl);
                        vint8m1_t low4bits_0 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(src0_0, 0x0F, vl));//位操作是不是用整形单元
                        //输出矩阵一行的一个位置的低32个数和32-64个数
                        const int64_t sb_32 = sb * 32 + z * 2;
                        suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[b].qs[sb_32], low4bits_0, vl);
                        low4bits_0 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(src0_0, 4, vl));
                        suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[b].qs[sb_32 + 1], low4bits_0, vl);

                    }

                    //处理scales和mins
                    //是否可以向量化实现
                    memcpy(utmp, b_ptr[b].scales + 48 * sb, 48);
                    utmp[12] = ((utmp[8] >> 4) & kmask2) | (((utmp[4] >> 6) & kmask3) << 4);
                    utmp[13] = ((utmp[9] >> 4) & kmask2) | (((utmp[5] >> 6) & kmask3) << 4);
                    utmp[14] = ((utmp[10] >> 4) & kmask2) | (((utmp[6] >> 6) & kmask3) << 4);
                    utmp[15] = ((utmp[11] >> 4) & kmask2) | (((utmp[7] >> 6) & kmask3) << 4);
                    const uint32_t uaux = utmp[4] & kmask1;
                    const uint32_t uaux1 = utmp[5] & kmask1;
                    const uint32_t uaux2 = utmp[6] & kmask1;
                    const uint32_t uaux3 = utmp[7] & kmask1;
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

                    
                    sum_block = __riscv_vwmacc_vv_i32m4(sum_block, suml_32_0, vec_scales_32, vl);

                    //mins* qsums
                    const int16_t sum1 = sum[sb];
                    sum_min_block = __riscv_vwmacc_vx_i32m4(sum_min_block, sum1, vec_mins_32, vl);
                    
                }

                float src1_d = a_ptr[b].d;

                vfloat32m4_t src0_d = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vreinterpret_v_u16m2_f16m2(__riscv_vle16_v_u16m2(b_ptr[b].d, vl)) ,vl);
                vfloat32m4_t src0_dmin = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vreinterpret_v_u16m2_f16m2(__riscv_vle16_v_u16m2(b_ptr[b].dmin, vl)) ,vl);

                sum_min_rows = __riscv_vfmacc_vv_f32m4(sum_min_rows, __riscv_vfcvt_f_x_v_f32m4(sum_min_block, vl), __riscv_vfmul_vf_f32m4(src0_dmin, src1_d, vl), vl);
                sum_rows = __riscv_vfmacc_vv_f32m4(sum_rows, __riscv_vfcvt_f_x_v_f32m4(sum_block, vl), __riscv_vfmul_vf_f32m4(src0_d, src1_d, vl), vl);
            }

            __riscv_vse32_v_f32m4(s + (y * nr + x * 32), __riscv_vfsub_vv_f32m4(sum_rows, sum_min_rows, vl), vl);

        }
    }

}
