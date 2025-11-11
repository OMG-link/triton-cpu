//基于block_q6_kx32_1.cpp，重排数据布局

struct block_q6_Kx32 {
    // uint8_t ql[QK_K / 2 * 32];
    uint8_t qlhl[QK_K / 4 * 32 * 3];
    int8_t scales[QK_K / 16 * 32];
    ggml_half d[32];
};


void ggml_gemm_q6_K_8x32_q8_K(int k, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx, const void *GGML_RESTRICT vy, int m, int n) {
    int vl = __riscv_vlenb(); // 字节数=VLEN/8

    const int mr = 12;
    const int nr = 32;
    const int anr = m - m % mr;
    const int nb = k / QK_K;
    // const int QK_SB_K = 32;
    const int mnr = mr * nr;

    assert(vl == nr);
    assert(k % QK_K == 0);
    // assert(m % mr == 0);
    assert(n % nr == 0);

    auto b_ptr_start = (const block_q6_Kx32 *)vx;
    auto a_ptr_start = (const block_q8_Kx12 *)vy;

    for(int y = 0; y < anr / mr; y++){
        const block_q8_Kx12 *a_ptr = a_ptr_start + (y * nb);
        for (int x = 0; x < n / nr; x++) { // N

            const block_q6_Kx32 *b_ptr = b_ptr_start + (x * nb);

            float sum_row[mnr] = {0.0};

            // 行累加和
            for (int64_t b = 0; b < nb; b++) { // K

                int32_t sum_block[mnr] = {0};   //子块和

                // Loop to iterate over the eight sub blocks of a super block - two sub blocks are processed per
                // iteration
                for (int sb = 0; sb < QK_K / 16; sb++) {
                    // 是否可以通过循环展开for z隐藏
                    vuint8m1_t src0_l = __riscv_vle8_v_u8m1(b_ptr[b].qlhl + sb * 384, vl);
                    vuint8m1_t ql_4bits = __riscv_vand_vx_u8m1(src0_l, 0x0F, vl);
                    vuint8m1_t src0_h = __riscv_vle8_v_u8m1(b_ptr[b].qlhl + sb * 384 + 32, vl);
                    vuint8m1_t qh_2bits = __riscv_vand_vx_u8m1(src0_h, 0x03, vl);
                    vuint8m1_t uqs_6 = __riscv_vor_vv_u8m1(ql_4bits, __riscv_vsll_vx_u8m1(qh_2bits, 0x04, vl), vl);
                    vint8m1_t qs_6 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(uqs_6), 32, vl);

                    const int64_t ssb = sb * 192;
                    vint16m2_t suml_32_0 = __riscv_vwmul_vx_i16m2(qs_6, a_ptr[b].qs[ssb], vl);
                    vint16m2_t suml_32_1 = __riscv_vwmul_vx_i16m2(qs_6, a_ptr[b].qs[ssb + 1], vl);
                    vint16m2_t suml_32_2 = __riscv_vwmul_vx_i16m2(qs_6, a_ptr[b].qs[ssb + 2], vl);
                    vint16m2_t suml_32_3 = __riscv_vwmul_vx_i16m2(qs_6, a_ptr[b].qs[ssb + 3], vl);
                    vint16m2_t suml_32_4 = __riscv_vwmul_vx_i16m2(qs_6, a_ptr[b].qs[ssb + 4], vl);
                    vint16m2_t suml_32_5 = __riscv_vwmul_vx_i16m2(qs_6, a_ptr[b].qs[ssb + 5], vl);
                    vint16m2_t suml_32_6 = __riscv_vwmul_vx_i16m2(qs_6, a_ptr[b].qs[ssb + 6], vl);
                    vint16m2_t suml_32_7 = __riscv_vwmul_vx_i16m2(qs_6, a_ptr[b].qs[ssb + 7], vl);
                    vint16m2_t suml_32_8 = __riscv_vwmul_vx_i16m2(qs_6, a_ptr[b].qs[ssb + 8], vl);
                    vint16m2_t suml_32_9 = __riscv_vwmul_vx_i16m2(qs_6, a_ptr[b].qs[ssb + 9], vl);
                    vint16m2_t suml_32_10 = __riscv_vwmul_vx_i16m2(qs_6, a_ptr[b].qs[ssb + 10], vl);
                    vint16m2_t suml_32_11 = __riscv_vwmul_vx_i16m2(qs_6, a_ptr[b].qs[ssb + 11], vl);

                    ql_4bits = __riscv_vsrl_vx_u8m1(src0_l, 0x04, vl);
                    qh_2bits = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(src0_h, 0x02, vl), 0x03, vl);
                    uqs_6 = __riscv_vor_vv_u8m1(ql_4bits, __riscv_vsll_vx_u8m1(qh_2bits, 0x04, vl), vl);
                    qs_6 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(uqs_6), 32, vl);
                    
                    suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[b].qs[ssb + 12], qs_6, vl);
                    suml_32_1 = __riscv_vwmacc_vx_i16m2(suml_32_1, a_ptr[b].qs[ssb + 13], qs_6, vl);
                    suml_32_2 = __riscv_vwmacc_vx_i16m2(suml_32_2, a_ptr[b].qs[ssb + 14], qs_6, vl);
                    suml_32_3 = __riscv_vwmacc_vx_i16m2(suml_32_3, a_ptr[b].qs[ssb + 15], qs_6, vl);
                    suml_32_4 = __riscv_vwmacc_vx_i16m2(suml_32_4, a_ptr[b].qs[ssb + 16], qs_6, vl);
                    suml_32_5 = __riscv_vwmacc_vx_i16m2(suml_32_5, a_ptr[b].qs[ssb + 17], qs_6, vl);
                    suml_32_6 = __riscv_vwmacc_vx_i16m2(suml_32_6, a_ptr[b].qs[ssb + 18], qs_6, vl);
                    suml_32_7 = __riscv_vwmacc_vx_i16m2(suml_32_7, a_ptr[b].qs[ssb + 19], qs_6, vl);
                    suml_32_8 = __riscv_vwmacc_vx_i16m2(suml_32_8, a_ptr[b].qs[ssb + 20], qs_6, vl);
                    suml_32_9 = __riscv_vwmacc_vx_i16m2(suml_32_9, a_ptr[b].qs[ssb + 21], qs_6, vl);
                    suml_32_10 = __riscv_vwmacc_vx_i16m2(suml_32_10, a_ptr[b].qs[ssb + 22], qs_6, vl);
                    suml_32_11 = __riscv_vwmacc_vx_i16m2(suml_32_11, a_ptr[b].qs[ssb + 23], qs_6, vl);

                    src0_l = __riscv_vle8_v_u8m1(b_ptr[b].qlhl + sb * 384 + 64, vl);
                    ql_4bits = __riscv_vand_vx_u8m1(src0_l, 0x0F, vl);
                    qh_2bits = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(src0_h, 0x04, vl), 0x03, vl);
                    uqs_6 = __riscv_vor_vv_u8m1(ql_4bits, __riscv_vsll_vx_u8m1(qh_2bits, 0x04, vl), vl);
                    qs_6 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(uqs_6), 32, vl);
                    suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[b].qs[ssb + 24], qs_6, vl);
                    suml_32_1 = __riscv_vwmacc_vx_i16m2(suml_32_1, a_ptr[b].qs[ssb + 25], qs_6, vl);
                    suml_32_2 = __riscv_vwmacc_vx_i16m2(suml_32_2, a_ptr[b].qs[ssb + 26], qs_6, vl);
                    suml_32_3 = __riscv_vwmacc_vx_i16m2(suml_32_3, a_ptr[b].qs[ssb + 27], qs_6, vl);
                    suml_32_4 = __riscv_vwmacc_vx_i16m2(suml_32_4, a_ptr[b].qs[ssb + 28], qs_6, vl);
                    suml_32_5 = __riscv_vwmacc_vx_i16m2(suml_32_5, a_ptr[b].qs[ssb + 29], qs_6, vl);
                    suml_32_6 = __riscv_vwmacc_vx_i16m2(suml_32_6, a_ptr[b].qs[ssb + 30], qs_6, vl);
                    suml_32_7 = __riscv_vwmacc_vx_i16m2(suml_32_7, a_ptr[b].qs[ssb + 31], qs_6, vl);
                    suml_32_8 = __riscv_vwmacc_vx_i16m2(suml_32_8, a_ptr[b].qs[ssb + 32], qs_6, vl);
                    suml_32_9 = __riscv_vwmacc_vx_i16m2(suml_32_9, a_ptr[b].qs[ssb + 33], qs_6, vl);
                    suml_32_10 = __riscv_vwmacc_vx_i16m2(suml_32_10, a_ptr[b].qs[ssb + 34], qs_6, vl);
                    suml_32_11 = __riscv_vwmacc_vx_i16m2(suml_32_11, a_ptr[b].qs[ssb + 35], qs_6, vl);

                    ql_4bits = __riscv_vsrl_vx_u8m1(src0_l, 0x04, vl);
                    qh_2bits = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(src0_h, 0x06, vl), 0x03, vl);
                    uqs_6 = __riscv_vor_vv_u8m1(ql_4bits, __riscv_vsll_vx_u8m1(qh_2bits, 0x04, vl), vl);
                    qs_6 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(uqs_6), 32, vl);
                    suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[b].qs[ssb + 36], qs_6, vl);
                    suml_32_1 = __riscv_vwmacc_vx_i16m2(suml_32_1, a_ptr[b].qs[ssb + 37], qs_6, vl);
                    suml_32_2 = __riscv_vwmacc_vx_i16m2(suml_32_2, a_ptr[b].qs[ssb + 38], qs_6, vl);
                    suml_32_3 = __riscv_vwmacc_vx_i16m2(suml_32_3, a_ptr[b].qs[ssb + 39], qs_6, vl);
                    suml_32_4 = __riscv_vwmacc_vx_i16m2(suml_32_4, a_ptr[b].qs[ssb + 40], qs_6, vl);
                    suml_32_5 = __riscv_vwmacc_vx_i16m2(suml_32_5, a_ptr[b].qs[ssb + 41], qs_6, vl);
                    suml_32_6 = __riscv_vwmacc_vx_i16m2(suml_32_6, a_ptr[b].qs[ssb + 42], qs_6, vl);
                    suml_32_7 = __riscv_vwmacc_vx_i16m2(suml_32_7, a_ptr[b].qs[ssb + 43], qs_6, vl);
                    suml_32_8 = __riscv_vwmacc_vx_i16m2(suml_32_8, a_ptr[b].qs[ssb + 44], qs_6, vl);
                    suml_32_9 = __riscv_vwmacc_vx_i16m2(suml_32_9, a_ptr[b].qs[ssb + 45], qs_6, vl);
                    suml_32_10 = __riscv_vwmacc_vx_i16m2(suml_32_10, a_ptr[b].qs[ssb + 46], qs_6, vl);
                    suml_32_11 = __riscv_vwmacc_vx_i16m2(suml_32_11, a_ptr[b].qs[ssb + 47], qs_6, vl);

                    for (int z = 1; z < 4; z++) {
                        // 权重矩阵的0-3列和32-35列
                        const int sb_z = sb * 384 + z * 96;
                        src0_l = __riscv_vle8_v_u8m1(b_ptr[b].qlhl + sb_z, vl);
                        ql_4bits = __riscv_vand_vx_u8m1(src0_l, 0x0F, vl);
                        src0_h = __riscv_vle8_v_u8m1(b_ptr[b].qlhl + sb_z + 32, vl);
                        qh_2bits = __riscv_vand_vx_u8m1(src0_h, 0x03, vl);
                        uqs_6 = __riscv_vor_vv_u8m1(ql_4bits, __riscv_vsll_vx_u8m1(qh_2bits, 0x04, vl), vl);
                        qs_6 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(uqs_6), 32, vl);

                        const int64_t ssb = sb * 192 + z * 48;
                        suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[b].qs[ssb], qs_6, vl);
                        suml_32_1 = __riscv_vwmacc_vx_i16m2(suml_32_1, a_ptr[b].qs[ssb + 1], qs_6, vl);
                        suml_32_2 = __riscv_vwmacc_vx_i16m2(suml_32_2, a_ptr[b].qs[ssb + 2], qs_6, vl);
                        suml_32_3 = __riscv_vwmacc_vx_i16m2(suml_32_3, a_ptr[b].qs[ssb + 3], qs_6, vl);
                        suml_32_4 = __riscv_vwmacc_vx_i16m2(suml_32_4, a_ptr[b].qs[ssb + 4], qs_6, vl);
                        suml_32_5 = __riscv_vwmacc_vx_i16m2(suml_32_5, a_ptr[b].qs[ssb + 5], qs_6, vl);
                        suml_32_6 = __riscv_vwmacc_vx_i16m2(suml_32_6, a_ptr[b].qs[ssb + 6], qs_6, vl);
                        suml_32_7 = __riscv_vwmacc_vx_i16m2(suml_32_7, a_ptr[b].qs[ssb + 7], qs_6, vl);
                        suml_32_8 = __riscv_vwmacc_vx_i16m2(suml_32_8, a_ptr[b].qs[ssb + 8], qs_6, vl);
                        suml_32_9 = __riscv_vwmacc_vx_i16m2(suml_32_9, a_ptr[b].qs[ssb + 9], qs_6, vl);
                        suml_32_10 = __riscv_vwmacc_vx_i16m2(suml_32_10, a_ptr[b].qs[ssb + 10], qs_6, vl);
                        suml_32_11 = __riscv_vwmacc_vx_i16m2(suml_32_11, a_ptr[b].qs[ssb + 11], qs_6, vl);

                        ql_4bits = __riscv_vsrl_vx_u8m1(src0_l, 0x04, vl);
                        qh_2bits = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(src0_h, 0x02, vl), 0x03, vl);
                        uqs_6 = __riscv_vor_vv_u8m1(ql_4bits, __riscv_vsll_vx_u8m1(qh_2bits, 0x04, vl), vl);
                        qs_6 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(uqs_6), 32, vl);
                        suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[b].qs[ssb + 12], qs_6, vl);
                        suml_32_1 = __riscv_vwmacc_vx_i16m2(suml_32_1, a_ptr[b].qs[ssb + 13], qs_6, vl);
                        suml_32_2 = __riscv_vwmacc_vx_i16m2(suml_32_2, a_ptr[b].qs[ssb + 14], qs_6, vl);
                        suml_32_3 = __riscv_vwmacc_vx_i16m2(suml_32_3, a_ptr[b].qs[ssb + 15], qs_6, vl);
                        suml_32_4 = __riscv_vwmacc_vx_i16m2(suml_32_4, a_ptr[b].qs[ssb + 16], qs_6, vl);
                        suml_32_5 = __riscv_vwmacc_vx_i16m2(suml_32_5, a_ptr[b].qs[ssb + 17], qs_6, vl);
                        suml_32_6 = __riscv_vwmacc_vx_i16m2(suml_32_6, a_ptr[b].qs[ssb + 18], qs_6, vl);
                        suml_32_7 = __riscv_vwmacc_vx_i16m2(suml_32_7, a_ptr[b].qs[ssb + 19], qs_6, vl);
                        suml_32_8 = __riscv_vwmacc_vx_i16m2(suml_32_8, a_ptr[b].qs[ssb + 20], qs_6, vl);
                        suml_32_9 = __riscv_vwmacc_vx_i16m2(suml_32_9, a_ptr[b].qs[ssb + 21], qs_6, vl);
                        suml_32_10 = __riscv_vwmacc_vx_i16m2(suml_32_10, a_ptr[b].qs[ssb + 22], qs_6, vl);
                        suml_32_11 = __riscv_vwmacc_vx_i16m2(suml_32_11, a_ptr[b].qs[ssb + 23], qs_6, vl);

                        src0_l = __riscv_vle8_v_u8m1(b_ptr[b].qlhl + sb_z + 64, vl);
                        ql_4bits = __riscv_vand_vx_u8m1(src0_l, 0x0F, vl);
                        qh_2bits = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(src0_h, 0x04, vl), 0x03, vl);
                        uqs_6 = __riscv_vor_vv_u8m1(ql_4bits, __riscv_vsll_vx_u8m1(qh_2bits, 0x04, vl), vl);
                        qs_6 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(uqs_6), 32, vl);   //
                        suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[b].qs[ssb + 24], qs_6, vl);
                        suml_32_1 = __riscv_vwmacc_vx_i16m2(suml_32_1, a_ptr[b].qs[ssb + 25], qs_6, vl);
                        suml_32_2 = __riscv_vwmacc_vx_i16m2(suml_32_2, a_ptr[b].qs[ssb + 26], qs_6, vl);
                        suml_32_3 = __riscv_vwmacc_vx_i16m2(suml_32_3, a_ptr[b].qs[ssb + 27], qs_6, vl);
                        suml_32_4 = __riscv_vwmacc_vx_i16m2(suml_32_4, a_ptr[b].qs[ssb + 28], qs_6, vl);
                        suml_32_5 = __riscv_vwmacc_vx_i16m2(suml_32_5, a_ptr[b].qs[ssb + 29], qs_6, vl);
                        suml_32_6 = __riscv_vwmacc_vx_i16m2(suml_32_6, a_ptr[b].qs[ssb + 30], qs_6, vl);
                        suml_32_7 = __riscv_vwmacc_vx_i16m2(suml_32_7, a_ptr[b].qs[ssb + 31], qs_6, vl);
                        suml_32_8 = __riscv_vwmacc_vx_i16m2(suml_32_8, a_ptr[b].qs[ssb + 32], qs_6, vl);
                        suml_32_9 = __riscv_vwmacc_vx_i16m2(suml_32_9, a_ptr[b].qs[ssb + 33], qs_6, vl);
                        suml_32_10 = __riscv_vwmacc_vx_i16m2(suml_32_10, a_ptr[b].qs[ssb + 34], qs_6, vl);
                        suml_32_11 = __riscv_vwmacc_vx_i16m2(suml_32_11, a_ptr[b].qs[ssb + 35], qs_6, vl);

                        ql_4bits = __riscv_vsrl_vx_u8m1(src0_l, 0x04, vl);
                        qh_2bits = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(src0_h, 0x06, vl), 0x03, vl);
                        uqs_6 = __riscv_vor_vv_u8m1(ql_4bits, __riscv_vsll_vx_u8m1(qh_2bits, 0x04, vl), vl);
                        qs_6 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(uqs_6), 32, vl);
                        suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[b].qs[ssb + 36], qs_6, vl);
                        suml_32_1 = __riscv_vwmacc_vx_i16m2(suml_32_1, a_ptr[b].qs[ssb + 37], qs_6, vl);
                        suml_32_2 = __riscv_vwmacc_vx_i16m2(suml_32_2, a_ptr[b].qs[ssb + 38], qs_6, vl);
                        suml_32_3 = __riscv_vwmacc_vx_i16m2(suml_32_3, a_ptr[b].qs[ssb + 39], qs_6, vl);
                        suml_32_4 = __riscv_vwmacc_vx_i16m2(suml_32_4, a_ptr[b].qs[ssb + 40], qs_6, vl);
                        suml_32_5 = __riscv_vwmacc_vx_i16m2(suml_32_5, a_ptr[b].qs[ssb + 41], qs_6, vl);
                        suml_32_6 = __riscv_vwmacc_vx_i16m2(suml_32_6, a_ptr[b].qs[ssb + 42], qs_6, vl);
                        suml_32_7 = __riscv_vwmacc_vx_i16m2(suml_32_7, a_ptr[b].qs[ssb + 43], qs_6, vl);
                        suml_32_8 = __riscv_vwmacc_vx_i16m2(suml_32_8, a_ptr[b].qs[ssb + 44], qs_6, vl);
                        suml_32_9 = __riscv_vwmacc_vx_i16m2(suml_32_9, a_ptr[b].qs[ssb + 45], qs_6, vl);
                        suml_32_10 = __riscv_vwmacc_vx_i16m2(suml_32_10, a_ptr[b].qs[ssb + 46], qs_6, vl);
                        suml_32_11 = __riscv_vwmacc_vx_i16m2(suml_32_11, a_ptr[b].qs[ssb + 47], qs_6, vl);
                    }

                    vint16m2_t vec_scales_32 = __riscv_vsext_vf2_i16m2(__riscv_vle8_v_i8m1(b_ptr[b].scales + sb * nr, vl), vl);
                    __riscv_vse32_v_i32m4(sum_block,__riscv_vwmacc_vv_i32m4(__riscv_vle32_v_i32m4(sum_block, vl), suml_32_0, vec_scales_32, vl),vl);
                    __riscv_vse32_v_i32m4(sum_block + nr,__riscv_vwmacc_vv_i32m4(__riscv_vle32_v_i32m4(sum_block + nr, vl), suml_32_1, vec_scales_32, vl),vl);
                    __riscv_vse32_v_i32m4(sum_block + nr * 2, __riscv_vwmacc_vv_i32m4(__riscv_vle32_v_i32m4(sum_block + nr * 2, vl), suml_32_2, vec_scales_32, vl), vl);
                    __riscv_vse32_v_i32m4(sum_block + nr * 3, __riscv_vwmacc_vv_i32m4(__riscv_vle32_v_i32m4(sum_block + nr * 3, vl), suml_32_3, vec_scales_32, vl), vl);
                    __riscv_vse32_v_i32m4(sum_block + nr * 4,__riscv_vwmacc_vv_i32m4(__riscv_vle32_v_i32m4(sum_block + nr * 4, vl), suml_32_4, vec_scales_32, vl), vl);
                    __riscv_vse32_v_i32m4(sum_block + nr * 5,__riscv_vwmacc_vv_i32m4(__riscv_vle32_v_i32m4(sum_block + nr * 5, vl), suml_32_5, vec_scales_32, vl), vl);
                    __riscv_vse32_v_i32m4(sum_block + nr * 6,__riscv_vwmacc_vv_i32m4(__riscv_vle32_v_i32m4(sum_block + nr * 6, vl), suml_32_6, vec_scales_32, vl), vl);
                    __riscv_vse32_v_i32m4(sum_block + nr * 7, __riscv_vwmacc_vv_i32m4(__riscv_vle32_v_i32m4(sum_block + nr * 7, vl), suml_32_7, vec_scales_32, vl), vl);
                    __riscv_vse32_v_i32m4(sum_block + nr * 8, __riscv_vwmacc_vv_i32m4(__riscv_vle32_v_i32m4(sum_block + nr * 8, vl), suml_32_8, vec_scales_32, vl), vl);
                    __riscv_vse32_v_i32m4(sum_block + nr * 9, __riscv_vwmacc_vv_i32m4(__riscv_vle32_v_i32m4(sum_block + nr * 9, vl), suml_32_9, vec_scales_32, vl), vl);
                    __riscv_vse32_v_i32m4(sum_block + nr * 10, __riscv_vwmacc_vv_i32m4(__riscv_vle32_v_i32m4(sum_block + nr * 10, vl), suml_32_10, vec_scales_32, vl), vl);
                    __riscv_vse32_v_i32m4(sum_block + nr * 11, __riscv_vwmacc_vv_i32m4(__riscv_vle32_v_i32m4(sum_block + nr * 11, vl), suml_32_11, vec_scales_32, vl), vl);
                }
                vfloat32m4_t src0_d = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vreinterpret_v_u16m2_f16m2(__riscv_vle16_v_u16m2(b_ptr[b].d, vl)), vl);

                __riscv_vse32_v_f32m4(sum_row, __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row, vl), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_block, vl), vl), __riscv_vfmul_vf_f32m4(src0_d, a_ptr[b].d[0], vl), vl), vl);
                __riscv_vse32_v_f32m4(sum_row + nr, __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr, vl), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_block + nr, vl), vl), __riscv_vfmul_vf_f32m4(src0_d, a_ptr[b].d[1], vl), vl), vl);
                __riscv_vse32_v_f32m4(sum_row + nr * 2, __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr * 2, vl), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_block + nr * 2, vl), vl), __riscv_vfmul_vf_f32m4(src0_d, a_ptr[b].d[2], vl), vl), vl);
                __riscv_vse32_v_f32m4(sum_row + nr * 3, __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr * 3, vl), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_block + nr * 3, vl), vl), __riscv_vfmul_vf_f32m4(src0_d, a_ptr[b].d[3], vl), vl), vl);
                __riscv_vse32_v_f32m4(sum_row + nr * 4, __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr * 4, vl), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_block + nr * 4, vl), vl), __riscv_vfmul_vf_f32m4(src0_d, a_ptr[b].d[4], vl), vl), vl);
                __riscv_vse32_v_f32m4(sum_row + nr * 5, __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr * 5, vl), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_block + nr * 5, vl), vl), __riscv_vfmul_vf_f32m4(src0_d, a_ptr[b].d[5], vl), vl), vl);
                __riscv_vse32_v_f32m4(sum_row + nr * 6, __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr * 6, vl), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_block + nr * 6, vl), vl), __riscv_vfmul_vf_f32m4(src0_d, a_ptr[b].d[6], vl), vl), vl);
                __riscv_vse32_v_f32m4(sum_row + nr * 7, __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr * 7, vl), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_block + nr * 7, vl), vl), __riscv_vfmul_vf_f32m4(src0_d, a_ptr[b].d[7], vl), vl), vl);
                __riscv_vse32_v_f32m4(sum_row + nr * 8, __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr * 8, vl), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_block + nr * 8, vl), vl), __riscv_vfmul_vf_f32m4(src0_d, a_ptr[b].d[8], vl), vl), vl);
                __riscv_vse32_v_f32m4(sum_row + nr * 9, __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr * 9, vl), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_block + nr * 9, vl), vl), __riscv_vfmul_vf_f32m4(src0_d, a_ptr[b].d[9], vl), vl), vl);
                __riscv_vse32_v_f32m4(sum_row + nr * 10, __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr * 10, vl), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_block + nr * 10, vl), vl),  __riscv_vfmul_vf_f32m4(src0_d, a_ptr[b].d[10], vl), vl), vl);
                __riscv_vse32_v_f32m4(sum_row + nr * 11, __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr * 11, vl), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_block + nr * 11, vl), vl),  __riscv_vfmul_vf_f32m4(src0_d, a_ptr[b].d[11], vl), vl), vl);
            }
            __riscv_vse32_v_f32m4((float *)(s + y * 12 * bs + x * nr), __riscv_vle32_v_f32m4(sum_row, vl), vl);
            __riscv_vse32_v_f32m4((float *)(s + ((y * 12 + 1) * bs + x * nr)), __riscv_vle32_v_f32m4(sum_row + nr, vl), vl);
            __riscv_vse32_v_f32m4((float *)(s + ((y * 12 + 2) * bs + x * nr)), __riscv_vle32_v_f32m4(sum_row + nr * 2, vl), vl);
            __riscv_vse32_v_f32m4((float *)(s + ((y * 12 + 3) * bs + x * nr)), __riscv_vle32_v_f32m4(sum_row + nr * 3, vl), vl);
            __riscv_vse32_v_f32m4((float *)(s + ((y * 12 + 4) * bs + x * nr)), __riscv_vle32_v_f32m4(sum_row + nr * 4, vl), vl);
            __riscv_vse32_v_f32m4((float *)(s + ((y * 12 + 5) * bs + x * nr)), __riscv_vle32_v_f32m4(sum_row + nr * 5, vl), vl);
            __riscv_vse32_v_f32m4((float *)(s + ((y * 12 + 6) * bs + x * nr)), __riscv_vle32_v_f32m4(sum_row + nr * 6, vl), vl);
            __riscv_vse32_v_f32m4((float *)(s + ((y * 12 + 7) * bs + x * nr)), __riscv_vle32_v_f32m4(sum_row + nr * 7, vl), vl);
            __riscv_vse32_v_f32m4((float *)(s + ((y * 12 + 8) * bs + x * nr)), __riscv_vle32_v_f32m4(sum_row + nr * 8, vl), vl);
            __riscv_vse32_v_f32m4((float *)(s + ((y * 12 + 9) * bs + x * nr)), __riscv_vle32_v_f32m4(sum_row + nr * 9, vl), vl);
            __riscv_vse32_v_f32m4((float *)(s + ((y * 12 + 10) * bs + x * nr)), __riscv_vle32_v_f32m4(sum_row + nr * 10, vl), vl);
            __riscv_vse32_v_f32m4((float *)(s + ((y * 12 + 11) * bs + x * nr)), __riscv_vle32_v_f32m4(sum_row + nr * 11, vl), vl);
        }

    }

    const block_q8_Kx4 *a_ptr_start1 = (const block_q8_Kx4 *)vy;

    for (int y = anr / 4; y < m / 4; y++) { // M

        const block_q8_Kx4 *a_ptr = a_ptr_start1 + (y * nb);
        for (int64_t x = 0; x < n / nr; x++) { // N

            const block_q6_Kx32 *b_ptr = b_ptr_start + (x * nb);

            // 行累加和
            float sum_row[nr * 4] = {0.0};

            for (int64_t b = 0; b < nb; b++) { // K
                int32_t sum_block[nr * 4] = {0};    //可以直接换成

                // Loop to iterate over the eight sub blocks of a super block - two sub blocks are processed per
                // iteration
                for (int sb = 0; sb < QK_K / 16; sb++) { // QK_K / nr * 2=4,一次处理src0的64列  //
                    vint16m2_t suml_32_0 = __riscv_vmv_v_x_i16m2(0, vl);
                    vint16m2_t suml_32_1 = __riscv_vmv_v_x_i16m2(0, vl);
                    vint16m2_t suml_32_2 = __riscv_vmv_v_x_i16m2(0, vl);
                    vint16m2_t suml_32_3 = __riscv_vmv_v_x_i16m2(0, vl);
                    for (int z = 0; z < 4; z++) {
                        // 权重矩阵的0-3列和32-35列
                        const int sb_z = sb * 384 + z * 96;
                        vuint8m1_t src0_l = __riscv_vle8_v_u8m1(b_ptr[b].qlhl + sb_z, vl);
                        vuint8m1_t ql_4bits = __riscv_vand_vx_u8m1(src0_l, 0x0F, vl);
                        vuint8m1_t src0_h = __riscv_vle8_v_u8m1(b_ptr[b].qlhl + sb_z + 32, vl);
                        vuint8m1_t qh_2bits = __riscv_vand_vx_u8m1(src0_h, 0x03, vl);
                        vuint8m1_t uqs_6 = __riscv_vor_vv_u8m1(ql_4bits, __riscv_vsll_vx_u8m1(qh_2bits, 0x04, vl), vl);
                        vint8m1_t qs_6 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(uqs_6), 32, vl);

                        const int64_t ssb = sb * nr * 2 + z * 16;
                        suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[b].qs[ssb], qs_6, vl);
                        suml_32_1 = __riscv_vwmacc_vx_i16m2(suml_32_1, a_ptr[b].qs[ssb + 1], qs_6, vl);
                        suml_32_2 = __riscv_vwmacc_vx_i16m2(suml_32_2, a_ptr[b].qs[ssb + 2], qs_6, vl);
                        suml_32_3 = __riscv_vwmacc_vx_i16m2(suml_32_3, a_ptr[b].qs[ssb + 3], qs_6, vl);

                        ql_4bits = __riscv_vsrl_vx_u8m1(src0_l, 0x04, vl);
                        qh_2bits = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(src0_h, 0x02, vl), 0x03, vl);
                        uqs_6 = __riscv_vor_vv_u8m1(ql_4bits, __riscv_vsll_vx_u8m1(qh_2bits, 0x04, vl), vl);
                        qs_6 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(uqs_6), 32, vl);
                        suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[b].qs[ssb + 4], qs_6, vl);
                        suml_32_1 = __riscv_vwmacc_vx_i16m2(suml_32_1, a_ptr[b].qs[ssb + 5], qs_6, vl);
                        suml_32_2 = __riscv_vwmacc_vx_i16m2(suml_32_2, a_ptr[b].qs[ssb + 6], qs_6, vl);
                        suml_32_3 = __riscv_vwmacc_vx_i16m2(suml_32_3, a_ptr[b].qs[ssb + 7], qs_6, vl);

                        src0_l = __riscv_vle8_v_u8m1(b_ptr[b].qlhl + sb_z + 64, vl);
                        ql_4bits = __riscv_vand_vx_u8m1(src0_l, 0x0F, vl);
                        qh_2bits = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(src0_h, 0x04, vl), 0x03, vl);
                        uqs_6 = __riscv_vor_vv_u8m1(ql_4bits, __riscv_vsll_vx_u8m1(qh_2bits, 0x04, vl), vl);
                        qs_6 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(uqs_6), 32, vl);
                        suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[b].qs[ssb + 8], qs_6, vl);
                        suml_32_1 = __riscv_vwmacc_vx_i16m2(suml_32_1, a_ptr[b].qs[ssb + 9], qs_6, vl);
                        suml_32_2 = __riscv_vwmacc_vx_i16m2(suml_32_2, a_ptr[b].qs[ssb + 10], qs_6, vl);
                        suml_32_3 = __riscv_vwmacc_vx_i16m2(suml_32_3, a_ptr[b].qs[ssb + 11], qs_6, vl);

                        ql_4bits = __riscv_vsrl_vx_u8m1(src0_l, 0x04, vl);
                        qh_2bits = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(src0_h, 0x06, vl), 0x03, vl);
                        uqs_6 = __riscv_vor_vv_u8m1(ql_4bits, __riscv_vsll_vx_u8m1(qh_2bits, 0x04, vl), vl);
                        qs_6 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(uqs_6), 32, vl);
                        suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[b].qs[ssb + 12], qs_6, vl);
                        suml_32_1 = __riscv_vwmacc_vx_i16m2(suml_32_1, a_ptr[b].qs[ssb + 13], qs_6, vl);
                        suml_32_2 = __riscv_vwmacc_vx_i16m2(suml_32_2, a_ptr[b].qs[ssb + 14], qs_6, vl);
                        suml_32_3 = __riscv_vwmacc_vx_i16m2(suml_32_3, a_ptr[b].qs[ssb + 15], qs_6, vl);
                    }

                    // 处理scales和mins

                    vint16m2_t vec_scales_32 = __riscv_vsext_vf2_i16m2(__riscv_vle8_v_i8m1(b_ptr[b].scales + sb * nr, vl), vl);

                    __riscv_vse32_v_i32m4(sum_block, __riscv_vwmacc_vv_i32m4(__riscv_vle32_v_i32m4(sum_block, vl), suml_32_0, vec_scales_32, vl), vl);
                    __riscv_vse32_v_i32m4(sum_block + nr, __riscv_vwmacc_vv_i32m4(__riscv_vle32_v_i32m4(sum_block + nr, vl), suml_32_1, vec_scales_32, vl), vl);
                    __riscv_vse32_v_i32m4(sum_block + nr * 2, __riscv_vwmacc_vv_i32m4(__riscv_vle32_v_i32m4(sum_block + nr * 2, vl), suml_32_2, vec_scales_32, vl), vl);
                    __riscv_vse32_v_i32m4(sum_block + nr * 3, __riscv_vwmacc_vv_i32m4(__riscv_vle32_v_i32m4(sum_block + nr * 3, vl), suml_32_3, vec_scales_32, vl), vl);

                }

                vfloat32m4_t src0_d = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vreinterpret_v_u16m2_f16m2(__riscv_vle16_v_u16m2(b_ptr[b].d, vl)), vl);

                __riscv_vse32_v_f32m4(sum_row, __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row, vl), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_block, vl), vl), __riscv_vfmul_vf_f32m4(src0_d, a_ptr[b].d[0], vl), vl), vl);
                __riscv_vse32_v_f32m4(sum_row + nr, __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr, vl), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_block + nr, vl), vl), __riscv_vfmul_vf_f32m4(src0_d, a_ptr[b].d[1], vl), vl), vl);
                __riscv_vse32_v_f32m4(sum_row + nr * 2, __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr * 2, vl),  __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_block + nr * 2, vl), vl), __riscv_vfmul_vf_f32m4(src0_d, a_ptr[b].d[2], vl), vl),  vl);
                __riscv_vse32_v_f32m4(sum_row + nr * 3, __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nr * 3, vl), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_block + nr * 3, vl), vl), __riscv_vfmul_vf_f32m4(src0_d, a_ptr[b].d[3], vl), vl), vl);

            }
            __riscv_vse32_v_f32m4((float *)(s + ((y * 4 + 0) * bs + x * nr)), __riscv_vle32_v_f32m4(sum_row, vl),  vl);
            __riscv_vse32_v_f32m4((float *)(s + ((y * 4 + 1) * bs + x * nr)), __riscv_vle32_v_f32m4(sum_row + nr, vl), vl);
            __riscv_vse32_v_f32m4((float *)(s + ((y * 4 + 2) * bs + x * nr)), __riscv_vle32_v_f32m4(sum_row + nr * 2, vl), vl);
            __riscv_vse32_v_f32m4((float *)(s + ((y * 4 + 3) * bs + x * nr)), __riscv_vle32_v_f32m4(sum_row + nr * 3, vl), vl);
        }
    }

}