//基于gemm1和gemm4修改， 12 * 32的内核,正确的版本
//边界处理使用4*32计算，好处是核心计算可以展开很大，缺点是中间结果需要占用数组，增加向量store和load操作。
void ggml_gemm_q4_0_8x32_q8_0(int n, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx,
                              const void *GGML_RESTRICT vy, int nr, int nc) {
    const int nb = n / QK4_0;
    const int ncols_interleaved = 8;
    const int blocklen = 8;

    assert(n % QK4_0 == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
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
    int anr = nr - nr % mrr;
    const int mnr = mrr * nrr;

    for (int y = 0; y < anr / mrr; y++) {

        const block_q8_0x12 *a_ptr = a_ptr_start + (y * nb);
        for (int x = 0; x < nc / nrr; x++) { // N

            const block_q4_0x32 *b_ptr = b_ptr_start + (x * nb);

            float sum_row[mnr] = {0.0};

            // 行累加和
            for (int64_t b = 0; b < nb; b++) { // K

                vint8m1_t src0_0 = __riscv_vle8_v_i8m1(b_ptr[b].qs, vl);
                vint8m1_t low4bits_0 = __riscv_vsra_vx_i8m1(__riscv_vsll_vx_i8m1(src0_0, 4, vl), 4, vl);
                vint16m2_t suml_32_0 = __riscv_vwmul_vx_i16m2(low4bits_0, a_ptr[b].qs[0], vl);
                vint16m2_t suml_32_1 = __riscv_vwmul_vx_i16m2(low4bits_0, a_ptr[b].qs[1], vl);
                vint16m2_t suml_32_2 = __riscv_vwmul_vx_i16m2(low4bits_0, a_ptr[b].qs[2], vl);
                vint16m2_t suml_32_3 = __riscv_vwmul_vx_i16m2(low4bits_0, a_ptr[b].qs[3], vl);
                vint16m2_t suml_32_4 = __riscv_vwmul_vx_i16m2(low4bits_0, a_ptr[b].qs[4], vl);
                vint16m2_t suml_32_5 = __riscv_vwmul_vx_i16m2(low4bits_0, a_ptr[b].qs[5], vl);
                vint16m2_t suml_32_6 = __riscv_vwmul_vx_i16m2(low4bits_0, a_ptr[b].qs[6], vl);
                vint16m2_t suml_32_7 = __riscv_vwmul_vx_i16m2(low4bits_0, a_ptr[b].qs[7], vl);
                vint16m2_t suml_32_8 = __riscv_vwmul_vx_i16m2(low4bits_0, a_ptr[b].qs[8], vl);
                vint16m2_t suml_32_9 = __riscv_vwmul_vx_i16m2(low4bits_0, a_ptr[b].qs[9], vl);
                vint16m2_t suml_32_10 = __riscv_vwmul_vx_i16m2(low4bits_0, a_ptr[b].qs[10], vl);
                vint16m2_t suml_32_11 = __riscv_vwmul_vx_i16m2(low4bits_0, a_ptr[b].qs[11], vl);
                low4bits_0 = __riscv_vsra_vx_i8m1(src0_0, 4, vl);
                suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[b].qs[12], low4bits_0, vl);
                suml_32_1 = __riscv_vwmacc_vx_i16m2(suml_32_1, a_ptr[b].qs[13], low4bits_0, vl);
                suml_32_2 = __riscv_vwmacc_vx_i16m2(suml_32_2, a_ptr[b].qs[14], low4bits_0, vl);
                suml_32_3 = __riscv_vwmacc_vx_i16m2(suml_32_3, a_ptr[b].qs[15], low4bits_0, vl);
                suml_32_4 = __riscv_vwmacc_vx_i16m2(suml_32_4, a_ptr[b].qs[16], low4bits_0, vl);
                suml_32_5 = __riscv_vwmacc_vx_i16m2(suml_32_5, a_ptr[b].qs[17], low4bits_0, vl);
                suml_32_6 = __riscv_vwmacc_vx_i16m2(suml_32_6, a_ptr[b].qs[18], low4bits_0, vl);
                suml_32_7 = __riscv_vwmacc_vx_i16m2(suml_32_7, a_ptr[b].qs[19], low4bits_0, vl);
                suml_32_8 = __riscv_vwmacc_vx_i16m2(suml_32_8, a_ptr[b].qs[20], low4bits_0, vl);
                suml_32_9 = __riscv_vwmacc_vx_i16m2(suml_32_9, a_ptr[b].qs[21], low4bits_0, vl);
                suml_32_10 = __riscv_vwmacc_vx_i16m2(suml_32_10, a_ptr[b].qs[22], low4bits_0, vl);
                suml_32_11 = __riscv_vwmacc_vx_i16m2(suml_32_11, a_ptr[b].qs[23], low4bits_0, vl);
                for (int z = 1; z < 8; z++) {
                    // 权重矩阵的0-3列和32-35列
                    vint8m1_t src0_0 = __riscv_vle8_v_i8m1(b_ptr[b].qs + z * nrr, vl);
                    vint8m1_t low4bits_0 = __riscv_vsra_vx_i8m1(__riscv_vsll_vx_i8m1(src0_0, 4, vl), 4, vl);
                    const int64_t ssb = z * 24;
                    suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[b].qs[ssb], low4bits_0, vl);
                    suml_32_1 = __riscv_vwmacc_vx_i16m2(suml_32_1, a_ptr[b].qs[ssb + 1], low4bits_0, vl);
                    suml_32_2 = __riscv_vwmacc_vx_i16m2(suml_32_2, a_ptr[b].qs[ssb + 2], low4bits_0, vl);
                    suml_32_3 = __riscv_vwmacc_vx_i16m2(suml_32_3, a_ptr[b].qs[ssb + 3], low4bits_0, vl);
                    suml_32_4 = __riscv_vwmacc_vx_i16m2(suml_32_4, a_ptr[b].qs[ssb + 4], low4bits_0, vl);
                    suml_32_5 = __riscv_vwmacc_vx_i16m2(suml_32_5, a_ptr[b].qs[ssb + 5], low4bits_0, vl);
                    suml_32_6 = __riscv_vwmacc_vx_i16m2(suml_32_6, a_ptr[b].qs[ssb + 6], low4bits_0, vl);
                    suml_32_7 = __riscv_vwmacc_vx_i16m2(suml_32_7, a_ptr[b].qs[ssb + 7], low4bits_0, vl);
                    suml_32_8 = __riscv_vwmacc_vx_i16m2(suml_32_8, a_ptr[b].qs[ssb + 8], low4bits_0, vl);
                    suml_32_9 = __riscv_vwmacc_vx_i16m2(suml_32_9, a_ptr[b].qs[ssb + 9], low4bits_0, vl);
                    suml_32_10 = __riscv_vwmacc_vx_i16m2(suml_32_10, a_ptr[b].qs[ssb + 10], low4bits_0, vl);
                    suml_32_11 = __riscv_vwmacc_vx_i16m2(suml_32_11, a_ptr[b].qs[ssb + 11], low4bits_0, vl);
                    low4bits_0 = __riscv_vsra_vx_i8m1(src0_0, 4, vl);
                    suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[b].qs[ssb + 12], low4bits_0, vl);
                    suml_32_1 = __riscv_vwmacc_vx_i16m2(suml_32_1, a_ptr[b].qs[ssb + 13], low4bits_0, vl);
                    suml_32_2 = __riscv_vwmacc_vx_i16m2(suml_32_2, a_ptr[b].qs[ssb + 14], low4bits_0, vl);
                    suml_32_3 = __riscv_vwmacc_vx_i16m2(suml_32_3, a_ptr[b].qs[ssb + 15], low4bits_0, vl);
                    suml_32_4 = __riscv_vwmacc_vx_i16m2(suml_32_4, a_ptr[b].qs[ssb + 16], low4bits_0, vl);
                    suml_32_5 = __riscv_vwmacc_vx_i16m2(suml_32_5, a_ptr[b].qs[ssb + 17], low4bits_0, vl);
                    suml_32_6 = __riscv_vwmacc_vx_i16m2(suml_32_6, a_ptr[b].qs[ssb + 18], low4bits_0, vl);
                    suml_32_7 = __riscv_vwmacc_vx_i16m2(suml_32_7, a_ptr[b].qs[ssb + 19], low4bits_0, vl);
                    suml_32_8 = __riscv_vwmacc_vx_i16m2(suml_32_8, a_ptr[b].qs[ssb + 20], low4bits_0, vl);
                    suml_32_9 = __riscv_vwmacc_vx_i16m2(suml_32_9, a_ptr[b].qs[ssb + 21], low4bits_0, vl);
                    suml_32_10 = __riscv_vwmacc_vx_i16m2(suml_32_10, a_ptr[b].qs[ssb + 22], low4bits_0, vl);
                    suml_32_11 = __riscv_vwmacc_vx_i16m2(suml_32_11, a_ptr[b].qs[ssb + 23], low4bits_0, vl);
                }

                int32_t temp[mnr] = {0};

                __riscv_vse32_v_i32m4(temp, __riscv_vwadd_wv_i32m4(__riscv_vle32_v_i32m4(temp, vl), suml_32_0, vl), vl);
                __riscv_vse32_v_i32m4(temp+nrr, __riscv_vwadd_wv_i32m4(__riscv_vle32_v_i32m4(temp+nrr, vl), suml_32_1, vl), vl);
                __riscv_vse32_v_i32m4(temp+nrr*2, __riscv_vwadd_wv_i32m4(__riscv_vle32_v_i32m4(temp+nrr*2, vl), suml_32_2, vl), vl);
                __riscv_vse32_v_i32m4(temp+nrr*3, __riscv_vwadd_wv_i32m4(__riscv_vle32_v_i32m4(temp+nrr*3, vl), suml_32_3, vl), vl);
                __riscv_vse32_v_i32m4(temp+nrr*4, __riscv_vwadd_wv_i32m4(__riscv_vle32_v_i32m4(temp+nrr*4, vl), suml_32_4, vl), vl);
                __riscv_vse32_v_i32m4(temp+nrr*5, __riscv_vwadd_wv_i32m4(__riscv_vle32_v_i32m4(temp+nrr*5, vl), suml_32_5, vl), vl);
                __riscv_vse32_v_i32m4(temp+nrr*6, __riscv_vwadd_wv_i32m4(__riscv_vle32_v_i32m4(temp+nrr*6, vl), suml_32_6, vl), vl);
                __riscv_vse32_v_i32m4(temp+nrr*7, __riscv_vwadd_wv_i32m4(__riscv_vle32_v_i32m4(temp+nrr*7, vl), suml_32_7, vl), vl);
                __riscv_vse32_v_i32m4(temp+nrr*8, __riscv_vwadd_wv_i32m4(__riscv_vle32_v_i32m4(temp+nrr*8, vl), suml_32_8, vl), vl);
                __riscv_vse32_v_i32m4(temp+nrr*9, __riscv_vwadd_wv_i32m4(__riscv_vle32_v_i32m4(temp+nrr*9, vl), suml_32_9, vl), vl);
                __riscv_vse32_v_i32m4(temp+nrr*10, __riscv_vwadd_wv_i32m4(__riscv_vle32_v_i32m4(temp+nrr*10, vl), suml_32_10, vl), vl);
                __riscv_vse32_v_i32m4(temp+nrr*11, __riscv_vwadd_wv_i32m4(__riscv_vle32_v_i32m4(temp+nrr*11, vl), suml_32_11, vl), vl);

                src0_0 = __riscv_vle8_v_i8m1(b_ptr[b].qs + 8 * nrr, vl);
                low4bits_0 = __riscv_vsra_vx_i8m1(__riscv_vsll_vx_i8m1(src0_0, 4, vl), 4, vl);
                suml_32_0 = __riscv_vwmul_vx_i16m2(low4bits_0, a_ptr[b].qs[192], vl);
                suml_32_1 = __riscv_vwmul_vx_i16m2(low4bits_0, a_ptr[b].qs[193], vl);
                suml_32_2 = __riscv_vwmul_vx_i16m2(low4bits_0, a_ptr[b].qs[194], vl);
                suml_32_3 = __riscv_vwmul_vx_i16m2(low4bits_0, a_ptr[b].qs[195], vl);
                suml_32_4 = __riscv_vwmul_vx_i16m2(low4bits_0, a_ptr[b].qs[196], vl);
                suml_32_5 = __riscv_vwmul_vx_i16m2(low4bits_0, a_ptr[b].qs[197], vl);
                suml_32_6 = __riscv_vwmul_vx_i16m2(low4bits_0, a_ptr[b].qs[198], vl);
                suml_32_7 = __riscv_vwmul_vx_i16m2(low4bits_0, a_ptr[b].qs[199], vl);
                suml_32_8 = __riscv_vwmul_vx_i16m2(low4bits_0, a_ptr[b].qs[200], vl);
                suml_32_9 = __riscv_vwmul_vx_i16m2(low4bits_0, a_ptr[b].qs[201], vl);
                suml_32_10 = __riscv_vwmul_vx_i16m2(low4bits_0, a_ptr[b].qs[202], vl);
                suml_32_11 = __riscv_vwmul_vx_i16m2(low4bits_0, a_ptr[b].qs[203], vl);
                low4bits_0 = __riscv_vsra_vx_i8m1(src0_0, 4, vl);
                suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[b].qs[204], low4bits_0, vl);
                suml_32_1 = __riscv_vwmacc_vx_i16m2(suml_32_1, a_ptr[b].qs[205], low4bits_0, vl);
                suml_32_2 = __riscv_vwmacc_vx_i16m2(suml_32_2, a_ptr[b].qs[206], low4bits_0, vl);
                suml_32_3 = __riscv_vwmacc_vx_i16m2(suml_32_3, a_ptr[b].qs[207], low4bits_0, vl);
                suml_32_4 = __riscv_vwmacc_vx_i16m2(suml_32_4, a_ptr[b].qs[208], low4bits_0, vl);
                suml_32_5 = __riscv_vwmacc_vx_i16m2(suml_32_5, a_ptr[b].qs[209], low4bits_0, vl);
                suml_32_6 = __riscv_vwmacc_vx_i16m2(suml_32_6, a_ptr[b].qs[210], low4bits_0, vl);
                suml_32_7 = __riscv_vwmacc_vx_i16m2(suml_32_7, a_ptr[b].qs[211], low4bits_0, vl);
                suml_32_8 = __riscv_vwmacc_vx_i16m2(suml_32_8, a_ptr[b].qs[212], low4bits_0, vl);
                suml_32_9 = __riscv_vwmacc_vx_i16m2(suml_32_9, a_ptr[b].qs[213], low4bits_0, vl);
                suml_32_10 = __riscv_vwmacc_vx_i16m2(suml_32_10, a_ptr[b].qs[214], low4bits_0, vl);
                suml_32_11 = __riscv_vwmacc_vx_i16m2(suml_32_11, a_ptr[b].qs[215], low4bits_0, vl);
                for (int z = 9; z < 16; z++) {
                    // 权重矩阵的0-3列和32-35列
                    vint8m1_t src0_0 = __riscv_vle8_v_i8m1(b_ptr[b].qs + z * nrr, vl);
                    vint8m1_t low4bits_0 = __riscv_vsra_vx_i8m1(__riscv_vsll_vx_i8m1(src0_0, 4, vl), 4, vl);
                    const int64_t ssb = z * 24;
                    suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[b].qs[ssb], low4bits_0, vl);
                    suml_32_1 = __riscv_vwmacc_vx_i16m2(suml_32_1, a_ptr[b].qs[ssb + 1], low4bits_0, vl);
                    suml_32_2 = __riscv_vwmacc_vx_i16m2(suml_32_2, a_ptr[b].qs[ssb + 2], low4bits_0, vl);
                    suml_32_3 = __riscv_vwmacc_vx_i16m2(suml_32_3, a_ptr[b].qs[ssb + 3], low4bits_0, vl);
                    suml_32_4 = __riscv_vwmacc_vx_i16m2(suml_32_4, a_ptr[b].qs[ssb + 4], low4bits_0, vl);
                    suml_32_5 = __riscv_vwmacc_vx_i16m2(suml_32_5, a_ptr[b].qs[ssb + 5], low4bits_0, vl);
                    suml_32_6 = __riscv_vwmacc_vx_i16m2(suml_32_6, a_ptr[b].qs[ssb + 6], low4bits_0, vl);
                    suml_32_7 = __riscv_vwmacc_vx_i16m2(suml_32_7, a_ptr[b].qs[ssb + 7], low4bits_0, vl);
                    suml_32_8 = __riscv_vwmacc_vx_i16m2(suml_32_8, a_ptr[b].qs[ssb + 8], low4bits_0, vl);
                    suml_32_9 = __riscv_vwmacc_vx_i16m2(suml_32_9, a_ptr[b].qs[ssb + 9], low4bits_0, vl);
                    suml_32_10 = __riscv_vwmacc_vx_i16m2(suml_32_10, a_ptr[b].qs[ssb + 10], low4bits_0, vl);
                    suml_32_11 = __riscv_vwmacc_vx_i16m2(suml_32_11, a_ptr[b].qs[ssb + 11], low4bits_0, vl);
                    low4bits_0 = __riscv_vsra_vx_i8m1(src0_0, 4, vl);
                    suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[b].qs[ssb + 12], low4bits_0, vl);
                    suml_32_1 = __riscv_vwmacc_vx_i16m2(suml_32_1, a_ptr[b].qs[ssb + 13], low4bits_0, vl);
                    suml_32_2 = __riscv_vwmacc_vx_i16m2(suml_32_2, a_ptr[b].qs[ssb + 14], low4bits_0, vl);
                    suml_32_3 = __riscv_vwmacc_vx_i16m2(suml_32_3, a_ptr[b].qs[ssb + 15], low4bits_0, vl);
                    suml_32_4 = __riscv_vwmacc_vx_i16m2(suml_32_4, a_ptr[b].qs[ssb + 16], low4bits_0, vl);
                    suml_32_5 = __riscv_vwmacc_vx_i16m2(suml_32_5, a_ptr[b].qs[ssb + 17], low4bits_0, vl);
                    suml_32_6 = __riscv_vwmacc_vx_i16m2(suml_32_6, a_ptr[b].qs[ssb + 18], low4bits_0, vl);
                    suml_32_7 = __riscv_vwmacc_vx_i16m2(suml_32_7, a_ptr[b].qs[ssb + 19], low4bits_0, vl);
                    suml_32_8 = __riscv_vwmacc_vx_i16m2(suml_32_8, a_ptr[b].qs[ssb + 20], low4bits_0, vl);
                    suml_32_9 = __riscv_vwmacc_vx_i16m2(suml_32_9, a_ptr[b].qs[ssb + 21], low4bits_0, vl);
                    suml_32_10 = __riscv_vwmacc_vx_i16m2(suml_32_10, a_ptr[b].qs[ssb + 22], low4bits_0, vl);
                    suml_32_11 = __riscv_vwmacc_vx_i16m2(suml_32_11, a_ptr[b].qs[ssb + 23], low4bits_0, vl);
                }

                __riscv_vse32_v_i32m4(temp, __riscv_vwadd_wv_i32m4(__riscv_vle32_v_i32m4(temp, vl), suml_32_0, vl), vl);
                __riscv_vse32_v_i32m4(temp+nrr, __riscv_vwadd_wv_i32m4(__riscv_vle32_v_i32m4(temp+nrr, vl), suml_32_1, vl), vl);
                __riscv_vse32_v_i32m4(temp+nrr*2, __riscv_vwadd_wv_i32m4(__riscv_vle32_v_i32m4(temp+nrr*2, vl), suml_32_2, vl), vl);
                __riscv_vse32_v_i32m4(temp+nrr*3, __riscv_vwadd_wv_i32m4(__riscv_vle32_v_i32m4(temp+nrr*3, vl), suml_32_3, vl), vl);
                __riscv_vse32_v_i32m4(temp+nrr*4, __riscv_vwadd_wv_i32m4(__riscv_vle32_v_i32m4(temp+nrr*4, vl), suml_32_4, vl), vl);
                __riscv_vse32_v_i32m4(temp+nrr*5, __riscv_vwadd_wv_i32m4(__riscv_vle32_v_i32m4(temp+nrr*5, vl), suml_32_5, vl), vl);
                __riscv_vse32_v_i32m4(temp+nrr*6, __riscv_vwadd_wv_i32m4(__riscv_vle32_v_i32m4(temp+nrr*6, vl), suml_32_6, vl), vl);
                __riscv_vse32_v_i32m4(temp+nrr*7, __riscv_vwadd_wv_i32m4(__riscv_vle32_v_i32m4(temp+nrr*7, vl), suml_32_7, vl), vl);
                __riscv_vse32_v_i32m4(temp+nrr*8, __riscv_vwadd_wv_i32m4(__riscv_vle32_v_i32m4(temp+nrr*8, vl), suml_32_8, vl), vl);
                __riscv_vse32_v_i32m4(temp+nrr*9, __riscv_vwadd_wv_i32m4(__riscv_vle32_v_i32m4(temp+nrr*9, vl), suml_32_9, vl), vl);
                __riscv_vse32_v_i32m4(temp+nrr*10, __riscv_vwadd_wv_i32m4(__riscv_vle32_v_i32m4(temp+nrr*10, vl), suml_32_10, vl), vl);
                __riscv_vse32_v_i32m4(temp+nrr*11, __riscv_vwadd_wv_i32m4(__riscv_vle32_v_i32m4(temp+nrr*11, vl), suml_32_11, vl), vl);


                vfloat32m4_t vec_scales_32 = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vreinterpret_v_u16m2_f16m2(__riscv_vle16_v_u16m2(b_ptr[b].d, vl)), vl);
            
                //先将sum结果与q4的d相乘，再与q8的d相乘
                __riscv_vse32_v_f32m4(sum_row,
                    __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row, vl), 
                        __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(temp, vl), vl), 
                        __riscv_vfmul_vf_f32m4(vec_scales_32, GGML_FP16_TO_FP32(a_ptr[b].d[0]), vl), 
                    vl),
                vl);
                __riscv_vse32_v_f32m4(sum_row + nrr,
                    __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nrr, vl),
                    __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(temp + nrr, vl), vl), 
                        __riscv_vfmul_vf_f32m4(vec_scales_32, GGML_FP16_TO_FP32(a_ptr[b].d[1]), vl), 
                    vl),
                vl);
                __riscv_vse32_v_f32m4(sum_row + nrr * 2,
                    __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nrr * 2, vl),
                    __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(temp + nrr * 2, vl), vl), 
                        __riscv_vfmul_vf_f32m4(vec_scales_32, GGML_FP16_TO_FP32(a_ptr[b].d[2]), vl), 
                    vl),                   
                vl);
                __riscv_vse32_v_f32m4(sum_row + nrr * 3,
                    __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nrr * 3, vl),
                    __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(temp + nrr * 3, vl), vl), 
                        __riscv_vfmul_vf_f32m4(vec_scales_32, GGML_FP16_TO_FP32(a_ptr[b].d[3]), vl), 
                    vl),                 
                vl);
                __riscv_vse32_v_f32m4(sum_row + nrr * 4,
                    __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nrr * 4, vl),
                    __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(temp + nrr * 4, vl), vl), 
                        __riscv_vfmul_vf_f32m4(vec_scales_32, GGML_FP16_TO_FP32(a_ptr[b].d[4]), vl), 
                    vl),                   
                 vl);
                __riscv_vse32_v_f32m4(sum_row + nrr * 5,
                    __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nrr * 5, vl),
                    __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(temp + nrr * 5, vl), vl), 
                        __riscv_vfmul_vf_f32m4(vec_scales_32, GGML_FP16_TO_FP32(a_ptr[b].d[5]), vl), 
                    vl),                 
                 vl);
                __riscv_vse32_v_f32m4(sum_row + nrr * 6,
                    __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nrr * 6, vl),
                    __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(temp + nrr * 6, vl), vl), 
                        __riscv_vfmul_vf_f32m4(vec_scales_32, GGML_FP16_TO_FP32(a_ptr[b].d[6]), vl), 
                    vl),                  
                vl);
                __riscv_vse32_v_f32m4(sum_row + nrr * 7,
                    __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nrr * 7, vl),
                    __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(temp + nrr * 7, vl), vl), 
                        __riscv_vfmul_vf_f32m4(vec_scales_32, GGML_FP16_TO_FP32(a_ptr[b].d[7]), vl), 
                    vl),                  
                vl);
                __riscv_vse32_v_f32m4(sum_row + nrr * 8,
                    __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nrr * 8, vl),
                    __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(temp + nrr * 8, vl), vl), 
                        __riscv_vfmul_vf_f32m4(vec_scales_32, GGML_FP16_TO_FP32(a_ptr[b].d[8]), vl), 
                    vl),                 
                vl);
                __riscv_vse32_v_f32m4(sum_row + nrr * 9,
                    __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nrr * 9, vl),
                    __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(temp + nrr * 9, vl), vl), 
                        __riscv_vfmul_vf_f32m4(vec_scales_32, GGML_FP16_TO_FP32(a_ptr[b].d[9]), vl), 
                    vl),                  
                vl);
                __riscv_vse32_v_f32m4(sum_row + nrr * 10,
                    __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nrr * 10, vl),
                    __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(temp + nrr * 10, vl), vl), 
                        __riscv_vfmul_vf_f32m4(vec_scales_32, GGML_FP16_TO_FP32(a_ptr[b].d[10]), vl), 
                    vl),                  
                vl);
                __riscv_vse32_v_f32m4(sum_row + nrr * 11,
                    __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + nrr * 11, vl),
                    __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(temp + nrr * 11, vl), vl), 
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

    // int64_t end_time1 = ggml_time_us();
    // int64_t gemm_time1 = end_time1 - start_time;
    // printf("anr/4: %d, nr/4: %d, gemm_time1: %.2lf ms", anr/4, nr/4, gemm_time1*1e-3);

    const block_q8_0x4 *a_ptr_start1 = (const block_q8_0x4 *)vy;

    for (int y = anr / 4; y < nr / 4; y++) { // M

       const block_q8_0x4 *a_ptr = a_ptr_start1 + (y * nb);
        for (int64_t x = 0; x < nc / nrr; x++) { // N

            const block_q4_0x32 *b_ptr = b_ptr_start + (x * nb);

            // 行累加和
            // float sum_row[nrr * 4] = {0.0};
            vfloat32m4_t sum_row0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t sum_row1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t sum_row2 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t sum_row3 = __riscv_vfmv_v_f_f32m4(0.0f, vl);

            for (int64_t b = 0; b < nb; b++) { // K

                // Loop to iterate over the eight sub blocks of a super block - two sub blocks are processed per
                // iteration
                vint16m2_t suml_32_0 = __riscv_vmv_v_x_i16m2(0, vl);
                vint16m2_t suml_32_1 = __riscv_vmv_v_x_i16m2(0, vl);
                vint16m2_t suml_32_2 = __riscv_vmv_v_x_i16m2(0, vl);
                vint16m2_t suml_32_3 = __riscv_vmv_v_x_i16m2(0, vl);
                for (int z = 0; z < 8; z++) {
                    // 权重矩阵的0-3列和32-35列
                    vint8m1_t src0_0 = __riscv_vle8_v_i8m1(b_ptr[b].qs + z * nrr, vl);
                    vint8m1_t low4bits_0 = __riscv_vsra_vx_i8m1(__riscv_vsll_vx_i8m1(src0_0, 4, vl), 4, vl);
                    const int64_t ssb = z * 8;
                    suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[b].qs[ssb], low4bits_0, vl);
                    suml_32_1 = __riscv_vwmacc_vx_i16m2(suml_32_1, a_ptr[b].qs[ssb + 1], low4bits_0, vl);
                    suml_32_2 = __riscv_vwmacc_vx_i16m2(suml_32_2, a_ptr[b].qs[ssb + 2], low4bits_0, vl);
                    suml_32_3 = __riscv_vwmacc_vx_i16m2(suml_32_3, a_ptr[b].qs[ssb + 3], low4bits_0, vl);
                    low4bits_0 = __riscv_vsra_vx_i8m1(src0_0, 4, vl);
                    suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[b].qs[ssb + 4], low4bits_0, vl);
                    suml_32_1 = __riscv_vwmacc_vx_i16m2(suml_32_1, a_ptr[b].qs[ssb + 5], low4bits_0, vl);
                    suml_32_2 = __riscv_vwmacc_vx_i16m2(suml_32_2, a_ptr[b].qs[ssb + 6], low4bits_0, vl);
                    suml_32_3 = __riscv_vwmacc_vx_i16m2(suml_32_3, a_ptr[b].qs[ssb + 7], low4bits_0, vl);
                }

                vint32m4_t suml_32_00 = __riscv_vmv_v_x_i32m4(0, vl);
                vint32m4_t suml_32_10 = __riscv_vmv_v_x_i32m4(0, vl);
                vint32m4_t suml_32_20 = __riscv_vmv_v_x_i32m4(0, vl);
                vint32m4_t suml_32_30 = __riscv_vmv_v_x_i32m4(0, vl);

                suml_32_00 = __riscv_vwadd_wv_i32m4(suml_32_00, suml_32_0, vl);
                suml_32_10 = __riscv_vwadd_wv_i32m4(suml_32_10, suml_32_1, vl);
                suml_32_20 = __riscv_vwadd_wv_i32m4(suml_32_20, suml_32_2, vl);
                suml_32_30 = __riscv_vwadd_wv_i32m4(suml_32_30, suml_32_3, vl);

                suml_32_0 = __riscv_vmv_v_x_i16m2(0, vl);
                suml_32_1 = __riscv_vmv_v_x_i16m2(0, vl);
                suml_32_2 = __riscv_vmv_v_x_i16m2(0, vl);
                suml_32_3 = __riscv_vmv_v_x_i16m2(0, vl);
                for (int z = 8; z < 16; z++) {
                    // 权重矩阵的0-3列和32-35列
                    vint8m1_t src0_0 = __riscv_vle8_v_i8m1(b_ptr[b].qs + z * nrr, vl);
                    vint8m1_t low4bits_0 = __riscv_vsra_vx_i8m1(__riscv_vsll_vx_i8m1(src0_0, 4, vl), 4, vl);
                    const int64_t ssb = z * 8;
                    suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[b].qs[ssb], low4bits_0, vl);
                    suml_32_1 = __riscv_vwmacc_vx_i16m2(suml_32_1, a_ptr[b].qs[ssb + 1], low4bits_0, vl);
                    suml_32_2 = __riscv_vwmacc_vx_i16m2(suml_32_2, a_ptr[b].qs[ssb + 2], low4bits_0, vl);
                    suml_32_3 = __riscv_vwmacc_vx_i16m2(suml_32_3, a_ptr[b].qs[ssb + 3], low4bits_0, vl);
                    low4bits_0 = __riscv_vsra_vx_i8m1(src0_0, 4, vl);
                    suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[b].qs[ssb + 4], low4bits_0, vl);
                    suml_32_1 = __riscv_vwmacc_vx_i16m2(suml_32_1, a_ptr[b].qs[ssb + 5], low4bits_0, vl);
                    suml_32_2 = __riscv_vwmacc_vx_i16m2(suml_32_2, a_ptr[b].qs[ssb + 6], low4bits_0, vl);
                    suml_32_3 = __riscv_vwmacc_vx_i16m2(suml_32_3, a_ptr[b].qs[ssb + 7], low4bits_0, vl);
                }

                suml_32_00 = __riscv_vwadd_wv_i32m4(suml_32_00, suml_32_0, vl);
                suml_32_10 = __riscv_vwadd_wv_i32m4(suml_32_10, suml_32_1, vl);
                suml_32_20 = __riscv_vwadd_wv_i32m4(suml_32_20, suml_32_2, vl);
                suml_32_30 = __riscv_vwadd_wv_i32m4(suml_32_30, suml_32_3, vl);

                // vint16m2_t vec_scales_32 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vle16_v_u16m2(b_ptr[b].d, vl));
                vfloat32m4_t vec_scales_32 = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vreinterpret_v_u16m2_f16m2(__riscv_vle16_v_u16m2(b_ptr[b].d, vl)), vl);

                sum_row0 = __riscv_vfmacc_vv_f32m4(sum_row0, __riscv_vfcvt_f_x_v_f32m4(suml_32_00, vl), __riscv_vfmul_vf_f32m4(vec_scales_32, GGML_FP16_TO_FP32(a_ptr[b].d[0]), vl), vl);
                sum_row1 = __riscv_vfmacc_vv_f32m4(sum_row1, __riscv_vfcvt_f_x_v_f32m4(suml_32_10, vl), __riscv_vfmul_vf_f32m4(vec_scales_32, GGML_FP16_TO_FP32(a_ptr[b].d[1]), vl), vl);
                sum_row2 = __riscv_vfmacc_vv_f32m4(sum_row2, __riscv_vfcvt_f_x_v_f32m4(suml_32_20, vl), __riscv_vfmul_vf_f32m4(vec_scales_32, GGML_FP16_TO_FP32(a_ptr[b].d[2]), vl), vl);
                sum_row3 = __riscv_vfmacc_vv_f32m4(sum_row3, __riscv_vfcvt_f_x_v_f32m4(suml_32_30, vl), __riscv_vfmul_vf_f32m4(vec_scales_32, GGML_FP16_TO_FP32(a_ptr[b].d[3]), vl), vl);
            }
                
            __riscv_vse32_v_f32m4((float *)(s + ((y * 4 + 0) * bs + x * nrr)), sum_row0, vl);
            __riscv_vse32_v_f32m4((float *)(s + ((y * 4 + 1) * bs + x * nrr)), sum_row1, vl);
            __riscv_vse32_v_f32m4((float *)(s + ((y * 4 + 2) * bs + x * nrr)), sum_row2, vl);
            __riscv_vse32_v_f32m4((float *)(s + ((y * 4 + 3) * bs + x * nrr)), sum_row3, vl);
        }
    }

}