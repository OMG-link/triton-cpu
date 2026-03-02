#include "ggml_def.h"

// using block_q4_Kx16 = block_q4_Kx<16>;
// using block_q8_Kx12 = block_q8_Kx<12>;
//#define __riscv_vwmacc_vx_i16m2(vs, x, vt, vl) __riscv_vwmacc_vv_i16m2(vs, __riscv_vmv_v_x_i8m1(x, vl), vt, vl)
void ggml_gemm_iq2_xxs_12x16_q8_K(int n, float * __restrict s, size_t bs, const void * __restrict vx, const void * __restrict vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 16;
    const int blocklen = 4;

    // assert (n % qk == 0);
    // assert (nr % 4 == 0);
    // assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    const int mrr = 12;
    const int nrr = 16;
    int anr = nr - nr % mrr;

    //const size_t vl = __riscv_vlenb();  //128 / 8 = 16

    uint8_t pattern_array[16] = {};
    int8_t qx_temp[512];
    int16_t base_data0[64] = { -1, -1, -1, -1,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, 
                                0, 0, 0, 0, -1, -1, -1, -1,  0, 0, 0, 0,  0, 0, 0, 0,
                                0, 0, 0, 0,  0, 0, 0, 0, -1, -1, -1, -1,  0, 0, 0, 0,
                                0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, -1, -1, -1, -1};

    vint32m1_t vzero = __riscv_vmv_s_x_i32m1(0, 4);
    
    for (int y = 0; y < anr / mrr; y++) { 
        const block_q8_Kx12 * a_ptr = (const block_q8_Kx12 *) vy + (y * nb);


        for(int x = 0; x < nc / nrr; x++){
            float sum_row[mrr*nrr] = {0.0};
            const block_iq2_xxsx16 * b_ptr = (const block_iq2_xxsx16 *) vx + (x * nb);


            for (int ibl = 0; ibl < nb  ; ++ibl) {
                int32_t sum_row0[mrr*nrr] = {0};
                vfloat32m4_t d16 = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vreinterpret_v_u16m2_f16m2(__riscv_vle16_v_u16m2(b_ptr[ibl].d, 16)), 16);
                auto qs = b_ptr[ibl].qs;
                

                for (int ib = 0; ib < QK_K/32; ++ib) {
                    vint32m4_t sums = __riscv_vmv_v_x_i32m4(0, 16);
                    
                    
                    for(int ibk = 0; ibk < 4; ibk++){
                        vuint8m1_t sas = __riscv_vle8_v_u8m1(b_ptr[ibl].sas + 64*ib + 16*ibk, 16);
                        // uint32_t pattern_val = 0x10080402;
                        
                        for(int i=0; i < 16; i+=4){
                            pattern_array[i] = 2;
                            pattern_array[i + 1] = 4;
                            pattern_array[i + 2] = 8;
                            pattern_array[i + 3] = 16;
                        }
                        vint16m2_t prod16 = __riscv_vwmul_vv_i16m2(__riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vv_u8m1(sas, __riscv_vmv_v_x_u8m1(1, 16), 16)), __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vle8_v_u8m1(pattern_array, 16)), 16);
                        
                        vint16m2_t masks0 = __riscv_vle16_v_i16m2(base_data0, 16);
                        masks0 = __riscv_vand_vv_i16m2(prod16, masks0, 16);
                        vint32m1_t sum16_0 = __riscv_vwredsum_vs_i16m2_i32m1(masks0, vzero, 16);
                        sums = __riscv_vslide1down_vx_i32m4(sums,__riscv_vmv_x_s_i32m1_i32(sum16_0), 16);

                        masks0 = __riscv_vle16_v_i16m2(base_data0 + 16, 16);
                        masks0 = __riscv_vand_vv_i16m2(prod16, masks0, 16);
                        sum16_0 = __riscv_vwredsum_vs_i16m2_i32m1(masks0, vzero, 16);
                        sums = __riscv_vslide1down_vx_i32m4(sums,__riscv_vmv_x_s_i32m1_i32(sum16_0), 16);

                        masks0 = __riscv_vle16_v_i16m2(base_data0 + 32, 16);
                        masks0 = __riscv_vand_vv_i16m2(prod16, masks0, 16);
                        sum16_0 = __riscv_vwredsum_vs_i16m2_i32m1(masks0, vzero, 16);
                        sums = __riscv_vslide1down_vx_i32m4(sums,__riscv_vmv_x_s_i32m1_i32(sum16_0), 16);

                        masks0 = __riscv_vle16_v_i16m2(base_data0 + 48, 16);
                        masks0 = __riscv_vand_vv_i16m2(prod16, masks0, 16);
                        sum16_0 = __riscv_vwredsum_vs_i16m2_i32m1(masks0, vzero, 16);
                        sums = __riscv_vslide1down_vx_i32m4(sums,__riscv_vmv_x_s_i32m1_i32(sum16_0), 16);

                        
                        vuint8m1_t signs128 = __riscv_vand_vx_u8m1(sas, 0xFE, 16);
                        signs128 = __riscv_vxor_vv_u8m1(signs128, __riscv_vsrl_vx_u8m1(signs128, 1, 16), 16);
                        
                        vuint8m1_t shuffle;
                        for( int i = 0; i < 8; ++i){
                            pattern_array[i] = 0;
                            pattern_array[i + 8] = 1;
                        }
                        shuffle = __riscv_vle8_v_u8m1(pattern_array, 16);
                        for (int i = 0; i < 16; i+=8) {
                            pattern_array[i]     = 1;
                            pattern_array[i+1]   = 2;
                            pattern_array[i+2]   = 4;
                            pattern_array[i+3]   = 8;
                            pattern_array[i+4]   = 16;
                            pattern_array[i+5]   = 32;
                            pattern_array[i+6]   = 64;
                            pattern_array[i+7]   = 128;
                        }
                        vuint8m1_t smask = __riscv_vle8_v_u8m1(pattern_array, 16);
                        
                        
                        vuint8m1_t m1 = __riscv_vmv_v_x_u8m1(1, 16);
                        vuint8m1_t step  = __riscv_vmv_v_x_u8m1(2, 16);
                        vint8m1_t qx0,qx1,qx2,qx3,qx4,qx5,qx6,qx7;
                        uint8_t qx_tmp0[16];
                        memcpy(qx_tmp0 + 0, &iq2xxs_grid[qs[0 + 16*ibk]], 8);
                        memcpy(qx_tmp0 + 8, &iq2xxs_grid[qs[1 + 16*ibk]], 8);
                        qx0 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vle8_v_u8m1(qx_tmp0, 16));
                        qx0 = __riscv_vmul_vv_i8m1(qx0, __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vor_vv_u8m1(__riscv_vmerge_vxm_u8m1(__riscv_vmv_v_x_u8m1(0, 16), 0xFF, __riscv_vmseq_vv_u8m1_b8(__riscv_vand_vv_u8m1(__riscv_vrgather_vv_u8m1(signs128, shuffle, 16), smask, 16), smask, 16), 16), m1, 16)), 16);
                        shuffle = __riscv_vadd_vv_u8m1(shuffle, step, 16);

                        //uint8_t qx_tmp1[32];
                        memcpy(qx_tmp0 + 0, &iq2xxs_grid[qs[2 + 16*ibk]], 8);
                        memcpy(qx_tmp0 + 8, &iq2xxs_grid[qs[3 + 16*ibk]], 8);
                        qx1 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vle8_v_u8m1(qx_tmp0, 16));
                        qx1 = __riscv_vmul_vv_i8m1(qx1, __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vor_vv_u8m1(__riscv_vmerge_vxm_u8m1(__riscv_vmv_v_x_u8m1(0, 16), 0xFF, __riscv_vmseq_vv_u8m1_b8(__riscv_vand_vv_u8m1(__riscv_vrgather_vv_u8m1(signs128, shuffle, 16), smask, 16), smask, 16), 16), m1, 16)), 16);
                        shuffle = __riscv_vadd_vv_u8m1(shuffle, step, 16);

                        // uint8_t qx_tmp2[32];
                        memcpy(qx_tmp0 + 0, &iq2xxs_grid[qs[4 + 16*ibk]], 8);
                        memcpy(qx_tmp0 + 8, &iq2xxs_grid[qs[5 + 16*ibk]], 8);
                        qx2 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vle8_v_u8m1(qx_tmp0, 16));
                        qx2 = __riscv_vmul_vv_i8m1(qx2, __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vor_vv_u8m1(__riscv_vmerge_vxm_u8m1(__riscv_vmv_v_x_u8m1(0, 16), 0xFF, __riscv_vmseq_vv_u8m1_b8(__riscv_vand_vv_u8m1(__riscv_vrgather_vv_u8m1(signs128, shuffle, 16), smask, 16), smask, 16), 16), m1, 16)), 16);
                        shuffle = __riscv_vadd_vv_u8m1(shuffle, step, 16);

                        // uint8_t qx_tmp3[32];
                        memcpy(qx_tmp0 + 0, &iq2xxs_grid[qs[6 + 16*ibk]], 8);
                        memcpy(qx_tmp0 + 8, &iq2xxs_grid[qs[7 + 16*ibk]], 8);
                        qx3 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vle8_v_u8m1(qx_tmp0, 16));
                        qx3 = __riscv_vmul_vv_i8m1(qx3, __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vor_vv_u8m1(__riscv_vmerge_vxm_u8m1(__riscv_vmv_v_x_u8m1(0, 16), 0xFF, __riscv_vmseq_vv_u8m1_b8(__riscv_vand_vv_u8m1(__riscv_vrgather_vv_u8m1(signs128, shuffle, 16), smask, 16), smask, 16), 16), m1, 16)), 16);
                        shuffle = __riscv_vadd_vv_u8m1(shuffle, step, 16);

                        memcpy(qx_tmp0 + 0, &iq2xxs_grid[qs[8 + 16*ibk]], 8);
                        memcpy(qx_tmp0 + 8, &iq2xxs_grid[qs[9 + 16*ibk]], 8);
                        qx4 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vle8_v_u8m1(qx_tmp0, 16));
                        qx4 = __riscv_vmul_vv_i8m1(qx4, __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vor_vv_u8m1(__riscv_vmerge_vxm_u8m1(__riscv_vmv_v_x_u8m1(0, 16), 0xFF, __riscv_vmseq_vv_u8m1_b8(__riscv_vand_vv_u8m1(__riscv_vrgather_vv_u8m1(signs128, shuffle, 16), smask, 16), smask, 16), 16), m1, 16)), 16);
                        shuffle = __riscv_vadd_vv_u8m1(shuffle, step, 16);

                        memcpy(qx_tmp0 + 0, &iq2xxs_grid[qs[10 + 16*ibk]], 8);
                        memcpy(qx_tmp0 + 8, &iq2xxs_grid[qs[11 + 16*ibk]], 8);
                        qx5 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vle8_v_u8m1(qx_tmp0, 16));
                        qx5 = __riscv_vmul_vv_i8m1(qx5, __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vor_vv_u8m1(__riscv_vmerge_vxm_u8m1(__riscv_vmv_v_x_u8m1(0, 16), 0xFF, __riscv_vmseq_vv_u8m1_b8(__riscv_vand_vv_u8m1(__riscv_vrgather_vv_u8m1(signs128, shuffle, 16), smask, 16), smask, 16), 16), m1, 16)), 16);
                        shuffle = __riscv_vadd_vv_u8m1(shuffle, step, 16);

                        memcpy(qx_tmp0 + 0, &iq2xxs_grid[qs[12 + 16*ibk]], 8);
                        memcpy(qx_tmp0 + 8, &iq2xxs_grid[qs[13 + 16*ibk]], 8);
                        qx6 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vle8_v_u8m1(qx_tmp0, 16));
                        qx6 = __riscv_vmul_vv_i8m1(qx6, __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vor_vv_u8m1(__riscv_vmerge_vxm_u8m1(__riscv_vmv_v_x_u8m1(0, 16), 0xFF, __riscv_vmseq_vv_u8m1_b8(__riscv_vand_vv_u8m1(__riscv_vrgather_vv_u8m1(signs128, shuffle, 16), smask, 16), smask, 16), 16), m1, 16)), 16);
                        shuffle = __riscv_vadd_vv_u8m1(shuffle, step, 16);

                        memcpy(qx_tmp0 + 0, &iq2xxs_grid[qs[14 + 16*ibk]], 8);
                        memcpy(qx_tmp0 + 8, &iq2xxs_grid[qs[15 + 16*ibk]], 8);
                        qx7 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vle8_v_u8m1(qx_tmp0, 16));
                        qx7 = __riscv_vmul_vv_i8m1(qx7, __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vor_vv_u8m1(__riscv_vmerge_vxm_u8m1(__riscv_vmv_v_x_u8m1(0, 16), 0xFF, __riscv_vmseq_vv_u8m1_b8(__riscv_vand_vv_u8m1(__riscv_vrgather_vv_u8m1(signs128, shuffle, 16), smask, 16), smask, 16), 16), m1, 16)), 16);

                        __riscv_vse8_v_i8m1(qx_temp + 0 + 128*ibk,qx0,16);
                        __riscv_vse8_v_i8m1(qx_temp + 16 + 128*ibk,qx1,16);
                        __riscv_vse8_v_i8m1(qx_temp + 32 + 128*ibk,qx2,16);
                        __riscv_vse8_v_i8m1(qx_temp + 48 + 128*ibk,qx3,16);
                        __riscv_vse8_v_i8m1(qx_temp + 64 + 128*ibk,qx4,16);
                        __riscv_vse8_v_i8m1(qx_temp + 80 + 128*ibk,qx5,16);
                        __riscv_vse8_v_i8m1(qx_temp + 96 + 128*ibk,qx6,16);
                        __riscv_vse8_v_i8m1(qx_temp + 112 + 128*ibk,qx7,16);

                    }
                    qs += 64;

                    vint16m2_t suml_32_0 = __riscv_vmv_v_x_i16m2(0, 16);
                    vint16m2_t suml_32_1 = __riscv_vmv_v_x_i16m2(0, 16);
                    vint16m2_t suml_32_2 = __riscv_vmv_v_x_i16m2(0, 16);
                    vint16m2_t suml_32_3 = __riscv_vmv_v_x_i16m2(0, 16);
                    vint16m2_t suml_32_4 = __riscv_vmv_v_x_i16m2(0, 16);
                    vint16m2_t suml_32_5 = __riscv_vmv_v_x_i16m2(0, 16);
                    vint16m2_t suml_32_6 = __riscv_vmv_v_x_i16m2(0, 16);
                    vint16m2_t suml_32_7 = __riscv_vmv_v_x_i16m2(0, 16);
                    vint16m2_t suml_32_8 = __riscv_vmv_v_x_i16m2(0, 16);
                    vint16m2_t suml_32_9 = __riscv_vmv_v_x_i16m2(0, 16);
                    vint16m2_t suml_32_10 = __riscv_vmv_v_x_i16m2(0, 16);
                    vint16m2_t suml_32_11 = __riscv_vmv_v_x_i16m2(0, 16);

                    vint8m1x4_t rhs;
                    vint8m1_t rhs0, rhs1, rhs2, rhs3;
                    for(int jj=0;jj<32;jj+=4)
                    {
                        rhs = __riscv_vlsseg4e8_v_i8m1x4(qx_temp + jj, 32, 16);
                        rhs0 = __riscv_vget_i8m1(rhs, 0);
                        rhs1 = __riscv_vget_i8m1(rhs, 1);
                        rhs2 = __riscv_vget_i8m1(rhs, 2);
                        rhs3 = __riscv_vget_i8m1(rhs, 3);

                        
                        suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[ibl].qs[0 + 12*jj + 384*ib], rhs0, 16);
                        suml_32_1 = __riscv_vwmacc_vx_i16m2(suml_32_1, a_ptr[ibl].qs[1 + 12*jj + 384*ib], rhs0, 16);
                        suml_32_2 = __riscv_vwmacc_vx_i16m2(suml_32_2, a_ptr[ibl].qs[2 + 12*jj + 384*ib], rhs0, 16);
                        suml_32_3 = __riscv_vwmacc_vx_i16m2(suml_32_3, a_ptr[ibl].qs[3 + 12*jj + 384*ib], rhs0, 16);
                        suml_32_4 = __riscv_vwmacc_vx_i16m2(suml_32_4, a_ptr[ibl].qs[4 + 12*jj + 384*ib], rhs0, 16);
                        suml_32_5 = __riscv_vwmacc_vx_i16m2(suml_32_5, a_ptr[ibl].qs[5 + 12*jj + 384*ib], rhs0, 16);
                        suml_32_6 = __riscv_vwmacc_vx_i16m2(suml_32_6, a_ptr[ibl].qs[6 + 12*jj + 384*ib], rhs0, 16);
                        suml_32_7 = __riscv_vwmacc_vx_i16m2(suml_32_7, a_ptr[ibl].qs[7 + 12*jj + 384*ib], rhs0, 16);
                        suml_32_8 = __riscv_vwmacc_vx_i16m2(suml_32_8, a_ptr[ibl].qs[8 + 12*jj + 384*ib], rhs0, 16);
                        suml_32_9 = __riscv_vwmacc_vx_i16m2(suml_32_9, a_ptr[ibl].qs[9 + 12*jj + 384*ib], rhs0, 16);
                        suml_32_10 = __riscv_vwmacc_vx_i16m2(suml_32_10, a_ptr[ibl].qs[10 + 12*jj + 384*ib], rhs0, 16);
                        suml_32_11 = __riscv_vwmacc_vx_i16m2(suml_32_11, a_ptr[ibl].qs[11 + 12*jj + 384*ib], rhs0, 16);

                        suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[ibl].qs[12 + 12*jj + 384*ib], rhs1, 16);
                        suml_32_1 = __riscv_vwmacc_vx_i16m2(suml_32_1, a_ptr[ibl].qs[13 + 12*jj + 384*ib], rhs1, 16);
                        suml_32_2 = __riscv_vwmacc_vx_i16m2(suml_32_2, a_ptr[ibl].qs[14 + 12*jj + 384*ib], rhs1, 16);
                        suml_32_3 = __riscv_vwmacc_vx_i16m2(suml_32_3, a_ptr[ibl].qs[15 + 12*jj + 384*ib], rhs1, 16);
                        suml_32_4 = __riscv_vwmacc_vx_i16m2(suml_32_4, a_ptr[ibl].qs[16 + 12*jj + 384*ib], rhs1, 16);
                        suml_32_5 = __riscv_vwmacc_vx_i16m2(suml_32_5, a_ptr[ibl].qs[17 + 12*jj + 384*ib], rhs1, 16);
                        suml_32_6 = __riscv_vwmacc_vx_i16m2(suml_32_6, a_ptr[ibl].qs[18 + 12*jj + 384*ib], rhs1, 16);
                        suml_32_7 = __riscv_vwmacc_vx_i16m2(suml_32_7, a_ptr[ibl].qs[19 + 12*jj + 384*ib], rhs1, 16);
                        suml_32_8 = __riscv_vwmacc_vx_i16m2(suml_32_8, a_ptr[ibl].qs[20 + 12*jj + 384*ib], rhs1, 16);
                        suml_32_9 = __riscv_vwmacc_vx_i16m2(suml_32_9, a_ptr[ibl].qs[21 + 12*jj + 384*ib], rhs1, 16);
                        suml_32_10 = __riscv_vwmacc_vx_i16m2(suml_32_10, a_ptr[ibl].qs[22 + 12*jj + 384*ib], rhs1, 16);
                        suml_32_11 = __riscv_vwmacc_vx_i16m2(suml_32_11, a_ptr[ibl].qs[23 + 12*jj + 384*ib], rhs1, 16);

                        suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[ibl].qs[24 + 12*jj + 384*ib], rhs2, 16);
                        suml_32_1 = __riscv_vwmacc_vx_i16m2(suml_32_1, a_ptr[ibl].qs[25 + 12*jj + 384*ib], rhs2, 16);
                        suml_32_2 = __riscv_vwmacc_vx_i16m2(suml_32_2, a_ptr[ibl].qs[26 + 12*jj + 384*ib], rhs2, 16);
                        suml_32_3 = __riscv_vwmacc_vx_i16m2(suml_32_3, a_ptr[ibl].qs[27 + 12*jj + 384*ib], rhs2, 16);
                        suml_32_4 = __riscv_vwmacc_vx_i16m2(suml_32_4, a_ptr[ibl].qs[28 + 12*jj + 384*ib], rhs2, 16);
                        suml_32_5 = __riscv_vwmacc_vx_i16m2(suml_32_5, a_ptr[ibl].qs[29 + 12*jj + 384*ib], rhs2, 16);
                        suml_32_6 = __riscv_vwmacc_vx_i16m2(suml_32_6, a_ptr[ibl].qs[30 + 12*jj + 384*ib], rhs2, 16);
                        suml_32_7 = __riscv_vwmacc_vx_i16m2(suml_32_7, a_ptr[ibl].qs[31 + 12*jj + 384*ib], rhs2, 16);
                        suml_32_8 = __riscv_vwmacc_vx_i16m2(suml_32_8, a_ptr[ibl].qs[32 + 12*jj + 384*ib], rhs2, 16);
                        suml_32_9 = __riscv_vwmacc_vx_i16m2(suml_32_9, a_ptr[ibl].qs[33 + 12*jj + 384*ib], rhs2, 16);
                        suml_32_10 = __riscv_vwmacc_vx_i16m2(suml_32_10, a_ptr[ibl].qs[34 + 12*jj + 384*ib], rhs2, 16);
                        suml_32_11 = __riscv_vwmacc_vx_i16m2(suml_32_11, a_ptr[ibl].qs[35 + 12*jj + 384*ib], rhs2, 16);

                        suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[ibl].qs[36 + 12*jj + 384*ib], rhs3, 16);
                        suml_32_1 = __riscv_vwmacc_vx_i16m2(suml_32_1, a_ptr[ibl].qs[37 + 12*jj + 384*ib], rhs3, 16);
                        suml_32_2 = __riscv_vwmacc_vx_i16m2(suml_32_2, a_ptr[ibl].qs[38 + 12*jj + 384*ib], rhs3, 16);
                        suml_32_3 = __riscv_vwmacc_vx_i16m2(suml_32_3, a_ptr[ibl].qs[39 + 12*jj + 384*ib], rhs3, 16);
                        suml_32_4 = __riscv_vwmacc_vx_i16m2(suml_32_4, a_ptr[ibl].qs[40 + 12*jj + 384*ib], rhs3, 16);
                        suml_32_5 = __riscv_vwmacc_vx_i16m2(suml_32_5, a_ptr[ibl].qs[41 + 12*jj + 384*ib], rhs3, 16);
                        suml_32_6 = __riscv_vwmacc_vx_i16m2(suml_32_6, a_ptr[ibl].qs[42 + 12*jj + 384*ib], rhs3, 16);
                        suml_32_7 = __riscv_vwmacc_vx_i16m2(suml_32_7, a_ptr[ibl].qs[43 + 12*jj + 384*ib], rhs3, 16);
                        suml_32_8 = __riscv_vwmacc_vx_i16m2(suml_32_8, a_ptr[ibl].qs[44 + 12*jj + 384*ib], rhs3, 16);
                        suml_32_9 = __riscv_vwmacc_vx_i16m2(suml_32_9, a_ptr[ibl].qs[45 + 12*jj + 384*ib], rhs3, 16);
                        suml_32_10 = __riscv_vwmacc_vx_i16m2(suml_32_10, a_ptr[ibl].qs[46 + 12*jj + 384*ib], rhs3, 16);
                        suml_32_11 = __riscv_vwmacc_vx_i16m2(suml_32_11, a_ptr[ibl].qs[47 + 12*jj + 384*ib], rhs3, 16);
                    }

                    vint32m4_t suml_0 = __riscv_vmv_v_x_i32m4(0, 16);
                    vint32m4_t suml_1 = __riscv_vmv_v_x_i32m4(0, 16);
                    vint32m4_t suml_2 = __riscv_vmv_v_x_i32m4(0, 16);
                    vint32m4_t suml_3 = __riscv_vmv_v_x_i32m4(0, 16);
                    vint32m4_t suml_4 = __riscv_vmv_v_x_i32m4(0, 16);
                    vint32m4_t suml_5 = __riscv_vmv_v_x_i32m4(0, 16);
                    vint32m4_t suml_6 = __riscv_vmv_v_x_i32m4(0, 16);
                    vint32m4_t suml_7 = __riscv_vmv_v_x_i32m4(0, 16);
                    vint32m4_t suml_8 = __riscv_vmv_v_x_i32m4(0, 16);
                    vint32m4_t suml_9 = __riscv_vmv_v_x_i32m4(0, 16);
                    vint32m4_t suml_10 = __riscv_vmv_v_x_i32m4(0, 16);
                    vint32m4_t suml_11 = __riscv_vmv_v_x_i32m4(0, 16);
                    suml_0 = __riscv_vwadd_wv_i32m4(suml_0, suml_32_0, 16);
                    suml_1 = __riscv_vwadd_wv_i32m4(suml_1, suml_32_1, 16);
                    suml_2 = __riscv_vwadd_wv_i32m4(suml_2, suml_32_2, 16);
                    suml_3 = __riscv_vwadd_wv_i32m4(suml_3, suml_32_3, 16);
                    suml_4 = __riscv_vwadd_wv_i32m4(suml_4, suml_32_4, 16);
                    suml_5 = __riscv_vwadd_wv_i32m4(suml_5, suml_32_5, 16);
                    suml_6 = __riscv_vwadd_wv_i32m4(suml_6, suml_32_6, 16);
                    suml_7 = __riscv_vwadd_wv_i32m4(suml_7, suml_32_7, 16);
                    suml_8 = __riscv_vwadd_wv_i32m4(suml_8, suml_32_8, 16);
                    suml_9 = __riscv_vwadd_wv_i32m4(suml_9, suml_32_9, 16);
                    suml_10 = __riscv_vwadd_wv_i32m4(suml_10, suml_32_10, 16);
                    suml_11 = __riscv_vwadd_wv_i32m4(suml_11, suml_32_11, 16);

                    vint32m4_t scales = __riscv_vadd_vv_i32m4(sums, __riscv_vmv_v_x_i32m4(1, 16), 16);
                    __riscv_vse32_v_i32m4 (sum_row0, __riscv_vmacc_vv_i32m4(__riscv_vle32_v_i32m4(sum_row0, 16), scales, suml_0, 16), 16);
                    __riscv_vse32_v_i32m4 (sum_row0 + 16, __riscv_vmacc_vv_i32m4(__riscv_vle32_v_i32m4(sum_row0 + 16, 16), scales, suml_1, 16), 16);
                    __riscv_vse32_v_i32m4 (sum_row0 + 32, __riscv_vmacc_vv_i32m4(__riscv_vle32_v_i32m4(sum_row0 + 32, 16), scales, suml_2, 16), 16);
                    __riscv_vse32_v_i32m4 (sum_row0 + 48, __riscv_vmacc_vv_i32m4(__riscv_vle32_v_i32m4(sum_row0 + 48, 16), scales, suml_3, 16), 16);
                    __riscv_vse32_v_i32m4 (sum_row0 + 64, __riscv_vmacc_vv_i32m4(__riscv_vle32_v_i32m4(sum_row0 + 64, 16), scales, suml_4, 16), 16);
                    __riscv_vse32_v_i32m4 (sum_row0 + 80, __riscv_vmacc_vv_i32m4(__riscv_vle32_v_i32m4(sum_row0 + 80, 16), scales, suml_5, 16), 16);
                    __riscv_vse32_v_i32m4 (sum_row0 + 96, __riscv_vmacc_vv_i32m4(__riscv_vle32_v_i32m4(sum_row0 + 96, 16), scales, suml_6, 16), 16);
                    __riscv_vse32_v_i32m4 (sum_row0 + 112, __riscv_vmacc_vv_i32m4(__riscv_vle32_v_i32m4(sum_row0 + 112, 16), scales, suml_7, 16), 16);
                    __riscv_vse32_v_i32m4 (sum_row0 + 128, __riscv_vmacc_vv_i32m4(__riscv_vle32_v_i32m4(sum_row0 + 128, 16), scales, suml_8, 16), 16);
                    __riscv_vse32_v_i32m4 (sum_row0 + 144, __riscv_vmacc_vv_i32m4(__riscv_vle32_v_i32m4(sum_row0 + 144, 16), scales, suml_9, 16), 16);
                    __riscv_vse32_v_i32m4 (sum_row0 + 160, __riscv_vmacc_vv_i32m4(__riscv_vle32_v_i32m4(sum_row0 + 160, 16), scales, suml_10, 16), 16);
                    __riscv_vse32_v_i32m4 (sum_row0 + 176, __riscv_vmacc_vv_i32m4(__riscv_vle32_v_i32m4(sum_row0 + 176, 16), scales, suml_11, 16), 16);

                }
                __riscv_vse32_v_f32m4 (sum_row, __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row, 16), __riscv_vfmul_vv_f32m4(d16, __riscv_vfmv_v_f_f32m4(a_ptr[ibl].d[0], 16), 16), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_row0, 16), 16), 16), 16);
                __riscv_vse32_v_f32m4 (sum_row + 16, __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + 16, 16), __riscv_vfmul_vv_f32m4(d16, __riscv_vfmv_v_f_f32m4(a_ptr[ibl].d[1], 16), 16), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_row0 + 16, 16), 16), 16), 16);
                __riscv_vse32_v_f32m4 (sum_row + 32, __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + 32, 16), __riscv_vfmul_vv_f32m4(d16, __riscv_vfmv_v_f_f32m4(a_ptr[ibl].d[2], 16), 16), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_row0 + 32, 16), 16), 16), 16);
                __riscv_vse32_v_f32m4 (sum_row + 48, __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + 48, 16), __riscv_vfmul_vv_f32m4(d16, __riscv_vfmv_v_f_f32m4(a_ptr[ibl].d[3], 16), 16), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_row0 + 48, 16), 16), 16), 16);
                __riscv_vse32_v_f32m4 (sum_row + 64, __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + 64, 16), __riscv_vfmul_vv_f32m4(d16, __riscv_vfmv_v_f_f32m4(a_ptr[ibl].d[4], 16), 16), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_row0 + 64, 16), 16), 16), 16);
                __riscv_vse32_v_f32m4 (sum_row + 80, __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + 80, 16), __riscv_vfmul_vv_f32m4(d16, __riscv_vfmv_v_f_f32m4(a_ptr[ibl].d[5], 16), 16), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_row0 + 80, 16), 16), 16), 16);
                __riscv_vse32_v_f32m4 (sum_row + 96, __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + 96, 16), __riscv_vfmul_vv_f32m4(d16, __riscv_vfmv_v_f_f32m4(a_ptr[ibl].d[6], 16), 16), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_row0 + 96, 16), 16), 16), 16);
                __riscv_vse32_v_f32m4 (sum_row + 112, __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + 112, 16), __riscv_vfmul_vv_f32m4(d16, __riscv_vfmv_v_f_f32m4(a_ptr[ibl].d[7], 16), 16), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_row0 + 112, 16), 16), 16), 16);
                __riscv_vse32_v_f32m4 (sum_row + 128, __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + 128, 16), __riscv_vfmul_vv_f32m4(d16, __riscv_vfmv_v_f_f32m4(a_ptr[ibl].d[8], 16), 16), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_row0 + 128, 16), 16), 16), 16);
                __riscv_vse32_v_f32m4 (sum_row + 144, __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + 144, 16), __riscv_vfmul_vv_f32m4(d16, __riscv_vfmv_v_f_f32m4(a_ptr[ibl].d[9], 16), 16), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_row0 + 144, 16), 16), 16), 16);
                __riscv_vse32_v_f32m4 (sum_row + 160, __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + 160, 16), __riscv_vfmul_vv_f32m4(d16, __riscv_vfmv_v_f_f32m4(a_ptr[ibl].d[10], 16), 16), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_row0 + 160, 16), 16), 16), 16);
                __riscv_vse32_v_f32m4 (sum_row + 176, __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + 176, 16), __riscv_vfmul_vv_f32m4(d16, __riscv_vfmv_v_f_f32m4(a_ptr[ibl].d[11], 16), 16), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_row0 + 176, 16), 16), 16), 16);
                
            }
            {
                __riscv_vse32_v_f32m4 (sum_row, __riscv_vfmul_vv_f32m4(__riscv_vfmv_v_f_f32m4(0.125f, 16), __riscv_vle32_v_f32m4(sum_row, 16), 16), 16);
                __riscv_vse32_v_f32m4 (sum_row + 16, __riscv_vfmul_vv_f32m4(__riscv_vfmv_v_f_f32m4(0.125f, 16), __riscv_vle32_v_f32m4(sum_row + 16, 16), 16), 16);
                __riscv_vse32_v_f32m4 (sum_row + 32, __riscv_vfmul_vv_f32m4(__riscv_vfmv_v_f_f32m4(0.125f, 16), __riscv_vle32_v_f32m4(sum_row + 32, 16), 16), 16);
                __riscv_vse32_v_f32m4 (sum_row + 48, __riscv_vfmul_vv_f32m4(__riscv_vfmv_v_f_f32m4(0.125f, 16), __riscv_vle32_v_f32m4(sum_row + 48, 16), 16), 16);
                __riscv_vse32_v_f32m4 (sum_row + 64, __riscv_vfmul_vv_f32m4(__riscv_vfmv_v_f_f32m4(0.125f, 16), __riscv_vle32_v_f32m4(sum_row + 64, 16), 16), 16);
                __riscv_vse32_v_f32m4 (sum_row + 80, __riscv_vfmul_vv_f32m4(__riscv_vfmv_v_f_f32m4(0.125f, 16), __riscv_vle32_v_f32m4(sum_row + 80, 16), 16), 16);
                __riscv_vse32_v_f32m4 (sum_row + 96, __riscv_vfmul_vv_f32m4(__riscv_vfmv_v_f_f32m4(0.125f, 16), __riscv_vle32_v_f32m4(sum_row + 96, 16), 16), 16);
                __riscv_vse32_v_f32m4 (sum_row + 112, __riscv_vfmul_vv_f32m4(__riscv_vfmv_v_f_f32m4(0.125f, 16), __riscv_vle32_v_f32m4(sum_row + 112, 16), 16), 16);
                __riscv_vse32_v_f32m4 (sum_row + 128, __riscv_vfmul_vv_f32m4(__riscv_vfmv_v_f_f32m4(0.125f, 16), __riscv_vle32_v_f32m4(sum_row + 128, 16), 16), 16);
                __riscv_vse32_v_f32m4 (sum_row + 144, __riscv_vfmul_vv_f32m4(__riscv_vfmv_v_f_f32m4(0.125f, 16), __riscv_vle32_v_f32m4(sum_row + 144, 16), 16), 16);
                __riscv_vse32_v_f32m4 (sum_row + 160, __riscv_vfmul_vv_f32m4(__riscv_vfmv_v_f_f32m4(0.125f, 16), __riscv_vle32_v_f32m4(sum_row + 160, 16), 16), 16);
                __riscv_vse32_v_f32m4 (sum_row + 176, __riscv_vfmul_vv_f32m4(__riscv_vfmv_v_f_f32m4(0.125f, 16), __riscv_vle32_v_f32m4(sum_row + 176, 16), 16), 16);

                __riscv_vse32_v_f32m4(&s[(y * 12 + 0) * bs + x * ncols_interleaved], __riscv_vle32_v_f32m4(sum_row, 16), 16);
                __riscv_vse32_v_f32m4(&s[(y * 12 + 1) * bs + x * ncols_interleaved], __riscv_vle32_v_f32m4(sum_row + 16, 16), 16);
                __riscv_vse32_v_f32m4(&s[(y * 12 + 2) * bs + x * ncols_interleaved], __riscv_vle32_v_f32m4(sum_row + 32, 16), 16);
                __riscv_vse32_v_f32m4(&s[(y * 12 + 3) * bs + x * ncols_interleaved], __riscv_vle32_v_f32m4(sum_row + 48, 16), 16);
                __riscv_vse32_v_f32m4(&s[(y * 12 + 4) * bs + x * ncols_interleaved], __riscv_vle32_v_f32m4(sum_row + 64, 16), 16);
                __riscv_vse32_v_f32m4(&s[(y * 12 + 5) * bs + x * ncols_interleaved], __riscv_vle32_v_f32m4(sum_row + 80, 16), 16);
                __riscv_vse32_v_f32m4(&s[(y * 12 + 6) * bs + x * ncols_interleaved], __riscv_vle32_v_f32m4(sum_row + 96, 16), 16);
                __riscv_vse32_v_f32m4(&s[(y * 12 + 7) * bs + x * ncols_interleaved], __riscv_vle32_v_f32m4(sum_row + 112, 16), 16);
                __riscv_vse32_v_f32m4(&s[(y * 12 + 8) * bs + x * ncols_interleaved], __riscv_vle32_v_f32m4(sum_row + 128, 16), 16);
                __riscv_vse32_v_f32m4(&s[(y * 12 + 9) * bs + x * ncols_interleaved], __riscv_vle32_v_f32m4(sum_row + 144, 16), 16);
                __riscv_vse32_v_f32m4(&s[(y * 12 + 10) * bs + x * ncols_interleaved], __riscv_vle32_v_f32m4(sum_row + 160, 16), 16);
                __riscv_vse32_v_f32m4(&s[(y * 12 + 11) * bs + x * ncols_interleaved], __riscv_vle32_v_f32m4(sum_row + 176, 16), 16);

            }
        }

    }

    for (int y = anr/4; y < 4; y ++) { 
        const block_q8_Kx4 * a_ptr = (const block_q8_Kx4 *) vy + (y * nb);

        for(int x = 0; x < nc / nrr; x++){
            float sum_row[nrr*4] = {0.0};
            const block_iq2_xxsx16 * b_ptr = (const block_iq2_xxsx16 *) vx + (x * nb);
            
            for (int ibl = 0; ibl < nb  ; ++ibl) {
                int32_t sum_row0[nrr*4] = {0};
                vfloat32m4_t d16 = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vreinterpret_v_u16m2_f16m2(__riscv_vle16_v_u16m2(b_ptr[ibl].d, 16)), 16);
                auto qs = b_ptr[ibl].qs;
                
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    vint32m4_t sums = __riscv_vmv_v_x_i32m4(0, 16);
                    for(int ibk = 0; ibk < 4; ibk++){
                        vuint8m1_t sas = __riscv_vle8_v_u8m1(b_ptr[ibl].sas + 64*ib + 16*ibk, 16);
                        // uint32_t pattern_val = 0x10080402;
                        
                        for(int i=0; i < 16; i+=4){
                            pattern_array[i] = 2;
                            pattern_array[i + 1] = 4;
                            pattern_array[i + 2] = 8;
                            pattern_array[i + 3] = 16;
                        }
                        vint16m2_t prod16 = __riscv_vwmul_vv_i16m2(__riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vv_u8m1(sas, __riscv_vmv_v_x_u8m1(1, 16), 16)), __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vle8_v_u8m1(pattern_array, 16)), 16);
                        


                        vint16m2_t masks0 = __riscv_vle16_v_i16m2(base_data0, 16);
                        masks0 = __riscv_vand_vv_i16m2(prod16, masks0, 16);
                        vint32m1_t sum16_0 = __riscv_vwredsum_vs_i16m2_i32m1(masks0, vzero, 16);
                        sums = __riscv_vslide1down_vx_i32m4(sums,__riscv_vmv_x_s_i32m1_i32(sum16_0), 16);

                        masks0 = __riscv_vle16_v_i16m2(base_data0 + 16, 16);
                        masks0 = __riscv_vand_vv_i16m2(prod16, masks0, 16);
                        sum16_0 = __riscv_vwredsum_vs_i16m2_i32m1(masks0, vzero, 16);
                        sums = __riscv_vslide1down_vx_i32m4(sums,__riscv_vmv_x_s_i32m1_i32(sum16_0), 16);

                        masks0 = __riscv_vle16_v_i16m2(base_data0 + 32, 16);
                        masks0 = __riscv_vand_vv_i16m2(prod16, masks0, 16);
                        sum16_0 = __riscv_vwredsum_vs_i16m2_i32m1(masks0, vzero, 16);
                        sums = __riscv_vslide1down_vx_i32m4(sums,__riscv_vmv_x_s_i32m1_i32(sum16_0), 16);

                        masks0 = __riscv_vle16_v_i16m2(base_data0 + 48, 16);
                        masks0 = __riscv_vand_vv_i16m2(prod16, masks0, 16);
                        sum16_0 = __riscv_vwredsum_vs_i16m2_i32m1(masks0, vzero, 16);
                        sums = __riscv_vslide1down_vx_i32m4(sums,__riscv_vmv_x_s_i32m1_i32(sum16_0), 16);

                        vuint8m1_t signs128 = __riscv_vand_vx_u8m1(sas, 0xFE, 16);
                        signs128 = __riscv_vxor_vv_u8m1(signs128, __riscv_vsrl_vx_u8m1(signs128, 1, 16), 16);
                        
                        vuint8m1_t shuffle;
                        for( int i = 0; i < 8; ++i){
                            pattern_array[i] = 0;
                            pattern_array[i + 8] = 1;
                        }
                        shuffle = __riscv_vle8_v_u8m1(pattern_array, 16);
                        for (int i = 0; i < 16; i+=8) {
                            pattern_array[i]     = 1;
                            pattern_array[i+1]   = 2;
                            pattern_array[i+2]   = 4;
                            pattern_array[i+3]   = 8;
                            pattern_array[i+4]   = 16;
                            pattern_array[i+5]   = 32;
                            pattern_array[i+6]   = 64;
                            pattern_array[i+7]   = 128;
                        }
                        vuint8m1_t smask = __riscv_vle8_v_u8m1(pattern_array, 16);
                        
                        
                        vuint8m1_t m1 = __riscv_vmv_v_x_u8m1(1, 16);
                        vuint8m1_t step  = __riscv_vmv_v_x_u8m1(2, 16);
                        vint8m1_t qx0,qx1,qx2,qx3,qx4,qx5,qx6,qx7;
                        uint8_t qx_tmp0[16];
                        memcpy(qx_tmp0 + 0, &iq2xxs_grid[qs[0 + 16*ibk]], 8);
                        memcpy(qx_tmp0 + 8, &iq2xxs_grid[qs[1 + 16*ibk]], 8);
                        qx0 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vle8_v_u8m1(qx_tmp0, 16));
                        qx0 = __riscv_vmul_vv_i8m1(qx0, __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vor_vv_u8m1(__riscv_vmerge_vxm_u8m1(__riscv_vmv_v_x_u8m1(0, 16), 0xFF, __riscv_vmseq_vv_u8m1_b8(__riscv_vand_vv_u8m1(__riscv_vrgather_vv_u8m1(signs128, shuffle, 16), smask, 16), smask, 16), 16), m1, 16)), 16);
                        shuffle = __riscv_vadd_vv_u8m1(shuffle, step, 16);

                        //uint8_t qx_tmp1[32];
                        memcpy(qx_tmp0 + 0, &iq2xxs_grid[qs[2 + 16*ibk]], 8);
                        memcpy(qx_tmp0 + 8, &iq2xxs_grid[qs[3 + 16*ibk]], 8);
                        qx1 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vle8_v_u8m1(qx_tmp0, 16));
                        qx1 = __riscv_vmul_vv_i8m1(qx1, __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vor_vv_u8m1(__riscv_vmerge_vxm_u8m1(__riscv_vmv_v_x_u8m1(0, 16), 0xFF, __riscv_vmseq_vv_u8m1_b8(__riscv_vand_vv_u8m1(__riscv_vrgather_vv_u8m1(signs128, shuffle, 16), smask, 16), smask, 16), 16), m1, 16)), 16);
                        shuffle = __riscv_vadd_vv_u8m1(shuffle, step, 16);

                        // uint8_t qx_tmp2[32];
                        memcpy(qx_tmp0 + 0, &iq2xxs_grid[qs[4 + 16*ibk]], 8);
                        memcpy(qx_tmp0 + 8, &iq2xxs_grid[qs[5 + 16*ibk]], 8);
                        qx2 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vle8_v_u8m1(qx_tmp0, 16));
                        qx2 = __riscv_vmul_vv_i8m1(qx2, __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vor_vv_u8m1(__riscv_vmerge_vxm_u8m1(__riscv_vmv_v_x_u8m1(0, 16), 0xFF, __riscv_vmseq_vv_u8m1_b8(__riscv_vand_vv_u8m1(__riscv_vrgather_vv_u8m1(signs128, shuffle, 16), smask, 16), smask, 16), 16), m1, 16)), 16);
                        shuffle = __riscv_vadd_vv_u8m1(shuffle, step, 16);

                        // uint8_t qx_tmp3[32];
                        memcpy(qx_tmp0 + 0, &iq2xxs_grid[qs[6 + 16*ibk]], 8);
                        memcpy(qx_tmp0 + 8, &iq2xxs_grid[qs[7 + 16*ibk]], 8);
                        qx3 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vle8_v_u8m1(qx_tmp0, 16));
                        qx3 = __riscv_vmul_vv_i8m1(qx3, __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vor_vv_u8m1(__riscv_vmerge_vxm_u8m1(__riscv_vmv_v_x_u8m1(0, 16), 0xFF, __riscv_vmseq_vv_u8m1_b8(__riscv_vand_vv_u8m1(__riscv_vrgather_vv_u8m1(signs128, shuffle, 16), smask, 16), smask, 16), 16), m1, 16)), 16);
                        shuffle = __riscv_vadd_vv_u8m1(shuffle, step, 16);

                        memcpy(qx_tmp0 + 0, &iq2xxs_grid[qs[8 + 16*ibk]], 8);
                        memcpy(qx_tmp0 + 8, &iq2xxs_grid[qs[9 + 16*ibk]], 8);
                        qx4 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vle8_v_u8m1(qx_tmp0, 16));
                        qx4 = __riscv_vmul_vv_i8m1(qx4, __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vor_vv_u8m1(__riscv_vmerge_vxm_u8m1(__riscv_vmv_v_x_u8m1(0, 16), 0xFF, __riscv_vmseq_vv_u8m1_b8(__riscv_vand_vv_u8m1(__riscv_vrgather_vv_u8m1(signs128, shuffle, 16), smask, 16), smask, 16), 16), m1, 16)), 16);
                        shuffle = __riscv_vadd_vv_u8m1(shuffle, step, 16);

                        memcpy(qx_tmp0 + 0, &iq2xxs_grid[qs[10 + 16*ibk]], 8);
                        memcpy(qx_tmp0 + 8, &iq2xxs_grid[qs[11 + 16*ibk]], 8);
                        qx5 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vle8_v_u8m1(qx_tmp0, 16));
                        qx5 = __riscv_vmul_vv_i8m1(qx5, __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vor_vv_u8m1(__riscv_vmerge_vxm_u8m1(__riscv_vmv_v_x_u8m1(0, 16), 0xFF, __riscv_vmseq_vv_u8m1_b8(__riscv_vand_vv_u8m1(__riscv_vrgather_vv_u8m1(signs128, shuffle, 16), smask, 16), smask, 16), 16), m1, 16)), 16);
                        shuffle = __riscv_vadd_vv_u8m1(shuffle, step, 16);

                        memcpy(qx_tmp0 + 0, &iq2xxs_grid[qs[12 + 16*ibk]], 8);
                        memcpy(qx_tmp0 + 8, &iq2xxs_grid[qs[13 + 16*ibk]], 8);
                        qx6 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vle8_v_u8m1(qx_tmp0, 16));
                        qx6 = __riscv_vmul_vv_i8m1(qx6, __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vor_vv_u8m1(__riscv_vmerge_vxm_u8m1(__riscv_vmv_v_x_u8m1(0, 16), 0xFF, __riscv_vmseq_vv_u8m1_b8(__riscv_vand_vv_u8m1(__riscv_vrgather_vv_u8m1(signs128, shuffle, 16), smask, 16), smask, 16), 16), m1, 16)), 16);
                        shuffle = __riscv_vadd_vv_u8m1(shuffle, step, 16);

                        memcpy(qx_tmp0 + 0, &iq2xxs_grid[qs[14 + 16*ibk]], 8);
                        memcpy(qx_tmp0 + 8, &iq2xxs_grid[qs[15 + 16*ibk]], 8);
                        qx7 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vle8_v_u8m1(qx_tmp0, 16));
                        qx7 = __riscv_vmul_vv_i8m1(qx7, __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vor_vv_u8m1(__riscv_vmerge_vxm_u8m1(__riscv_vmv_v_x_u8m1(0, 16), 0xFF, __riscv_vmseq_vv_u8m1_b8(__riscv_vand_vv_u8m1(__riscv_vrgather_vv_u8m1(signs128, shuffle, 16), smask, 16), smask, 16), 16), m1, 16)), 16);

                        __riscv_vse8_v_i8m1(qx_temp + 0 + 128*ibk,qx0,16);
                        __riscv_vse8_v_i8m1(qx_temp + 16 + 128*ibk,qx1,16);
                        __riscv_vse8_v_i8m1(qx_temp + 32 + 128*ibk,qx2,16);
                        __riscv_vse8_v_i8m1(qx_temp + 48 + 128*ibk,qx3,16);
                        __riscv_vse8_v_i8m1(qx_temp + 64 + 128*ibk,qx4,16);
                        __riscv_vse8_v_i8m1(qx_temp + 80 + 128*ibk,qx5,16);
                        __riscv_vse8_v_i8m1(qx_temp + 96 + 128*ibk,qx6,16);
                        __riscv_vse8_v_i8m1(qx_temp + 112 + 128*ibk,qx7,16);

                    }
                    qs += 64;


                    vint16m2_t suml_32_0 = __riscv_vmv_v_x_i16m2(0, 16);
                    vint16m2_t suml_32_1 = __riscv_vmv_v_x_i16m2(0, 16);
                    vint16m2_t suml_32_2 = __riscv_vmv_v_x_i16m2(0, 16);
                    vint16m2_t suml_32_3 = __riscv_vmv_v_x_i16m2(0, 16);

                    vint8m1x4_t rhs;
                    vint8m1_t rhs0, rhs1, rhs2, rhs3;
                    for(int jj=0;jj<32;jj+=4)
                    {
                        rhs = __riscv_vlsseg4e8_v_i8m1x4(qx_temp + jj, 32, 16);
                        rhs0 = __riscv_vget_i8m1(rhs, 0);
                        rhs1 = __riscv_vget_i8m1(rhs, 1);
                        rhs2 = __riscv_vget_i8m1(rhs, 2);
                        rhs3 = __riscv_vget_i8m1(rhs, 3);

                        
                        suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[ibl].qs[0 + 4*jj + 128*ib], rhs0, 16);
                        suml_32_1 = __riscv_vwmacc_vx_i16m2(suml_32_1, a_ptr[ibl].qs[1 + 4*jj + 128*ib], rhs0, 16);
                        suml_32_2 = __riscv_vwmacc_vx_i16m2(suml_32_2, a_ptr[ibl].qs[2 + 4*jj + 128*ib], rhs0, 16);
                        suml_32_3 = __riscv_vwmacc_vx_i16m2(suml_32_3, a_ptr[ibl].qs[3 + 4*jj + 128*ib], rhs0, 16);

                        suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[ibl].qs[4 + 4*jj + 128*ib], rhs1, 16);
                        suml_32_1 = __riscv_vwmacc_vx_i16m2(suml_32_1, a_ptr[ibl].qs[5 + 4*jj + 128*ib], rhs1, 16);
                        suml_32_2 = __riscv_vwmacc_vx_i16m2(suml_32_2, a_ptr[ibl].qs[6 + 4*jj + 128*ib], rhs1, 16);
                        suml_32_3 = __riscv_vwmacc_vx_i16m2(suml_32_3, a_ptr[ibl].qs[7 + 4*jj + 128*ib], rhs1, 16);

                        suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[ibl].qs[8 + 4*jj + 128*ib], rhs2, 16);
                        suml_32_1 = __riscv_vwmacc_vx_i16m2(suml_32_1, a_ptr[ibl].qs[9 + 4*jj + 128*ib], rhs2, 16);
                        suml_32_2 = __riscv_vwmacc_vx_i16m2(suml_32_2, a_ptr[ibl].qs[10 + 4*jj + 128*ib], rhs2, 16);
                        suml_32_3 = __riscv_vwmacc_vx_i16m2(suml_32_3, a_ptr[ibl].qs[11 + 4*jj + 128*ib], rhs2, 16);

                        suml_32_0 = __riscv_vwmacc_vx_i16m2(suml_32_0, a_ptr[ibl].qs[12 + 4*jj + 128*ib], rhs3, 16);
                        suml_32_1 = __riscv_vwmacc_vx_i16m2(suml_32_1, a_ptr[ibl].qs[13 + 4*jj + 128*ib], rhs3, 16);
                        suml_32_2 = __riscv_vwmacc_vx_i16m2(suml_32_2, a_ptr[ibl].qs[14 + 4*jj + 128*ib], rhs3, 16);
                        suml_32_3 = __riscv_vwmacc_vx_i16m2(suml_32_3, a_ptr[ibl].qs[15 + 4*jj + 128*ib], rhs3, 16);
                    }

                    vint32m4_t suml_0 = __riscv_vmv_v_x_i32m4(0, 16);
                    vint32m4_t suml_1 = __riscv_vmv_v_x_i32m4(0, 16);
                    vint32m4_t suml_2 = __riscv_vmv_v_x_i32m4(0, 16);
                    vint32m4_t suml_3 = __riscv_vmv_v_x_i32m4(0, 16);
                    suml_0 = __riscv_vwadd_wv_i32m4(suml_0, suml_32_0, 16);
                    suml_1 = __riscv_vwadd_wv_i32m4(suml_1, suml_32_1, 16);
                    suml_2 = __riscv_vwadd_wv_i32m4(suml_2, suml_32_2, 16);
                    suml_3 = __riscv_vwadd_wv_i32m4(suml_3, suml_32_3, 16);

                    vint32m4_t scales = __riscv_vadd_vv_i32m4(sums, __riscv_vmv_v_x_i32m4(1, 16), 16);
                    __riscv_vse32_v_i32m4 (sum_row0, __riscv_vmacc_vv_i32m4(__riscv_vle32_v_i32m4(sum_row0, 16), scales, suml_0, 16), 16);
                    __riscv_vse32_v_i32m4 (sum_row0 + 16, __riscv_vmacc_vv_i32m4(__riscv_vle32_v_i32m4(sum_row0 + 16, 16), scales, suml_1, 16), 16);
                    __riscv_vse32_v_i32m4 (sum_row0 + 32, __riscv_vmacc_vv_i32m4(__riscv_vle32_v_i32m4(sum_row0 + 32, 16), scales, suml_2, 16), 16);
                    __riscv_vse32_v_i32m4 (sum_row0 + 48, __riscv_vmacc_vv_i32m4(__riscv_vle32_v_i32m4(sum_row0 + 48, 16), scales, suml_3, 16), 16);

                    
                }

                __riscv_vse32_v_f32m4 (sum_row, __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row, 16), __riscv_vfmul_vv_f32m4(d16, __riscv_vfmv_v_f_f32m4(a_ptr[ibl].d[0], 16), 16), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_row0, 16), 16), 16), 16);
                __riscv_vse32_v_f32m4 (sum_row + 16, __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + 16, 16), __riscv_vfmul_vv_f32m4(d16, __riscv_vfmv_v_f_f32m4(a_ptr[ibl].d[1], 16), 16), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_row0 + 16, 16), 16), 16), 16);
                __riscv_vse32_v_f32m4 (sum_row + 32, __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + 32, 16), __riscv_vfmul_vv_f32m4(d16, __riscv_vfmv_v_f_f32m4(a_ptr[ibl].d[2], 16), 16), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_row0 + 32, 16), 16), 16), 16);
                __riscv_vse32_v_f32m4 (sum_row + 48, __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sum_row + 48, 16), __riscv_vfmul_vv_f32m4(d16, __riscv_vfmv_v_f_f32m4(a_ptr[ibl].d[3], 16), 16), __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(sum_row0 + 48, 16), 16), 16), 16);

            }

            __riscv_vse32_v_f32m4 (sum_row, __riscv_vfmul_vv_f32m4(__riscv_vfmv_v_f_f32m4(0.125f, 16), __riscv_vle32_v_f32m4(sum_row, 16), 16), 16);
            __riscv_vse32_v_f32m4 (sum_row + 16, __riscv_vfmul_vv_f32m4(__riscv_vfmv_v_f_f32m4(0.125f, 16), __riscv_vle32_v_f32m4(sum_row + 16, 16), 16), 16);
            __riscv_vse32_v_f32m4 (sum_row + 32, __riscv_vfmul_vv_f32m4(__riscv_vfmv_v_f_f32m4(0.125f, 16), __riscv_vle32_v_f32m4(sum_row + 32, 16), 16), 16);
            __riscv_vse32_v_f32m4 (sum_row + 48, __riscv_vfmul_vv_f32m4(__riscv_vfmv_v_f_f32m4(0.125f, 16), __riscv_vle32_v_f32m4(sum_row + 48, 16), 16), 16);

            __riscv_vse32_v_f32m4(&s[(y * 4 + 0) * bs + x * ncols_interleaved], __riscv_vle32_v_f32m4(sum_row, 16), 16);
            __riscv_vse32_v_f32m4(&s[(y * 4 + 1) * bs + x * ncols_interleaved], __riscv_vle32_v_f32m4(sum_row + 16, 16), 16);
            __riscv_vse32_v_f32m4(&s[(y * 4 + 2) * bs + x * ncols_interleaved], __riscv_vle32_v_f32m4(sum_row + 32, 16), 16);
            __riscv_vse32_v_f32m4(&s[(y * 4 + 3) * bs + x * ncols_interleaved], __riscv_vle32_v_f32m4(sum_row + 48, 16), 16);

        }

    }
}