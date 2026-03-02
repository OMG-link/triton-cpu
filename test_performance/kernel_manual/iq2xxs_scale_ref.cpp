#include "ggml_def.h"


void dequantize_row_iq2_xxs(const block_iq2_xxs * restrict x, float * restrict y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            memcpy(aux32, x[i].qs + 4*ib32, 2*sizeof(uint32_t));
            const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f; // 最高 4 位是 scale，还是有二级量化的 
            for (int l = 0; l < 4; ++l) { // 4 * 2^8 = 1024 
                const uint8_t * grid = (const uint8_t *)(iq2xxs_grid + aux8[l]); // 2^8 = 256 
                const uint8_t  signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127]; // 127 <=> 0111 1111 
                for (int j = 0; j < 8; ++j) {
                    // subblock scale
                    y[j] = db * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f); // 1 10 100 1000 10000 
                }
                y += 8;
            }
        }
    }
}


void ggml_vec_dot_iq2_xxs_q8_K(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq2_xxs * restrict x = vx;
    const block_q8_K    * restrict y = vy;

    const int nb = n / QK_K; // n /  256

    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) { // 超块
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t * restrict q2 = x[i].qs;
        const int8_t   * restrict q8 = y[i].qs;
        int32_t bsum = 0;
        // 8 个分组 256/32=8
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            memcpy(aux32, q2, 2*sizeof(uint32_t));
            q2 += 4; // 每次 32 个数， 32 * 2 bit = 64bit = 4 个 uint16
            const uint32_t ls = 2*(aux32[1] >> 28) + 1;
            int32_t sumi = 0;
            // 32 个数
            for (int l = 0; l < 4; ++l) {
                //  k 维度 8 个数为一个小分组，进行查表 
                const uint8_t * grid = (const uint8_t *)(iq2xxs_grid + aux8[l]);
                const uint8_t  signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
                for (int j = 0; j < 8; ++j) {
                    sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * ls;
        }
        sumf += d * bsum;
    }
    *s = 0.125f * sumf;

}
