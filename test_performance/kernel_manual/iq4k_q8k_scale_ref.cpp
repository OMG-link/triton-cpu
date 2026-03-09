typedef uint16_t ggml_half;

#define GGML_TABLE_BEGIN(type, name, size) static const type name[size] = {
#define GGML_TABLE_END() };


GGML_TABLE_BEGIN(int8_t, iq4k_values, 32)
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    -123, -100, -79, -61, -45, -31, -18,  -6, 5, 17, 29, 42, 57, 73, 93, 117
GGML_TABLE_END()

typedef struct {
    ggml_half d;
    uint16_t extra; // 2 bit per SUB_QK_K(32) elements, total 256 elements, 256 / 32 * 2 bit / 8 = 2 bytes
    uint8_t  scales_h[QK_K/64]; //  2 bits per scale，16 个元素一组  256 / 16 * 2 bit / 8 = 256 / 64
    uint8_t  scales_l[QK_K/32]; //  4 bits per scale，16 个元素一组  256 / 16 * 4 bit / 8 = 256 / 32
    uint8_t  qs[QK_K/2];
} block_iq4_k;

typedef struct {
    float   d;              // delta
    float   sum;            // sum of quants in the entire block
    int8_t  qs[QK_K];       // quants
    int16_t bsums[QK_K/16]; // sum of quants in groups of 16
} block_q8_K;
static_assert(sizeof(block_q8_K) == 2*sizeof(float) + QK_K + QK_K/16*sizeof(int16_t), "wrong q8_K block size/padding");

// TODO 参考
void vec_dot_iq4_k_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ4_K, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif

    const int nb = n / QK_K;

    const block_iq4_k * x = (const block_iq4_k *)vx;
    const block_q8_K  * y = (const block_q8_K *)vy;

    float sumf = 0;
    for (int ibl = 0; ibl < nb; ++ibl) { // 256 超块
        const float d4d8 = GGML_FP16_TO_FP32(x[ibl].d) * y[ibl].d;
        uint16_t extra = x[ibl].extra;
        uint32_t h = *((const uint32_t *)x[ibl].scales_h);
        const uint8_t * qs = x[ibl].qs;
        const int8_t  * q8 = y[ibl].qs;
        int32_t sum = 0;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            const int ls1 = ((x[ibl].scales_l[ib] & 0xf) | ((h << 4) & 0x30)) - 32;
            const int ls2 = ((x[ibl].scales_l[ib] >>  4) | ((h << 2) & 0x30)) - 32;
            h >>= 4;
            // 确定是低 16 个值的 value 表 还是 高 16 个值的 value 表
            const int8_t * values1 = iq4k_values + 16*(extra & 1);
            const int8_t * values2 = iq4k_values +  8*(extra & 2);
            extra >>= 2;
            int sumi1 = 0, sumi2 = 0;
            for (int j = 0; j < 16; ++j) {
                sumi1 += q8[j+ 0] * values1[qs[j] & 0xf]; // int32
                sumi2 += q8[j+16] * values2[qs[j] >>  4]; // int32
            }
            sum += ls1*sumi1 + ls2*sumi2;
            qs += 16;
            q8 += 32;
        }
        sumf += d4d8 * sum;
    }
    *s = sumf;

}