typedef uint16_t ggml_half;
typedef uint32_t ggml_half2;

#ifdef _MSC_VER
#define GGML_EXTENSION
#else // _MSC_VER
#define GGML_EXTENSION __extension__
#endif // _MSC_VER

#define QK_K 256
#define K_SCALE_SIZE 12

typedef struct {
    GGML_EXTENSION union {
        struct {
            ggml_half d;    // super-block scale for quantized scales datatype: uint16_t 256 超块
            ggml_half dmin; // super-block scale for quantized mins datatype: uint16_t 256 超块
        } GGML_COMMON_AGGR_S;
        ggml_half2 dm;  // union 类型，翻遍取两个值 
    } GGML_COMMON_AGGR_U;
    uint8_t scales[K_SCALE_SIZE]; // 8组， 每组 32 K_SCALE_SIZE = 12.  8 * (6+6) bits / 8 = 12 bytes  => scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];           // 4--bit quants 
} block_q4_K;

// This is only used for intermediate quantization and dot products
// bsums: 16 groups of 16 quants summed together
typedef struct {
    float   d;              // delta
    int8_t  qs[QK_K];       // quants
    int16_t bsums[QK_K/16]; // 256/16 = 16, sum of quants in groups of 16 => 求和 16 组 sum 
} block_q8_K;


void ggml_vec_dot_q4_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    static_assert(sizeof(block_q4_K) == 2 * sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2, "wrong q4_K block size/padding");

    const block_q4_K * GGML_RESTRICT x = vx;  // 一个

    static_assert(sizeof(block_q8_K) == sizeof(float) + QK_K + QK_K/16 * sizeof(int16_t), "wrong q8_K block size/padding");

    const block_q8_K * GGML_RESTRICT y = vy;

    // block num 
    const int nb = n / QK_K;  // QK_K=256, which is the superblock size for q4_K and q8_K

    // 用于后续从紧凑存储的 scales 和 mins 数据中提取正确的值 
    // 这是一种高效的位操作技巧 
    static const uint32_t kmask1 = 0x3f3f3f3f; // 0x3f mask for 6 bits, 用作位掩码（bitmask）


    static const uint32_t kmask2 = 0x0f0f0f0f; // 0x0f mask for 4 bits 
    static const uint32_t kmask3 = 0x03030303; // 0x03 mask for 2 bits 

    uint32_t utmp[4]; // 16 int8， 8个scale， 8个min 原先是 12 字节

    const uint8_t * scales = (const uint8_t*)&utmp[0]; // 提取 sub_block 的 scale 
    const uint8_t * mins   = (const uint8_t*)&utmp[2]; // 提起 sub_block 的 min 

    int8_t  aux8[QK_K]; // int8_t aux8[256]，存放从 q4_K 块中反量化出的4位整数值 
    int16_t aux16[8];   // 存 int8 * int8，每个存 vec[4] * vec[4] 的结果，8个元素 
    float   sums [8];   // 
    int32_t aux32[8];   // 存 int8 * int8 * scale (权重的缩放系数) 的结果 

    memset(sums, 0, 8*sizeof(float));

    float sumf = 0; // fp32 存最终结果
    // 每次一个 superblock（256个元素），每个 superblock 包含 8 个 subblock，每个 subblock 包含 32 个元素
    for (int i = 0; i < nb; ++i) { 
        const uint8_t * GGML_RESTRICT q4 = x[i].qs; // 权重数据, uint8_t
        const  int8_t * GGML_RESTRICT q8 = y[i].qs; // 激活数据, int8
        memset(aux32, 0, 8*sizeof(int32_t)); 
        int8_t * GGML_RESTRICT a = aux8;
        
        // 反量化 q4 数据，提取出4位整数值，存放到 aux8 数组中 
        // 每个迭代处理一个超块（256个元素）256/64 = 4
        for (int j = 0; j < QK_K/64; ++j) { // 4，每个循环两个超块
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF); 
            a += 32;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4); 
            a += 32; q4 += 32;
        }

        // 得到 量化参数 scale/min ？
        memcpy(utmp, x[i].scales, 12); // 每个 subblock 的 scale 和 min，每个 6bit 
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);  // 6bit 存储方式 
        const uint32_t uaux = utmp[1] & kmask1; 
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4); // scale 
        utmp[2] = uaux; // min
        utmp[0] &= kmask1; // scale

        // 输入的 sum 
        int sumi = 0; 
        // 256/16 = 16 个 group，每个 group 16 个元素 
        // y[i].bsums[j] 是激活的 sum 
        for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2]; // 激活的 sum * (d_min) 权重的subblock 的 min
        
        // 
        a = aux8;


        int is = 0; 

        // 计算每个 subblock， 一共 8 个，每个 32 大小  
        // aux32[8] 输出，每个 
        for (int j = 0; j < QK_K/32; ++j) { 
            int32_t scale = scales[is++]; // sacle 
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];      // int16 = 激活int8 * 4位整数值(int8) => 用 FMA 向量化  
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l]; // int32 = 激活int8 * 4位整数值(int8) * scale  
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];      // int16 = 激活int8 * 4位整数值(int8) => 用 FMA 向量化 
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l]; // int32 = 激活int8 * 4位整数值(int8) * scale
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];      // int16 = 激活int8 * 4位整数值(int8) => 用 FMA 向量化  
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l]; // weight 的 scale 
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];      // int16 = 激活int8 * 4位整数值(int8) => 用 FMA 向量化 
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l]; 
            q8 += 8; a += 8;
        }
        
        // x[i].d 是 superblock 的 scale (subblock 共享) 
        // y[i].d 是激活的 scale，所有值共享 
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d; // d  = d_in * d_w
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l]; // 用 fp32 存二级量化
        const float dmin = GGML_CPU_FP16_TO_FP32(x[i].dmin) * y[i].d; // d[i].dmin 是 权重 superblock 的 min 的 scale 
        // y[i].d 是 激活的 scale 
        sumf -= dmin * sumi;  // 最终结果 fp32 
    }

    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}
