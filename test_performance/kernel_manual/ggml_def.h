#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <riscv_vector.h>

#define QK_K 256
#define QK_SB_K 32

#ifdef __cplusplus
// restrict not standard in C++
#if defined(__GNUC__)
#define GGML_RESTRICT __restrict__
#elif defined(__clang__)
#define GGML_RESTRICT __restrict
#elif defined(_MSC_VER)
#define GGML_RESTRICT __restrict
#else
#define GGML_RESTRICT
#endif
#else
#if defined(_MSC_VER) && (__STDC_VERSION__ < 201112L)
#define GGML_RESTRICT __restrict
#else
#define GGML_RESTRICT restrict
#endif
#endif

#define UNUSED(x) (void)(x)

// 自定义带消息的断言宏
#ifndef NDEBUG
    #define ASSERT_MSG(cond, fmt, ...) \
        do { \
            if (!(cond)) { \
                fprintf(stderr, "[ASSERT] %s:%d: " fmt "\n", \
                        __FILE__, __LINE__, ##__VA_ARGS__); \
                fprintf(stderr, "  Failed: %s\n", #cond); \
                abort(); \
            } \
        } while(0)
#else
    #define ASSERT_MSG(cond, fmt, ...) ((void)0)
#endif

// Legacy for RVV 0.7.1
#define __riscv_vzext_vf2_u16m2(x, vl) __riscv_vwaddu_vx_u16m2((x), 0, (vl))

typedef uint16_t ggml_half;

template <int VL> struct block_q4_Kx {
    ggml_half d[VL];                           // super-block scale for quantized scales
    ggml_half dmin[VL];                        // super-block scale for quantized mins
    uint8_t scales[VL * (QK_K / QK_SB_K) * 2]; // scales and mins, quantized with 6 bits, but stored with 8 bits
    uint8_t qs[VL * QK_K / 2];                 // 4--bit quants
};

template <int VL> struct block_q8_Kx {
    float d[VL];                         // delta
    uint8_t qs[VL * QK_K];               // quants
    uint16_t bsums[VL * QK_K / QK_SB_K]; // sum of quants in groups of 32
};



#define QK8_0 32
typedef struct {
    ggml_half d;       // delta
    int8_t  qs[QK8_0]; // quants
} block_q8_0;


// K 量化后 quant 位数 
// N 寄存器分块 nr 大小 
template <int K, int N> struct block {
    ggml_half d[N];                         // deltas for N qK_0 blocks
    int8_t    qs[(QK8_0 * N * K) / 8];         // quants for N qK_0 blocks : sizeof(qs) = 512 bytes
};

using block_q4_0x32 = block<4, 32>;

#define GGML_UNUSED(x) (void)(x)

void ggml_gemm_q4_K_8x32_q8_K(int n, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx, const void *GGML_RESTRICT vy, int nr, int nc);

void ggml_gemv_q4_0_8x32_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);