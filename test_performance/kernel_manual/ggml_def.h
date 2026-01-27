#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <riscv_vector.h>

#ifdef _MSC_VER
#define GGML_EXTENSION
#else // _MSC_VER
#define GGML_EXTENSION __extension__
#endif // _MSC_VER

#define QK_K 256
#define QK_SB_K 32
#define QK4_0 32
#define K_SCALE_SIZE 12

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
typedef uint32_t ggml_half2;

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

// This is only used for intermediate quantization and dot products
typedef struct {
    float   d;              // delta
    int8_t  qs[QK_K];       // quants
    int16_t bsums[QK_K/16]; // sum of quants in groups of 16
} block_q8_K;

#define GGML_COMMON_AGGR_U
#define GGML_COMMON_AGGR_S

// 4-bit quantization
// 8 blocks of 32 elements each
// weight is represented as x = a * q + b
// Effectively 4.5 bits per weight
typedef struct {
    GGML_EXTENSION union {
        struct {
            ggml_half d;    // super-block scale for quantized scales
            ggml_half dmin; // super-block scale for quantized mins
        } GGML_COMMON_AGGR_S;
        ggml_half2 dm;
    } GGML_COMMON_AGGR_U;
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];           // 4--bit quants
} block_q4_K;


#define QK4_0 32
typedef struct {
    ggml_half d;           // delta
    uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0;

#define QK8_0 32
typedef struct {
    ggml_half d;       // delta
    int8_t  qs[QK8_0]; // quants
} block_q8_0;

#define GGML_FP16_TO_FP32(x) float(x)
#define GGML_CPU_FP16_TO_FP32(x) float(x)


// K 量化后 quant 位数 
// N 寄存器分块 nr 大小 
template <int K, int N> struct alignas(8)  block {
    ggml_half d[N];                         // deltas for N qK_0 blocks
    uint8_t    qs[(QK8_0 * N * K) / 8];         // quants for N qK_0 blocks : sizeof(qs) = 512 bytes
};

using block_q4_0x32 = block<4, 32>;
using block_q8_0x12 = block<8, 12>;
using block_q8_0x8 = block<8, 8>;
using block_q8_0x4  = block<8, 4>;

#define GGML_UNUSED(x) (void)(x)

void ggml_gemm_q4_K_8x32_q8_K(int n, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx, const void *GGML_RESTRICT vy, int nr, int nc);

void ggml_gemm_q4_0_8x32_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);

void ggml_gemm_iq4_K_12x32_q8_K(int k, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT iq4k, const void *GGML_RESTRICT q8k, int m, int n);

void ggml_vec_dot_q4_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void ggml_vec_dot_q4_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);