#include <cstdio>

#include "ggml_def.h"
#include "perf.h"
#include "timer.hpp"

const int VL = 32;

using InBlock = block_q8_Kx<12>;
using KerBlock = block_q4_Kx<VL>;

// 分块参数由命令行决定 

int main(int argc, char **argv) {

    //  A [480][1536]
    //  B [1536][1536]

    // parse command line args M/N/K
    if (argc != 4) {
        printf("Usage: %s <M> <N> <K>\n", argv      [0]);
        return 1;
    }
    const int m = atoi(argv[1]);
    const int n = atoi(argv[2]);
    const int k = atoi(argv[3]);


    // M * K @int8： 0.7 MB 
    // N * K @int8: 2.25 MB 
    // L1 data cache: 32 KB  per core 
    // L2 cache: 512 KB 
    
    // 单核： 相比 细粒度融合 慢了 3%  why 这么少 ? 

    // for () B =>（1536 * 1536） int4/int2/gather 反量化 
        // dequant 访存（int4 => int8） 数据量 * 2 
        // GEMV 量化融合敏感  memory bound  =>  
        // GEMM 量化融合敏感较低 compute bound =>  找一些 memory bound 场景，多核 or M 小计算访存低 
    
    // for () => matmul （480 * 1536）* (1536 * 1536) 
        // cache 分块/ 寄存器分块 

    // for () => 后处理  


    // M 和 N 比较接近时，计算访存比较高效
    // MN / (M + N) = 12 * 128 / (12 + 128) = 10.769 
    // constexpr int k = 1536;
    // constexpr int m = 144;
    // constexpr int n = 128;

    // M 变大时，计算密度趋近于 128 
    // MN / (M + N) = 256 * 128 / (256 + 128) = 85.333 
    // constexpr int k = 1536; 
    // constexpr int m = 256; 
    // constexpr int n = 128; 


    // M << N, 计算访存比约等于 M 
    // MN / (M + N) = 12 * 1536 / (12 + 1536) = 11.706 
    // constexpr int k = 1536;
    // constexpr int m = 12;
    // constexpr int n = 1536;

    // MN / (M + N) = 256 * 1536 / (256 + 1536) = 219.43 
    // constexpr int k = 1536;
    // constexpr int m = 256;
    // constexpr int n = 1536;



    // MN / (M + N) = 12 * 8960 / (12 + 8960) = 11.98
    // constexpr int k = 1536;
    // constexpr int m = 12;
    // constexpr int n = 8960;

    // MN / (M + N) = 256 * 1536 / (256 + 8960) = 246.15
    // constexpr int k = 1536;
    // constexpr int m = 256;
    // constexpr int n = 8960;


    // M * K @int8： 0.7 MB 
    // N * K @int8: 2.25 MB 
    // L1 data cache: 32 KB  per core 
    // L2 cache: 512 KB 

    // constexpr int k = 512;
    // constexpr int m = 12;
    // constexpr int n = 32;
    // constexpr int k = 1024;
    // constexpr int m = 1024;
    // constexpr int n = 1024;
    
    const size_t bs = n;

    assert(k % QK_K == 0);
    assert(m % 12 == 0);
    assert(n % VL == 0);

    const int nb = k / QK_K;
    const int nTile = n / VL;
    const int mTile = m / 12; 

    // 1) output
    float *s = static_cast<float *>(calloc(m * bs, sizeof(float)));

    // 2) Q4 weights
    size_t n_blocks_vx = nb * nTile; // 
    KerBlock *vx = static_cast<KerBlock *>(calloc(n_blocks_vx, sizeof(*vx)));

    // 3) Q8 activations
    size_t n_blocks_vy = nb * mTile;
    InBlock *vy = static_cast<InBlock *>(calloc(n_blocks_vy, sizeof(*vy)));

    // times test should be repeated:
    int T = 50;

    // Warmup
    ggml_gemm_q4_K_8x32_q8_K(k, s, bs, vx, vy, m, n);

    // Perf-setup
    int fd_cycles = perf_event_cycles();
    int fd_l1da = perf_event_l1d_access();
    int fd_l1dm = perf_event_l1d_miss();
    perf_reset(fd_cycles);
    perf_reset(fd_l1da);
    perf_reset(fd_l1dm);

    // Main test
    for (int t = 0; t < T; t++) {
        ggml_gemm_q4_K_8x32_q8_K(k, s, bs, vx, vy, m, n);
    }

    // Perf-cleanup
    perf_disable(fd_cycles);
    perf_disable(fd_l1da);
    perf_disable(fd_l1dm);
    auto cycles = perf_read(fd_cycles);
    auto l1d_access = perf_read(fd_l1da);
    auto l1d_miss = perf_read(fd_l1dm);
    printf("cycle = %lu \t l1d_access = %lu \t l1d_miss = %lu \t miss_rate = %.2f%%\n", cycles, l1d_access, l1d_miss, (double)l1d_miss / l1d_access * 100);

    int64_t cycle_total = cycles;
    int64_t cycle_per_test = cycle_total / T;

    int64_t n_fma = static_cast<int64_t>(k) * mTile * nTile * 12 * VL; // 考虑实际形状，为了避免 padding，这里用的整数倍的寄存器分块大小 
    int64_t fma_per_cycle;

#ifdef SPACEMIT_X60
    fma_per_cycle = 16;
#else
#ifdef XUANTIE_C910
    fma_per_cycle = 16;
#else
#error "Unknown CPU. Cannot determine peak FMA"
#endif
#endif

    int64_t theoretical_cycle = n_fma / fma_per_cycle;
    int64_t actual_cycle = cycle_per_test;
    double utilization = static_cast<double>(actual_cycle) / theoretical_cycle;

    printf("乘加数: %zu\n", n_fma);
    printf("理论需要周期数: %ld\n", theoretical_cycle);
    printf("实际执行周期数: %ld\n", actual_cycle);
    printf("实际-理论比值: %.2f (%.2f%%)\n", utilization, 100 / utilization);

    // free(s);
    // free(vx);
    // free(vy);
}
