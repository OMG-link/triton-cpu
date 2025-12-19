#include <cstdio>

#include "ggml_def.h"
#include "perf.h"
#include "timer.hpp"

const int VL = 32;

#define FREQ 1.6
#define VLEN 256
#define ELEM_WID 16

using InBlock = block_q8_0x12;
using KerBlock = block_q4_0x32;

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

    
    const size_t bs = n;

    assert(k % QK4_0 == 0);
    assert(m % 12 == 0);
    assert(n % VL == 0);

    const int nb = k / QK4_0;
    const int nTile = n / VL;
    const int mTile = m / 12; 

    // 1) output
    float *s = static_cast<float *>(calloc(m * bs, sizeof(float)));

    // 2) Q4 weights
    size_t n_blocks_vx = nb * nTile; // 
    KerBlock *vx = static_cast<KerBlock *>(calloc(n_blocks_vx, sizeof(block_q4_0x32)));

    // 3) Q8 activations
    size_t n_blocks_vy = nb * mTile;
    InBlock *vy = static_cast<InBlock *>(calloc(n_blocks_vy, sizeof(block_q8_0x12)));

    // times test should be repeated:
    int peak_fops = FREQ * VLEN / ELEM_WID;
    int T = 3 > ((1e9 * peak_fops) / (m * n * k)) ? 3 : ((1e9 * peak_fops) / (m * n * k)) ;


    // Warmup
    ggml_gemm_q4_0_12x32_q8_0(k, s, bs, vx, vy, m, n);

    // Perf-setup
    int fd_cycles = perf_event_cycles();
    int fd_l1da = perf_event_l1d_access();
    int fd_l1dm = perf_event_l1d_miss();
    perf_reset(fd_cycles);
    perf_reset(fd_l1da);
    perf_reset(fd_l1dm);

    // Main test
    for (int t = 0; t < T; t++) {
        ggml_gemm_q4_0_12x32_q8_0(k, s, bs, vx, vy, m, n);
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
    printf("cycle_per_test: %ld, cycle_total: %ld, T: %d\n", cycle_per_test, cycle_total, T);

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
