#include <cstdio>

#include "ggml_def.h"
#include "perf.h"
#include "timer.hpp"


const int VL = 32;


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

    const int nb = k / QK4_0;

    // 1) output
    float *s = static_cast<float *>(calloc(m * n, sizeof(float)));

    // 2) Q4 weights
    block_q4_0 *vx = static_cast<block_q4_0 *>(calloc(n * nb, sizeof(block_q4_0)));

    // 3) Q8 activations
    block_q8_0 *vy = static_cast<block_q8_0 *>(calloc(m * nb, sizeof(block_q8_0)));

    // times test should be repeated:
    int T = 50;

    // Warmup
    // ggml_gemm_q4_0_12x32_q8_0(k, s, bs, vx, vy, m, n);

    // Perf-setup
    int fd_cycles = perf_event_cycles();
    int fd_l1da = perf_event_l1d_access();
    int fd_l1dm = perf_event_l1d_miss();
    
    perf_reset(fd_cycles);
    perf_reset(fd_l1da);
    perf_reset(fd_l1dm);

    // Main test
    for(size_t i = 0; i < m; i++) { // M vy 激活
        for(size_t j = 0; j < n; j++) { // N vx 
                float result = 0;
                ggml_vec_dot_q4_0_q8_0(k, &result, n, vx + j * nb, 1, vy + i * nb, 1, 1);
                *(s + i * n + j) = result;
        }
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

    int64_t n_fma = static_cast<int64_t>(k) * m * n; // 考虑实际形状，为了避免 padding，这里用的整数倍的寄存器分块大小 
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
