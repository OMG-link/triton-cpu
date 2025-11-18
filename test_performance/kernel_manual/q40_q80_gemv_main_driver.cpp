#include <cstdio>
#include <cstdlib>

#include "ggml_def.h"
#include "perf.h"
#include "timer.hpp"

const int QK4_0 = 32;

// 计算需要准备的不同数据组数
int calculate_num_batches(int n, int k, double target_gb = 2.0) {
    size_t weight_bytes = (n / QK4_0) * (k / QK4_0) * sizeof(block_q4_0x32);
    double size_gb = static_cast<double>(weight_bytes) / (1024.0 * 1024.0 * 1024.0);
    int batches = static_cast<int>(target_gb / size_gb);
    return batches > 0 ? batches : 1;
}

// 伪随机数生成器
static uint32_t xorshift32(uint32_t state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

int main(int argc, char **argv) {
    // 解析命令行参数 N K [ROUNDS]
    if (argc < 3) {
        printf("用法: %s <N> <K> [ROUNDS]\n", argv[0]);
        printf("  N: 输出维度 (必须是32的倍数)\n");
        printf("  K: 输入维度 (必须是32的倍数)\n");
        printf("  ROUNDS: 测试轮数 (可选，默认为5)\n");
        return 1;
    }
    
    const int n = atoi(argv[1]);
    const int k = atoi(argv[2]);
    const int rounds = (argc >= 4) ? atoi(argv[3]) : 5;
    
    // 参数验证
    if (n <= 0 || k <= 0 || n % 32 != 0 || k % 32 != 0) {
        printf("错误: N 和 K 必须是正数且是32的倍数\n");
        printf("当前值: N=%d, K=%d\n", n, k);
        return 1;
    }
    
    if (rounds <= 0) {
        printf("错误: ROUNDS 必须是正数\n");
        return 1;
    }
    
    printf("====================================\n");
    printf("GEMV Q4_0 x Q8_0 带宽性能测试\n");
    printf("====================================\n");
    printf("矩阵形状: [1 x %d] x [%d x %d]\n", k, k, n);
    printf("测试轮数: %d\n", rounds);
    printf("====================================\n\n");
    
    const int nb = k / QK4_0;
    const size_t bs = n;
    
    // 计算需要准备的不同数据组数
    int num_batches = calculate_num_batches(n, k, 2.0);
    printf("准备 %d 组不同数据 (目标总量: ~2GB)\n\n", num_batches);
    
    // 分配内存
    float *s = static_cast<float *>(calloc(n, sizeof(float)));
    size_t n_blocks_vx = (n / 32) * nb;
    size_t n_blocks_vy = nb;
    
    block_q4_0x32 **vx_batches = static_cast<block_q4_0x32 **>(malloc(num_batches * sizeof(block_q4_0x32*)));
    block_q8_0 **vy_batches = static_cast<block_q8_0 **>(malloc(num_batches * sizeof(block_q8_0*)));
    
    if (!s || !vx_batches || !vy_batches) {
        printf("错误: 内存分配失败\n");
        return 1;
    }
    
    for (int b = 0; b < num_batches; b++) {
        vx_batches[b] = static_cast<block_q4_0x32 *>(calloc(n_blocks_vx, sizeof(block_q4_0x32)));
        vy_batches[b] = static_cast<block_q8_0 *>(calloc(n_blocks_vy, sizeof(block_q8_0)));
        if (!vx_batches[b] || !vy_batches[b]) {
            printf("错误: 第 %d 组数据分配失败\n", b);
            return 1;
        }
    }
    
    // 初始化数据 - 每组使用不同的随机种子
    printf("初始化测试数据...\n");
    uint32_t rng_state = 12345;
    
    for (int b = 0; b < num_batches; b++) {
        rng_state = xorshift32(rng_state);
        uint32_t seed = rng_state;
        
        // 初始化权重
        for (size_t i = 0; i < n_blocks_vx; i++) {
            for (int j = 0; j < 32; j++) {
                seed = xorshift32(seed);
                vx_batches[b][i].d[j] = 0x3800 + (seed % 0x0800);
            }
            for (int j = 0; j < 512; j++) {
                seed = xorshift32(seed);
                vx_batches[b][i].qs[j] = (seed % 256);
            }
        }
        
        // 初始化激活
        for (size_t i = 0; i < n_blocks_vy; i++) {
            seed = xorshift32(seed);
            vy_batches[b][i].d = 0x3800 + (seed % 0x0800);
            for (int j = 0; j < QK4_0; j++) {
                seed = xorshift32(seed);
                vy_batches[b][i].qs[j] = static_cast<int8_t>((seed % 256) - 128);
            }
        }
    }
    
    double total_mb = num_batches * ((n_blocks_vx * sizeof(block_q4_0x32)) + 
                                      (n_blocks_vy * sizeof(block_q8_0))) / (1024.0 * 1024.0);
    printf("数据初始化完成: %.2f MB (%.2f GB)\n\n", total_mb, total_mb / 1024.0);
    
    // Warmup
    printf("Warmup...\n");
    for (int i = 0; i < 3; i++) {
        ggml_gemv_q4_0_8x32_q8_0(k, s, bs, vx_batches[0], vy_batches[0], 1, n);
    }
    printf("完成\n\n");
    
    // 性能测试
    int fd_cycles = perf_event_cycles();
    int fd_l1da = perf_event_l1d_access();
    int fd_l1dm = perf_event_l1d_miss();
    
    double total_bandwidth = 0.0;
    double total_time_ns = 0.0;
    
    printf("开始性能测试...\n");
    printf("------------------------------------\n");
    
#ifdef SPACEMIT_X60
    const double cpu_freq_ghz = 1.6;
    const double theoretical_bw = 6.76; // 8 threads 
#else
#ifdef XUANTIE_C910
    const double cpu_freq_ghz = 1.8;
    const double theoretical_bw = 8.0;
#else
    const double cpu_freq_ghz = 1.6;
    const double theoretical_bw = 8.0;
#endif
#endif
    
    double total_gb = total_mb / 1024.0;
    
    for (int round = 0; round < rounds; round++) {
        flush_l3_cache();
        
        perf_reset(fd_cycles);
        perf_reset(fd_l1da);
        perf_reset(fd_l1dm);
        
        Timer timer;
        timer.start();
        
        #pragma unroll 10
        for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
            ggml_gemv_q4_0_8x32_q8_0(k, s, bs, vx_batches[batch_idx], vy_batches[batch_idx], 1, n);
        }
        
        int64_t elapsed_cycles = timer.stop();
        
        perf_disable(fd_cycles);
        perf_disable(fd_l1da);
        perf_disable(fd_l1dm);
        
        uint64_t cycles = perf_read(fd_cycles);
        uint64_t l1d_access = perf_read(fd_l1da);
        uint64_t l1d_miss = perf_read(fd_l1dm);
        
        double time_ns = elapsed_cycles / cpu_freq_ghz;
        double time_s = time_ns / 1e9;
        double bandwidth = total_gb / time_s;
        double miss_rate = (l1d_access > 0) ? (double)l1d_miss / l1d_access * 100.0 : 0.0;
        
        printf("第 %d 轮:\n", round + 1);
        printf("  周期: %lu | 时间: %.3f ms | 带宽: %.2f GB/s\n", 
               cycles, time_s * 1000.0, bandwidth);
        printf("  L1D: 访问=%lu 缺失=%lu (%.2f%%)\n", l1d_access, l1d_miss, miss_rate);
        printf("\n");
        
        total_bandwidth += bandwidth;
        total_time_ns += time_ns;
    }
    
    perf_close_event(fd_cycles);
    perf_close_event(fd_l1da);
    perf_close_event(fd_l1dm);
    
    // 统计结果
    double avg_bandwidth = total_bandwidth / rounds;
    double avg_latency = (total_time_ns / rounds) / 1e6;
    double bandwidth_util = (avg_bandwidth / theoretical_bw) * 100.0;
    
    printf("====================================\n");
    printf("测试总结\n");
    printf("====================================\n");
    printf("平均带宽: %.2f GB/s\n", avg_bandwidth);
    printf("平均延迟: %.3f ms\n", avg_latency);
    printf("理论带宽: %.2f GB/s\n", theoretical_bw);
    printf("带宽利用率: %.2f%%\n", bandwidth_util);
    printf("====================================\n");
    
    // 释放内存
    free(s);
    for (int b = 0; b < num_batches; b++) {
        free(vx_batches[b]);
        free(vy_batches[b]);
    }
    free(vx_batches);
    free(vy_batches);
    
    return 0;
}
