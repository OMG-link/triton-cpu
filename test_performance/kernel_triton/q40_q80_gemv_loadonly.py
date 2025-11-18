import torch
import time
import numpy as np

import triton
import triton.language as tl


BLOCK_SIZE_M = 1
BLOCK_SIZE_N = 512
USE_GPU = False

# load

@triton.jit
def q40_q80_gemv_kernel(
    q8_0_vector_ptr,  # int8 
    q8_0_scale_ptr,   # uint16 
    q4_0_matrix_ptr,  # int4 packed in uint8 
    q4_0_scale_ptr,   # uint16 
    output_ptr,       # float32 
    K,
    N,
):
    QK_8_0 = tl.constexpr(32)
    QK_8_0_DATA_SIZE = tl.constexpr(QK_8_0 // 2)  # 每个 uint8 包含两个 int4
    NR = tl.constexpr(32)
    N_block = tl.constexpr(K // QK_8_0)

    start_n = tl.program_id(axis=0)

    q4_0_matrix_offset = start_n * N_block * (QK_8_0_DATA_SIZE * NR)
    q4_0_scales_offset = start_n * N_block * NR
    # float result_row[N]
    sumK_32_scale_f32 = tl.zeros((32,), dtype=tl.float32)
    result_row_offset = start_n * NR
    
    for n_block in range(N_block):
        q4_0_block_offset = q4_0_matrix_offset + n_block * (QK_8_0_DATA_SIZE * NR)
        q8_0_block_offset = n_block * QK_8_0 

        # sumK_32 = tl.zeros((32,), dtype=tl.int32)  # k 维度 32 次累加
        
        # sumK_16 = tl.zeros((32,), dtype=tl.int16) # k 维度 16 次累加
        for k in range(0, QK_8_0_DATA_SIZE // 2): # [0, 8]
            q4_0_data = tl.load(q4_0_matrix_ptr + q4_0_block_offset + k * NR + tl.arange(0, 32))
            # q4_low_4bits = (q4_0_data & 0xF).cast(tl.int8)
            # q4_high_4bits = (q4_0_data >> 4).cast(tl.int8)

            # q8_0_data = tl.load(q8_0_vector_ptr + q8_0_block_offset + k).cast(tl.int8)
            # sumK_16 += (q8_0_data[:] * q4_low_4bits[:])
            # q8_0_data = tl.load(q8_0_vector_ptr + q8_0_block_offset + k + 1).cast(tl.int8)
            # sumK_16 += (q8_0_data[:] * q4_high_4bits[:])
        # sumK_32 += sumK_16
        
        
        
        sumK_16 = tl.zeros((32,), dtype=tl.int16) # k 维度 16 次累加
        for k in range(8, QK_8_0_DATA_SIZE): # [0, 8]
            q4_0_data = tl.load(q4_0_matrix_ptr + q4_0_block_offset + k * NR + tl.arange(0, 32))
            # q4_low_4bits = (q4_0_data & 0xF).cast(tl.int8)
            # q4_high_4bits = (q4_0_data >> 4).cast(tl.int8)

            # q8_0_data = tl.load(q8_0_vector_ptr + q8_0_block_offset + k).cast(tl.int8)
            # sumK_16 += (q8_0_data[:] * q4_low_4bits[:])
            # q8_0_data = tl.load(q8_0_vector_ptr + q8_0_block_offset + k + 1).cast(tl.int8)
            # sumK_16 += (q8_0_data[:] * q4_high_4bits[:])
        # sumK_32 += sumK_16

        # dequant
        # weight_scale_data
        # q4_0_scales_block_offset = q4_0_scales_offset + n_block * NR
        # q8_0_scale_offset = n_block

        # q4_scale_data = tl.load(q4_0_scale_ptr + q4_0_scales_block_offset + tl.arange(0, 32))
        # q8_scale_data = tl.load(q8_0_scale_ptr + q8_0_scale_offset)
        # q8_scale_f32 = q8_scale_data.cast(tl.float32)
        # q4_scale_f32 = q4_scale_data.cast(tl.float32)
        # scale = q8_scale_f32[:] * q4_scale_f32[:]
        # sumK_32_scale_f32 += sumK_32.cast(tl.float32) * scale

    # tl.store(output_ptr + result_row_offset + tl.arange(0, 32), sumK_32_scale_f32)


def calculate_num_batches(n, k, target_gb=2.0):
    """计算需要准备的不同数据组数"""
    QK4_0 = 32
    # 计算每组数据的大小
    # q4_0_matrix: (n * k / 2) bytes
    # q4_0_scale: (n * k / 32) * 2 bytes (uint16)
    # q8_0_vector: k bytes
    # q8_0_scale: (k / 32) * 2 bytes (uint16)
    weight_bytes = n * (k // 2) + n * (k // 32) * 2
    activation_bytes = k + (k // 32) * 2
    total_bytes = weight_bytes + activation_bytes
    
    size_gb = total_bytes / (1024.0 ** 3)
    batches = int(target_gb / size_gb)
    return max(batches, 1)


def benchmark_bandwidth(n, k, rounds=5, warmup=3, target_gb=2.0):
    """
    带宽性能测试
    
    参数:
        n: 输出维度 (必须是32的倍数)
        k: 输入维度 (必须是32的倍数)
        rounds: 测试轮数
        warmup: 预热轮数
        target_gb: 目标数据总量 (GB)
    """
    assert n % 32 == 0, "N 必须是32的倍数"
    assert k % 32 == 0, "K 必须是32的倍数"
    
    NR = 32
    QK_8_0 = 32
    
    print("=" * 50)
    print("GEMV Q4_0 x Q8_0 带宽性能测试")
    print("=" * 50)
    print(f"矩阵形状: [1 x {k}] x [{k} x {n}]")
    print(f"测试轮数: {rounds}")
    print("=" * 50)
    print()
    
    # 计算需要准备的不同数据组数
    num_batches = calculate_num_batches(n, k, target_gb)
    print(f"准备 {num_batches} 组不同数据 (目标总量: ~{target_gb}GB)\n")
    
    # 初始化数据批次
    q8_0_vector_batches = []
    q8_0_scale_batches = []
    q4_0_matrix_batches = []
    q4_0_scale_batches = []
    
    print("初始化测试数据...")
    for b in range(num_batches):
        # 使用不同的随机种子
        torch.manual_seed(12345 + b)
        
        q8_0_vector = torch.randint(-128, 127, (k,), dtype=torch.int8)
        q8_0_scale = torch.randint(1, 255, (k // 32,), dtype=torch.uint16)
        q4_0_matrix = torch.randint(0, 255, (n * (k // 2),), dtype=torch.uint8)
        q4_0_scale = torch.randint(1, 255, (n * (k // 32),), dtype=torch.uint16)
        
        q8_0_vector_batches.append(q8_0_vector)
        q8_0_scale_batches.append(q8_0_scale)
        q4_0_matrix_batches.append(q4_0_matrix)
        q4_0_scale_batches.append(q4_0_scale)
    
    # 计算总数据量
    weight_mb = sum([(q4_0_matrix_batches[i].numel() * q4_0_matrix_batches[i].element_size() +
                      q4_0_scale_batches[i].numel() * q4_0_scale_batches[i].element_size()) 
                     for i in range(num_batches)]) / (1024.0 ** 2)
    activation_mb = sum([(q8_0_vector_batches[i].numel() * q8_0_vector_batches[i].element_size() +
                          q8_0_scale_batches[i].numel() * q8_0_scale_batches[i].element_size()) 
                         for i in range(num_batches)]) / (1024.0 ** 2)
    total_mb = weight_mb + activation_mb
    total_gb = total_mb / 1024.0
    
    print(f"数据初始化完成: {total_mb:.2f} MB ({total_gb:.2f} GB)")
    print(f"  权重: {weight_mb:.2f} MB")
    print(f"  激活: {activation_mb:.2f} MB\n")
    
    # 预热
    print(f"Warmup ({warmup} 轮)...")
    output = torch.zeros((n,), dtype=torch.float32)
    grid = (n // NR,)
    
    for _ in range(warmup):
        q40_q80_gemv_kernel[grid](
            q8_0_vector_ptr=q8_0_vector_batches[0],
            q8_0_scale_ptr=q8_0_scale_batches[0],
            q4_0_matrix_ptr=q4_0_matrix_batches[0],
            q4_0_scale_ptr=q4_0_scale_batches[0],
            output_ptr=output,
            K=k,
            N=n,
            num_threads=8,
        )
    print("完成\n")
    
    # 性能测试
    print("开始性能测试...")
    print("-" * 50)
    
    bandwidths = []
    latencies = []
    
    for round_idx in range(rounds): 
        start_time = time.perf_counter() 
        
        # batch = 2 # 测试多组数据
        for batch_idx in range(num_batches):
            # 现在多核时间跟手写汇编算子差别比较大
            # 启动开销 => 
            q40_q80_gemv_kernel[grid](
                q8_0_vector_ptr=q8_0_vector_batches[batch_idx],
                q8_0_scale_ptr=q8_0_scale_batches[batch_idx],
                q4_0_matrix_ptr=q4_0_matrix_batches[batch_idx],
                q4_0_scale_ptr=q4_0_scale_batches[batch_idx],
                output_ptr=output,
                K=k,
                N=n,
                num_threads=8,
            )
        
        end_time = time.perf_counter() 
        elapsed_s = end_time - start_time 
        elapsed_ms = elapsed_s * 1000.0 
        bandwidth = total_gb / elapsed_s 
        
        bandwidths.append(bandwidth)
        latencies.append(elapsed_ms)
        
        print(f"第 {round_idx + 1} 轮:")
        print(f"  时间: {elapsed_ms:.3f} ms | 带宽: {bandwidth:.2f} GB/s")
        print()
    
    # 统计结果
    avg_bandwidth = np.mean(bandwidths)
    std_bandwidth = np.std(bandwidths)
    min_bandwidth = np.min(bandwidths)
    max_bandwidth = np.max(bandwidths)
    avg_latency = np.mean(latencies)
    
    # 理论带宽（根据平台调整）
    theoretical_bw = 5.83  # GB/s，根据实际硬件调整 
    bandwidth_util = (avg_bandwidth / theoretical_bw) * 100.0 
    
    print("=" * 50)
    print("测试总结")
    print("=" * 50)
    print(f"平均带宽: {avg_bandwidth:.2f} GB/s (±{std_bandwidth:.2f})")
    print(f"带宽范围: [{min_bandwidth:.2f}, {max_bandwidth:.2f}] GB/s")
    print(f"平均延迟: {avg_latency:.3f} ms")
    print(f"理论带宽: {theoretical_bw:.2f} GB/s")
    print(f"带宽利用率: {bandwidth_util:.2f}%")
    print("=" * 50)
    
    return {
        'avg_bandwidth': avg_bandwidth,
        'std_bandwidth': std_bandwidth,
        'min_bandwidth': min_bandwidth,
        'max_bandwidth': max_bandwidth,
        'avg_latency': avg_latency,
        'bandwidth_util': bandwidth_util,
        'bandwidths': bandwidths,
        'latencies': latencies,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='GEMV Q4_0 x Q8_0 带宽性能测试')
    parser.add_argument('--n', type=int, default=512, help='输出维度 (必须是32的倍数)')
    parser.add_argument('--k', type=int, default=1024, help='输入维度 (必须是32的倍数)')
    parser.add_argument('--rounds', type=int, default=5, help='测试轮数')
    parser.add_argument('--warmup', type=int, default=3, help='预热轮数')
    parser.add_argument('--target-gb', type=float, default=2.0, help='目标数据总量 (GB)')
    parser.add_argument('--simple-test', action='store_true', help='运行简单功能测试')
    
    args = parser.parse_args()
    
    if args.simple_test:
        # 简单功能测试
        print("运行简单功能测试...")
        K = 1024
        N = 512
        NR = 32

        q8_0_vector = torch.randint(-128, 127, (K,), dtype=torch.int8)
        q8_0_scale = torch.randint(1, 255, (K // 32,), dtype=torch.uint16)
        q4_0_matrix = torch.randint(0, 255, (N * (K // 2),), dtype=torch.uint8)
        q4_0_scale = torch.randint(1, 255, (N * (K // 32),), dtype=torch.uint16)
        output = torch.zeros((N,), dtype=torch.float32)

        grid = (N // NR, )
        q40_q80_gemv_kernel[grid](
            q8_0_vector_ptr=q8_0_vector,
            q8_0_scale_ptr=q8_0_scale,
            q4_0_matrix_ptr=q4_0_matrix,
            q4_0_scale_ptr=q4_0_scale,
            output_ptr=output,
            K=K,
            N=N,
        )
        print("简单测试完成")
        print(f"输出形状: {output.shape}")
        print(f"输出范围: [{output.min():.2f}, {output.max():.2f}]")
    else:
        # 带宽性能测试
        benchmark_bandwidth(
            n=args.n,
            k=args.k,
            rounds=args.rounds,
            warmup=args.warmup,
            target_gb=args.target_gb
        )

