import torch
import time
import numpy as np
import os, sys
import triton
import triton.language as tl


BLOCK_SIZE_M = 1
BLOCK_SIZE_N = 512
USE_GPU = False

# load 指令全部删掉
# 

@triton.jit
def q40_q80_gemv_kernel(
    q8_0_vector_ptr,  # int8 
    q8_0_scale_ptr,   # uint16 
    q4_0_matrix_ptr,  # int4 packed in uint8 
    q4_0_scale_ptr,   # uint16 
    output_ptr,       # float32 
    K,
    N,
    batch: tl.constexpr = 1,
):
    pid_b = tl.program_id(axis=0)  # batch id
    start_n = tl.program_id(axis=1)  # N id

    QK_8_0 = tl.constexpr(32)
    QK_8_0_DATA_SIZE = tl.constexpr(QK_8_0 // 2)  # 每个 uint8 包含两个 int4
    NR = tl.constexpr(32)
    N_block = tl.constexpr(K // QK_8_0)

    # 增加 batch 维度偏移
    batch_q8_offset = pid_b * K
    batch_q8_scale_offset = pid_b * (K // QK_8_0)
    batch_q4_matrix_offset = pid_b * N * (K // 2)
    batch_q4_scale_offset = pid_b * N * (K // 32)
    batch_output_offset = pid_b * N

    q4_0_matrix_offset = batch_q4_matrix_offset + start_n * N_block * (QK_8_0_DATA_SIZE * NR)
    q4_0_scales_offset = batch_q4_scale_offset + start_n * N_block * NR
    # float result_row[N]
    sumK_32_scale_f32 = tl.zeros((32,), dtype=tl.float32)
    result_row_offset = batch_output_offset + start_n * NR
    
    for n_block in range(N_block):
        q4_0_block_offset = q4_0_matrix_offset + n_block * (QK_8_0_DATA_SIZE * NR)
        q8_0_block_offset = batch_q8_offset + n_block * QK_8_0

        sumK_32 = tl.zeros((32,), dtype=tl.int32)  # k 维度 32 次累加 
        sumK_16 = tl.zeros((32,), dtype=tl.int16) # k 维度 16 次累加 

        for k in range(0, QK_8_0_DATA_SIZE // 2): # [0, 8]
            q4_0_data = tl.load(q4_0_matrix_ptr + q4_0_block_offset + k * NR + tl.arange(0, 32))

            q4_low_4bits = (q4_0_data & 0xF).cast(tl.int16)
            q4_high_4bits = (q4_0_data >> 4).cast(tl.int16)

            q8_0_data = tl.load(q8_0_vector_ptr + q8_0_block_offset + k).cast(tl.int16)
            sumK_16 += (q8_0_data[:] * q4_low_4bits[:])
            q8_0_data = tl.load(q8_0_vector_ptr + q8_0_block_offset + k + 1).cast(tl.int16)
            sumK_16 += (q8_0_data[:] * q4_high_4bits[:])
        sumK_32 += sumK_16
        
        
        
        sumK_16 = tl.zeros((32,), dtype=tl.int16) # k 维度 16 次累加
        for k in range(8, QK_8_0_DATA_SIZE): # [0, 8]
            q4_0_data = tl.load(q4_0_matrix_ptr + q4_0_block_offset + k * NR + tl.arange(0, 32))
            q4_low_4bits = (q4_0_data & 0xF).cast(tl.int16)
            q4_high_4bits = (q4_0_data >> 4).cast(tl.int16)

            q8_0_data = tl.load(q8_0_vector_ptr + q8_0_block_offset + k).cast(tl.int16)
            sumK_16 += (q8_0_data[:] * q4_low_4bits[:])
            q8_0_data = tl.load(q8_0_vector_ptr + q8_0_block_offset + k + 1).cast(tl.int16)
            sumK_16 += (q8_0_data[:] * q4_high_4bits[:])
        sumK_32 += sumK_16

        # dequant
        # weight_scale_data
        q4_0_scales_block_offset = q4_0_scales_offset + n_block * NR
        q8_0_scale_offset = batch_q8_scale_offset + n_block

        q4_scale_data = tl.load(q4_0_scale_ptr + q4_0_scales_block_offset + tl.arange(0, 32))
        q8_scale_data = tl.load(q8_0_scale_ptr + q8_0_scale_offset)
        q8_scale_f32 = q8_scale_data.cast(tl.float32)
        q4_scale_f32 = q4_scale_data.cast(tl.float32)
        scale = q8_scale_f32[:] * q4_scale_f32[:]
        sumK_32_scale_f32 += sumK_32.cast(tl.float32) * scale

    tl.store(output_ptr + result_row_offset + tl.arange(0, 32), sumK_32_scale_f32)


############################################################
# 统一驱动支持: 若存在 gemv_driver 则使用其中的带宽测试
############################################################
if os.path.dirname(__file__) not in sys.path:
    sys.path.append(os.path.dirname(__file__))
try:
    import bench_gemv_driver as gemv_driver  # 提供 run_bandwidth
except ImportError:
    gemv_driver = None  # 允许单独脚本存在


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='GEMV Q4_0 x Q8_0 带宽性能测试')
    parser.add_argument('--n', type=int, default=512, help='输出维度 (必须是32的倍数)')
    parser.add_argument('--k', type=int, default=1024, help='输入维度 (必须是32的倍数)')
    parser.add_argument('--batch', type=int, default=1, help='batch 维度')
    parser.add_argument('--rounds', type=int, default=5, help='测试轮数')
    parser.add_argument('--warmup', type=int, default=3, help='预热轮数')
    parser.add_argument('--target-gb', type=float, default=2.0, help='目标数据总量 (GB)')
    parser.add_argument('--simple-test', action='store_true', help='运行简单功能测试')
    parser.add_argument("--num_threads", type=int, default=4, help="设置triton内核的线程数")
    
    args = parser.parse_args()
    
    if args.simple_test:
        # 简单功能测试
        print("运行简单功能测试...")
        K = 1024
        N = 512
        batch = args.batch
        NR = 32

        q8_0_vector = torch.randint(-128, 127, (batch, K), dtype=torch.int8)
        q8_0_scale = torch.randint(1, 255, (batch, K // 32), dtype=torch.uint16)
        q4_0_matrix = torch.randint(0, 255, (batch, N * (K // 2)), dtype=torch.uint8)
        q4_0_scale = torch.randint(1, 255, (batch, N * (K // 32)), dtype=torch.uint16)
        output = torch.zeros((batch, N), dtype=torch.float32)

        grid = (batch, N // NR)
        q40_q80_gemv_kernel[grid](
            q8_0_vector_ptr=q8_0_vector,
            q8_0_scale_ptr=q8_0_scale,
            q4_0_matrix_ptr=q4_0_matrix,
            q4_0_scale_ptr=q4_0_scale,
            output_ptr=output,
            K=K,
            N=N,
            batch=batch,
            num_threads=args.num_threads,
        )
        print("简单测试完成")
        print(f"输出形状: {output.shape}")
        print(f"输出范围: [{output.min():.2f}, {output.max():.2f}]")
    else:
        if gemv_driver is None:
            raise RuntimeError("未找到 gemv_driver，无法执行统一带宽测试。请确保 gemv_driver.py 位于同目录。")
        gemv_driver.run_bandwidth(
            kernel_name='q40_q80',
            N=args.n,
            K=args.k,
            rounds=args.rounds,
            warmup=args.warmup,
            target_gb=args.target_gb,
            threads=args.num_threads,
        )

