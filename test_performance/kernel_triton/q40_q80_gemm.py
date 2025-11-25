import torch
import time
import numpy as np

import triton
import triton.language as tl

'''
TRITON_ALWAYS_COMPILE=1 TRITON_CPU_BACKEND=1 python q40_q80_gemm.py
TRITON_ALWAYS_COMPILE=1 TRITON_CPU_BACKEND=1 python bench_driver.py --metric gflops --csv-out bench.csv --png-out bench.png
'''

@triton.jit
def q40_q80_gemm_kernel(
    q4_0_matrix_ptr,         # q4_0 矩阵指针, two int packed in uint8
    q4_0_scale_ptr,          # q4_0 量化比例指针, uint16
    q8_0_matrix_ptr,         # q8_0 矩阵指, int8
    q8_0_scale_ptr,          # q8_0 量化比例指针, uint16
    output_ptr,              # 输出指针，float
    M,                       # 矩阵行数
    N,                       # 矩阵列数
    K,                       # 矩阵公共维度，这里表示元素数量而不是字节数
):
    QK_8_0 : tl.constexpr = 32
    QK_4_0_DATA_SIZE : tl.constexpr = QK_8_0 // 2  # 每个 uint8 包含两个 q4_0 元素
    MR : tl.constexpr = 12
    NR : tl.constexpr = 32
    N_block : tl.constexpr = K // QK_8_0

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    q8_0_block_ptr = tl.make_block_ptr(
        base=q8_0_matrix_ptr,
        shape=(M//MR, K//QK_8_0, 2, QK_8_0//2, MR),
        strides=(K * MR, QK_8_0 * MR ,QK_8_0 * MR // 2, MR, 1),
        offsets=(pid_m,0,0, 0, 0),
        block_shape=(1,1,1, QK_8_0//2, MR),
        order=(4,3,2,1,0)
    )

    q4_0_block_ptr = tl.make_block_ptr(
        base=q4_0_matrix_ptr,
        shape=(N//NR, K//QK_8_0, QK_4_0_DATA_SIZE, NR),
        strides=(K * NR // 2, QK_4_0_DATA_SIZE * NR, NR, 1),
        offsets=(pid_n, 0, 0, 0),
        block_shape=(1,1,QK_4_0_DATA_SIZE, NR),
        order=(3,2,1,0)
    )

    q8_0_scale_block_ptr = tl.make_block_ptr(
        base=q8_0_scale_ptr,
        shape=(M//MR, K//QK_8_0, MR),
        strides=(K * MR // QK_8_0, MR, 1),
        offsets=(pid_m, 0, 0),
        block_shape=(1, 1, MR),
        order=(2, 1, 0)
    )

    q4_0_scale_block_ptr = tl.make_block_ptr(
        base=q4_0_scale_ptr,
        shape=(N//NR, K//QK_8_0, NR),
        strides=(K * NR // QK_8_0, NR, 1),
        offsets=(pid_n, 0, 0),
        block_shape=(1, 1, NR),
        order=(2, 1, 0)
    )

    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(M//MR, N//NR, MR, NR),
        strides=(N * MR, MR * NR, NR, 1),
        offsets=(pid_m, pid_n, 0, 0),
        block_shape=(1, 1, MR, NR),
        order=(3, 2, 1, 0)
    )

    # 精度对齐需求
    acc = tl.zeros((MR, NR), dtype=tl.float32)

    for k_block in range(N_block):
        sum_block = tl.zeros((MR, NR), dtype=tl.int32)
        
        # 加载 q4 数据和 scale
        q4_data_ptr = tl.advance(q4_0_block_ptr, offsets = (0, k_block, 0, 0))
        q4_data = tl.load(q4_data_ptr)  # [1, 1, QK_4_0_DATA_SIZE, NR]
        q4_data = q4_data.reshape(QK_4_0_DATA_SIZE, NR)  # [16, 32]
        
        q8_scale_ptr = tl.advance(q8_0_scale_block_ptr, offsets = (0, k_block, 0))
        q8_scale = tl.load(q8_scale_ptr)  # [1, 1, MR]
        q8_scale = q8_scale.reshape(MR)  # [12]
        
        q4_scale_ptr = tl.advance(q4_0_scale_block_ptr, offsets = (0, k_block, 0))
        q4_scale = tl.load(q4_scale_ptr)  # [1, 1, NR]
        q4_scale = q4_scale.reshape(NR)  # [32]

        # 解包 q4 数据
        q4_low_4bit = (q4_data & 0xf).cast(tl.int16)  # [16, 32]
        q4_high_4bit = (q4_data >> 4).cast(tl.int16)  # [16, 32]
        
        # 加载 q8 数据的低 16 位并计算
        q8_data_ptr_low = tl.advance(q8_0_block_ptr, offsets = (0, k_block, 0, 0, 0))
        q8_data_low = tl.load(q8_data_ptr_low)  # [1, 1, 1, 16, 12]
        q8_data_low = q8_data_low.reshape(16, MR).cast(tl.int16)  # [16, 12]
        tmp_int16 = tl.dot(q8_data_low.T, q4_low_4bit, out_dtype=tl.int16)  # [12, 32]
        sum_block += tmp_int16.cast(tl.int32)

        # 加载 q8 数据的高 16 位并计算
        q8_data_ptr_high = tl.advance(q8_0_block_ptr, offsets = (0, k_block, 1, 0, 0))
        q8_data_high = tl.load(q8_data_ptr_high)  # [1, 1, 1, 16, 12]
        q8_data_high = q8_data_high.reshape(16, MR).cast(tl.int16)  # [16, 12]
        tmp_int16 = tl.dot(q8_data_high.T, q4_high_4bit, out_dtype=tl.int16)  # [12, 32]
        sum_block += tmp_int16.cast(tl.int32)

        # 量化比例应用
        q8_scale_reshaped = tl.reshape(q8_scale, (MR, 1))  # [12, 1]
        q4_scale_reshaped = tl.reshape(q4_scale, (1, NR))  # [1, 32]
        scale = tl.dot(q8_scale_reshaped, q4_scale_reshaped, out_dtype=tl.float32)  # [12, 32]

        acc += sum_block.cast(tl.float32) * scale

    # 写回结果 - reshape to match block_shape
    acc_reshaped = acc.reshape(1, 1, MR, NR)
    output_ptr = tl.advance(output_block_ptr, offsets=(0, 0, 0, 0))
    tl.store(output_ptr, acc_reshaped)

# -----------------------
# 数据准备函数
# -----------------------

MR = 12
NR = 32
QK_8_0 = 32

def prepare_random_q40_q80_inputs(M, K, N):
    """构造随机但形状匹配的量化输入张量, 用于仅性能测试 (不保证数值语义正确)."""
    assert M % MR == 0 and N % NR == 0 and K % QK_8_0 == 0, "形状不满足块大小约束"
    Mb = M // MR
    Nb = N // NR
    Kblocks = K // QK_8_0
    QK_4_0_DATA_SIZE = QK_8_0 // 2

    device = 'cpu'

    # q8_0_matrix: shape (Mb, Kblocks, 2, QK_8_0//2, MR)
    q8_0_matrix = torch.randint(-128, 127, (Mb, Kblocks, 2, QK_8_0//2, MR), dtype=torch.int8, device=device)
    q8_0_scale = torch.rand((Mb, Kblocks, MR), dtype=torch.float32, device=device)

    # q4_0 packed: (Nb, Kblocks, QK_4_0_DATA_SIZE, NR) with uint8 storing two int4
    q4_0_matrix = torch.randint(0, 255, (Nb, Kblocks, QK_4_0_DATA_SIZE, NR), dtype=torch.uint8, device=device)
    q4_0_scale = torch.rand((Nb, Kblocks, NR), dtype=torch.float32, device=device)

    # output
    output = torch.empty((M, N), dtype=torch.float32, device=device)

    return q4_0_matrix, q4_0_scale, q8_0_matrix, q8_0_scale, output



# -----------------------
# 调试测试接口
# -----------------------

if __name__ == "__main__":
    # 简单的功能测试
    M = 12
    N = 32
    K = 32

    print(f"测试配置: M={M}, N={N}, K={K}")
    print(f"块大小约束: M%{MR}==0, N%{NR}==0, K%{QK_8_0}==0")
    
    q4_0_matrix, q4_0_scale, q8_0_matrix, q8_0_scale, output = prepare_random_q40_q80_inputs(M, K, N)
    
    print(f"\n输入张量形状:")
    print(f"  q4_0_matrix: {q4_0_matrix.shape} (dtype={q4_0_matrix.dtype})")
    print(f"  q4_0_scale: {q4_0_scale.shape} (dtype={q4_0_scale.dtype})")
    print(f"  q8_0_matrix: {q8_0_matrix.shape} (dtype={q8_0_matrix.dtype})")
    print(f"  q8_0_scale: {q8_0_scale.shape} (dtype={q8_0_scale.dtype})")
    print(f"  output: {output.shape} (dtype={output.dtype})")

    grid = (M//MR, N//NR)
    print(f"\nGrid配置: {grid}")
    
    print("\n执行 kernel...")
    q40_q80_gemm_kernel[grid](
        q4_0_matrix, q4_0_scale, q8_0_matrix, q8_0_scale,
        output, M, N, K
    )

    print(f"\n输出结果:")
    print(f"  Shape: {output.shape}")
    print(f"  Min: {output.min().item():.4f}")
    print(f"  Max: {output.max().item():.4f}")
    print(f"  Mean: {output.mean().item():.4f}")
    print(f"\n前3x3元素:")
    print(output[:3, :3])
    
    # 简单性能测试
    print("\n" + "="*60)
    print("性能测试")
    print("="*60)
    median_ms, min_ms, max_ms = bench_q40_q80_case(M, K, N, rep_ms=100, warmup_ms=20)
    print(f"时间 (中位数): {median_ms:.3f} ms")
    print(f"时间 (最小): {min_ms:.3f} ms")
    print(f"时间 (最大): {max_ms:.3f} ms")
    
    # 计算 GOPS
    total_ops = 2.0 * M * N * K
    gops = (total_ops / (median_ms * 1e-3)) * 1e-9
    print(f"性能: {gops:.2f} GOPS@int8")
    
    print("\n测试完成!")
