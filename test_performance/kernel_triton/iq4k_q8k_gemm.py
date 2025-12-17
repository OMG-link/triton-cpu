import torch
import time
import numpy as np

import triton
import triton.language as tl
import time

# typedef struct {
#     ggml_half d;  
#     uint16_t extra;
#     uint8_t  scales_h[QK_K/64];
#     uint8_t  scales_l[QK_K/32];
#     uint8_t  qs[QK_K/2];
# } block_iq4_k;

# This is only used for intermediate quantization and dot products
# typedef struct {
#     float   d;              // delta
#     float   sum;            // sum of quants in the entire block
#     int8_t  qs[QK_K];       // quants
#     int16_t bsums[QK_K/16]; // sum of quants in groups of 16
# } block_q8_K;

# IQ4K 查找表 - 有4个不同的表，通过 extra 字段选择 
table1 = tl.constexpr([ -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 ]) 
table2 = tl.constexpr([ -123, -100, -79, -61, -45, -31, -18, -6, 5, 17, 29, 42, 57, 73, 93, 117 ]) 

#  TODO 针对不同架构的调优系统（ARM/RISC-V）
#  TODO 端到端测试性能，准确性（精度）
@triton.jit
def iq4k_q8k_matmul_kernel(
    iq4k_matrix_ptr,         # iq4k 矩阵指针, two int packed in uint8
    iq4k_d_ptr,              # iq4k 量化比例指针, uint16 (存储的FP16)
    iq4k_extra_ptr,          # iq4k 额外信息指针, uint16, 256 个元素超块一个值
    iq4k_scale_l_ptr,        # iq4k 低位量化比例指针, uint8
    iq4k_scale_h_ptr,        # iq4k 高位量化比例指针, uint8 (实际是4个uint8打包成uint32)
    q8k_matrix_ptr,          # q8k 矩阵指, int8
    q8k_d_ptr,               # q8k 量化比例指针, float
    output_ptr,              # 输出指针，float
    M,                       # 矩阵行数
    N,                       # 矩阵列数
    K,                       # 矩阵公共维度，这里表示元素数量而不是字节数
):
    QK_K : tl.constexpr = 256
    QK_4_K_DATA_SIZE = QK_K // 2  # 整个超块的数据大小 (128) - 仅用于数据生成
    NUM_SUB_BLOCKS : tl.constexpr = QK_K // 32  # 每个块有8个子块
    SUB_BLOCK_SIZE : tl.constexpr = 32
    QK_4_K_SUB_BLOCK_DATA_SIZE : tl.constexpr = SUB_BLOCK_SIZE // 2  # 每个子块的数据大小 (16)
    
    MR : tl.constexpr = 12
    NR : tl.constexpr = 32

    N_block : tl.constexpr = K // QK_K

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    value_table1 = tl.arange(0, 16).cast(tl.int8).reshape(1, 16).broadcast_to((NR, 16))
    value_table2 = tl.arange(0, 16).cast(tl.int8).reshape(1, 16).broadcast_to((NR, 16))


    # Q8K 数据布局: [M//MR, K//QK_K, NUM_SUB_BLOCKS, SUB_BLOCK_SIZE, MR]
    q8k_block_ptr = tl.make_block_ptr(
        base=q8k_matrix_ptr,
        shape=(M//MR, K//QK_K, NUM_SUB_BLOCKS, 2, SUB_BLOCK_SIZE//2, MR),
        strides=(K * MR, QK_K * MR, SUB_BLOCK_SIZE * MR, SUB_BLOCK_SIZE * MR // 2, MR, 1),
        offsets=(pid_m, 0, 0, 0, 0, 0),
        block_shape=(1, 1, 1, 1, SUB_BLOCK_SIZE//2, MR),
        order=(5, 4, 3, 2, 1, 0)
    )

    q8k_d_block_ptr = tl.make_block_ptr(
        base=q8k_d_ptr,
        shape=(M//MR, K//QK_K, MR),
        strides=(K * MR // QK_K, MR, 1),
        offsets=(pid_m, 0, 0),
        block_shape=(1, 1, MR),
        order=(2, 1, 0)
    )

    # IQ4K 数据布局: [N//NR, K//QK_K, NUM_SUB_BLOCKS, QK_4_K_SUB_BLOCK_DATA_SIZE, NR]
    # 修复: 使用 QK_4_K_SUB_BLOCK_DATA_SIZE (16) 而不是 QK_4_K_DATA_SIZE (128)
    iq4k_block_ptr = tl.make_block_ptr(
        base=iq4k_matrix_ptr,
        shape=(N//NR, K// QK_K, NUM_SUB_BLOCKS, QK_4_K_SUB_BLOCK_DATA_SIZE, NR),
        strides=(K * NR // 2, NUM_SUB_BLOCKS * QK_4_K_SUB_BLOCK_DATA_SIZE * NR, QK_4_K_SUB_BLOCK_DATA_SIZE * NR, NR, 1),
        offsets=(pid_n, 0, 0, 0, 0),
        block_shape=(1, 1, 1, QK_4_K_SUB_BLOCK_DATA_SIZE, NR),
        order=(4, 3, 2, 1, 0)
    )

    # 缩放因子和额外数据的指针
    iq4k_d_block_ptr = tl.make_block_ptr(
        base=iq4k_d_ptr,
        shape=(N//NR, K//QK_K, NR),
        strides=(K * NR // QK_K, NR, 1),
        offsets=(pid_n, 0, 0),
        block_shape=(1, 1, NR),
        order=(2, 1, 0)
    )

    iq4k_extra_block_ptr = tl.make_block_ptr(
        base=iq4k_extra_ptr,
        shape=(N//NR, K//QK_K, NR),
        strides=(K * NR // QK_K, NR, 1),
        offsets=(pid_n, 0, 0),
        block_shape=(1, 1, NR),
        order=(2, 1, 0)
    )

    iq4k_scale_l_block_ptr = tl.make_block_ptr(
        base=iq4k_scale_l_ptr,
        shape=(N//NR, K//QK_K, NUM_SUB_BLOCKS, NR),
        strides=(K * NR // QK_K * NUM_SUB_BLOCKS, NUM_SUB_BLOCKS * NR, NR, 1),
        offsets=(pid_n, 0, 0, 0),
        block_shape=(1, 1, 1, NR),
        order=(3, 2, 1, 0)
    )

    # UINT8 * QK // 64 
    iq4k_scale_h_block_ptr = tl.make_block_ptr(
        base=iq4k_scale_h_ptr,
        shape=(N//NR, K//QK_K, NUM_SUB_BLOCKS * 2 // 4, NR),
        strides=(K * NR // QK_K * (QK_K//64), (QK_K//64) * NR, NR, 1),
        offsets=(pid_n, 0, 0, 0),
        block_shape=(1, 1, 1, NR),
        order=(3, 2, 1, 0)
    )

    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(M//MR, N//NR, MR, NR),
        strides=(N * MR, MR * NR, NR, 1),
        offsets=(pid_m, pid_n, 0, 0),
        block_shape=(1, 1, MR, NR),
        order=(3, 2, 1, 0)
    )

    # 累加器
    acc = tl.zeros((MR, NR), dtype=tl.float32)

    # 遍历 K 维度的每个块
    for k_block in range(N_block):
        # 每个 block 一块
        extra_data_ptr = tl.advance(iq4k_extra_block_ptr, offsets=(0, k_block, 0))
        iq4k_extra_data = tl.load(extra_data_ptr) # uint16 * NR

        # TODO 精度逻辑 还有点问题 
        
        # 矩阵乘
        sum = tl.zeros((MR, NR), dtype=tl.int32)

        for sub_block in range(NUM_SUB_BLOCKS):
            q8k_data_ptr_k0_15 = tl.advance(q8k_block_ptr, offsets=(0, k_block, sub_block, 1, 0, 0))
            q8k_data_k0_15 = tl.load(q8k_data_ptr_k0_15).reshape(16, MR)  # 加载并 reshape 为 2D: (16, MR)
            
            q8k_data_ptr_k16_31 = tl.advance(q8k_block_ptr, offsets=(0, k_block, sub_block, 0, 0, 0))
            q8k_data_k16_31 = tl.load(q8k_data_ptr_k16_31).reshape(16, MR)  # 加载并 reshape 为 2D: (16, MR)


            # 一个 sub-block 里两个 int4 scale_low 
            iq4k_scale_l_data_ptr = tl.advance(iq4k_scale_l_block_ptr, offsets=(0, k_block, sub_block, 0)) 
            iq4k_scale_l_data = tl.load(iq4k_scale_l_data_ptr) # uint8 * NR 

            iq4k_scale_h_data_ptr = tl.advance(iq4k_scale_h_block_ptr, offsets=(0, k_block, sub_block // 2, 0)) # 每个 subblock 两个 scale_h, uint8 存储 4 个 uint2
            iq4k_scale_h_data = (tl.load(iq4k_scale_h_data_ptr)) >> (4 & (sub_block % 2))  # uint8 * NR，包含 4 个 uint2, 这里只要两个

            iq4k_scale_1 = (iq4k_scale_l_data & 0x0F) | (iq4k_scale_h_data << 4 & 0x30)  # uint4 的低 4 bit + 高 2 bit 
            iq4k_scale_2 = (iq4k_scale_l_data >> 4) | (iq4k_scale_h_data << 4 & 0xC0)  # uint4 的高 4 bit + 低 2 bit 

            # 查表反量化
            tlb_start_idx = (iq4k_extra_data >> (2 * sub_block))   # [NR] 
            tlb_start_idx_1 = (tlb_start_idx & 0x01)  # [NR] 
            tlb_start_idx_2 = (tlb_start_idx & 0x02)  # [NR] 

            # tlb_start_idx_1 / tlb_start_idx_2 是 (NR,) -> (1,NR) 以便广播到 (16,NR) 
            mask1 = tl.reshape(tlb_start_idx_1, (NR, 1)).broadcast_to((NR, QK_4_K_SUB_BLOCK_DATA_SIZE)).cast(tl.int1)  # (NR, 16) 
            mask2 = tl.reshape(tlb_start_idx_2, (NR, 1)).broadcast_to((NR, QK_4_K_SUB_BLOCK_DATA_SIZE)).cast(tl.int1)  # (NR, 16) 

            # 按列选择对应表 (16,NR)
            value_tlb1 = tl.where(mask1, value_table1, value_table2)  # (NR, 16)
            value_tlb2 = tl.where(mask2, value_table1, value_table2)  # (NR, 16)

            # 加载并解包 iq4k 数据
            iq4k_data_ptr = tl.advance(iq4k_block_ptr, offsets=(0, k_block, sub_block, 0, 0))
            iq4k_data_packed = tl.load(iq4k_data_ptr)  # uint8, shape (16, NR), 每个 uint8 包含两个 iq4k 数据
            
            iq4k_data_1 = (iq4k_data_packed & 0x0F).reshape(16, NR)  # uint4, [16, NR] 
            iq4k_data_2 = (iq4k_data_packed >> 4).reshape(16, NR)  # uint4, [16, NR] 

            # 按行维度 (axis=1) gather：iq4k_data_1 / iq4k_data_2 的取值范围 0..15 
            # 支持 axis = 1， index 索引不连续，表的数据 load 也不连续，内存离散 load 操作，这里表的内容其实完全可以消掉 
            # 好处：iq4k data 不需要转置。输入 K 在高维的外积布局下，先是得转置查表，查完表后再次转置适合外积计算布局 
            # ？ 转置开销大 还是 非连续内存地址 load 开销大  
            iq4k_data_k0_15_dequant_data = (tl.gather(src=value_tlb1, index=iq4k_data_1.T, axis=1)).trans()  # (16, NR) 
            iq4k_data_k16_31_dequant_data = tl.gather(src=value_tlb2, index=iq4k_data_2.T, axis=1).trans()  # (16, NR) 

            iq4k_scale_1 = iq4k_scale_1.reshape((1, NR)).broadcast_to((MR, NR))
            # （16， NR) (16, MR)
            sum1 = tl.zeros((MR, NR), dtype=tl.int16)
            iq4k_data_k0_15_dequant_data_cast = iq4k_data_k0_15_dequant_data.cast(tl.int16) # 16, 32
            q8k_data_k0_15 = q8k_data_k0_15.cast(tl.int16) # 16, 12
            
            sum1 = tl.dot(q8k_data_k0_15.T, iq4k_data_k0_15_dequant_data_cast, out_dtype = tl.int16)  # int16, shape (12, 32)

            sum += sum1 * iq4k_scale_1  # int32, shape (MR, NR)

            iq4k_scale_2 = iq4k_scale_2.reshape(1,NR).broadcast_to((MR, NR))
            sum2 = tl.zeros((MR, NR), dtype=tl.int16)
            iq4k_data_k16_31_dequant_data_cast = iq4k_data_k16_31_dequant_data.cast(tl.int16)
            q8k_data_k16_31 = q8k_data_k16_31.cast(tl.int16)
            sum2 = tl.dot(q8k_data_k16_31.T, iq4k_data_k16_31_dequant_data_cast, out_dtype = tl.int16)  # int16, shape (MR, NR)
            sum += sum2 * iq4k_scale_2  # int32, shape (MR, NR)

        # 量化比例
        q8k_d_data_ptr = tl.advance(q8k_d_block_ptr, offsets=(0, k_block, 0)) 
        q8k_d_data = tl.load(q8k_d_data_ptr)  # float, shape (MR)
        q8k_d_data_reshaped = tl.reshape(q8k_d_data, (MR, 1))
        iq4_d_data_ptr = tl.advance(iq4k_d_block_ptr, offsets=(0, k_block, 0)) 
        iq4_d_data = tl.load(iq4_d_data_ptr)  # float, shape (NR)
        iq4_d_data_reshaped = tl.reshape(iq4_d_data, (NR, 1))
        scale_matrix  = tl.dot(q8k_d_data_reshaped, iq4_d_data_reshaped.T, out_dtype=tl.float32)  # float, shape (MR, NR)
        acc += sum.cast(tl.float32) * scale_matrix

    # 写回结果
    acc = acc.reshape( 1,1,MR, NR)
    output_ptr_final = tl.advance(output_block_ptr, offsets=(0, 0, 0, 0))
    tl.store(output_ptr_final, acc)

MR = 12
NR = 32
QK_K = 256

def _gflo_ps_from_ms(ms, M, N, K):
    return 2.0 * M * N * K * 1e-9 / (ms * 1e-3)

def prepare_random_iq4k_q8k_inputs(M, K, N):
    """构造随机但形状匹配的量化输入张量, 用于仅性能测试 (不保证数值语义正确)."""
    assert M % MR == 0 and N % NR == 0 and K % QK_K == 0, "形状不满足块大小约束"
    Mb = M // MR
    Nb = N // NR
    Ksup = K // QK_K
    NUM_SUB_BLOCKS = QK_K // 32  # 8
    SUB_BLOCK_SIZE = 32

    device = 'cpu'

    # q8k_matrix: shape (Mb, Ksup, NUM_SUB_BLOCKS, 2, SUB_BLOCK_SIZE//2, MR)
    q8k_matrix = torch.randint(-128, 127, (Mb, Ksup, NUM_SUB_BLOCKS, 2, SUB_BLOCK_SIZE//2, MR), dtype=torch.int8, device=device)
    q8k_d = torch.rand((Mb, Ksup, MR), dtype=torch.float32, device=device)

    # iq4k packed: (Nb, Ksup, NUM_SUB_BLOCKS, QK_K//2, NR) with uint8 storing two int4
    iq4k_matrix = torch.randint(0, 255, (Nb, Ksup, NUM_SUB_BLOCKS, QK_K//2, NR), dtype=torch.uint8, device=device)
    iq4k_d = torch.rand((Nb, Ksup, NR), dtype=torch.float32, device=device)
    # extra: uint16, each 2 bits per sub-block; we random 0..3 per pair
    iq4k_extra = torch.randint(0, 0xFFFF, (Nb, Ksup, NR), dtype=torch.int32, device=device).to(torch.uint16)
    # scale_l: (Nb, Ksup, NUM_SUB_BLOCKS, NR) uint8
    iq4k_scale_l = torch.randint(0, 255, (Nb, Ksup, NUM_SUB_BLOCKS, NR), dtype=torch.uint8, device=device)
    # scale_h: (Nb, Ksup, NUM_SUB_BLOCKS*2//4, NR)
    iq4k_scale_h = torch.randint(0, 255, (Nb, Ksup, (NUM_SUB_BLOCKS * 2)//4, NR), dtype=torch.uint8, device=device)

    # output
    output = torch.empty((M, N), dtype=torch.float32, device=device)

    return iq4k_matrix, iq4k_d, iq4k_extra, iq4k_scale_l, iq4k_scale_h, q8k_matrix, q8k_d, output


# debug 测试接口
if __name__ == "__main__":
    # 简单的功能测试
    M = 1200
    N = 2048
    K = 2048

    iq4k_matrix, iq4k_d, iq4k_extra, iq4k_scale_l, iq4k_scale_h, q8k_matrix, q8k_d, output = prepare_random_iq4k_q8k_inputs(M, K, N)

    grid = (M//MR, N//NR)

    t0 = time.perf_counter()
    iq4k_q8k_matmul_kernel[grid](
        iq4k_matrix, iq4k_d, iq4k_extra, iq4k_scale_l, iq4k_scale_h,
        q8k_matrix, q8k_d, output, M, N, K, num_threads=1, n_kernel_repeat=30
    )
    t1 = time.perf_counter()
    msec = (t1 - t0) / 30
    print(f"Kernel execution time: {msec:.4f} ms")
    print(f"GFLOPS: {_gflo_ps_from_ms(msec, M, N, K):.4f}")
    freq, vlen = 1.6, 256
    peak_flops = 2.0 * vlen / 16 * freq  # GOPS
    print(f"Peak GFLOPS: {peak_flops:.4f}")
    print(f"Utilization: {_gflo_ps_from_ms(msec, M, N, K) / peak_flops * 100:.2f} %")
    
    print("Output:", output)
