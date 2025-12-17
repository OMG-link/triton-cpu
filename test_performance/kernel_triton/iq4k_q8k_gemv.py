import torch
import time
import numpy as np

import triton
import triton.language as tl

'''
TRITON_ALWAYS_COMPILE=1  TRITON_CPU_BACKEND=1 python3 ./iq4k_q8k_gemv.py --simple-test
'''

table1 =   [ -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 ]
table2 =   [ -123, -100, -79, -61, -45, -31, -18, -6, 5, 17, 29, 42, 57, 73, 93, 117 ]


@triton.jit
def iq4k_q8k_gemv_kernel(
    q8k_vector_ptr,   # int8 
    q8k_d_ptr,        # float 
    iq4k_extra_ptr,   # iq4k 额外信息指针, uint16, 256 个元素超块一个值，256 / 16 = 16 bit，指示查哪个表 
    iq4k_matrix_ptr,  # uint4 packed in uint8 
    iq4k_d_ptr,       # uint16 
    iq4k_scale_l_ptr, # iq4k 低位量化比例指针, uint8, 16个数据一个 4bit 值， 一共 16 个值，打包成 8 bit 
    iq4k_scale_h_ptr, # iq4k 高位量化比例指针, uint8 (实际是4个uint8打包成uint32) 
    output_ptr,       # float32  
    K,
    N,
    batch: tl.constexpr = 1,
):
    pid_b = tl.program_id(axis=0)  # batch id 
    pid_n = tl.program_id(axis=1)  # N id 
    QK_K : tl.constexpr = 256 
    QK_4_K_DATA_SIZE : tl.constexpr = QK_K // 2  # 每个 uint8 包含两个 iq4k 元素 
    QK_K_SUB_BLOCK_SIZE : tl.constexpr = 32
    QK_4_K_SUB_BLOCK_DATA_SIZE : tl.constexpr = QK_K_SUB_BLOCK_SIZE // 2 
    NUM_SUB_BLOCKS : tl.constexpr = QK_K // 32  # 每个块有8个子块 
    NR : tl.constexpr = 32


    value_table1 = tl.arange(0, 16).cast(tl.int8).reshape(1, 16).broadcast_to((NR, 16))
    value_table2 = tl.arange(0, 16).cast(tl.int8).reshape(1, 16).broadcast_to((NR, 16))



    # q8 向量按超块 & 子块划分: [batch, K//QK_K, NUM_SUB_BLOCKS, QK_K_SUB_BLOCK_SIZE]
    q8k_block_ptr = tl.make_block_ptr(
        base=q8k_vector_ptr,
        shape=(batch, K // QK_K, QK_K // QK_K_SUB_BLOCK_SIZE, QK_K_SUB_BLOCK_SIZE, 1),
        strides=((K // QK_K) * QK_K, QK_K, QK_K_SUB_BLOCK_SIZE, 1, 1),
        offsets=(pid_b, 0, 0, 0, 0),
        block_shape=(1, 1, 1, 1, 1),
        order=(4, 3, 2, 1, 0)
    )

    # q8 每个超块的缩放因子: shape=(batch, K//QK_K)
    q8k_d_block_ptr = tl.make_block_ptr(
        base=q8k_d_ptr,
        shape=(batch, K // QK_K),
        strides=(K // QK_K, 1),
        offsets=(pid_b, 0),
        block_shape=(1, 1),
        order=(1, 0)
    )

    # IQ4K 数据布局: [batch, N//NR, K//QK_K, NUM_SUB_BLOCKS, QK_4_K_DATA_SIZE, NR]
    iq4k_block_ptr = tl.make_block_ptr(
        base=iq4k_matrix_ptr,
        shape=(batch, N//NR, K//QK_K, NUM_SUB_BLOCKS, QK_4_K_SUB_BLOCK_DATA_SIZE, NR),
        strides=((N//NR) * K * NR // 2, K * NR // 2, NUM_SUB_BLOCKS * QK_4_K_SUB_BLOCK_DATA_SIZE * NR, QK_4_K_SUB_BLOCK_DATA_SIZE * NR, NR, 1),
        offsets=(pid_b, pid_n, 0, 0, 0, 0),
        block_shape=(1, 1, 1, 1, QK_4_K_SUB_BLOCK_DATA_SIZE, NR),
        order=(5, 4, 3, 2, 1, 0)
    )


    # 缩放因子和额外数据的指针
    iq4k_d_block_ptr = tl.make_block_ptr(
        base=iq4k_d_ptr,
        shape=(batch, N//NR, K//QK_K, NR),
        strides=((N//NR) * K * NR // QK_K, K * NR // QK_K, NR, 1),
        offsets=(pid_b, pid_n, 0, 0),
        block_shape=(1, 1, 1, NR),
        order=(3, 2, 1, 0)
    )

    iq4k_extra_block_ptr = tl.make_block_ptr(
        base=iq4k_extra_ptr,
        shape=(batch, N//NR, K//QK_K, NR),
        strides=((N//NR) * K * NR // QK_K, K * NR // QK_K, NR, 1),
        offsets=(pid_b, pid_n, 0, 0),
        block_shape=(1, 1, 1, NR),
        order=(3, 2, 1, 0)
    )

    # uint8 类型
    iq4k_scale_l_block_ptr = tl.make_block_ptr(
        base=iq4k_scale_l_ptr,
        shape=(batch, N//NR, K//QK_K, NUM_SUB_BLOCKS * 2 // 2, NR),
        strides=((N//NR) * K * NR * NUM_SUB_BLOCKS // QK_K, K * NR * NUM_SUB_BLOCKS // QK_K, NUM_SUB_BLOCKS * NR, NR, 1),
        offsets=(pid_b, pid_n, 0, 0, 0),
        block_shape=(1, 1, 1, 1, NR),
        order=(4, 3, 2, 1, 0)
    )

    # uint8 类型
    iq4k_scale_h_block_ptr = tl.make_block_ptr(
        base=iq4k_scale_h_ptr,
        # iq4 的子块分组大小为 16，分块数量为 q8 两倍, 故 * 2
        # 每 4 个 subblock 高 2 bit 打包为 uint8, 故 / 4
        shape=(batch, N//NR, K//QK_K, NUM_SUB_BLOCKS * 2 // 4, NR),
        strides=((N//NR) * K * NR // QK_K * (QK_K//64), K * NR // QK_K * (QK_K//64), (QK_K//64) * NR, NR, 1),
        offsets=(pid_b, pid_n, 0, 0, 0),
        block_shape=(1, 1, 1, 1, NR),
        order=(4, 3, 2, 1, 0)
    )

    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(batch, N//NR, NR),
        strides=((N//NR) * NR, NR, 1),
        offsets=(pid_b, pid_n, 0),
        block_shape=(1, 1, NR),
        order=(2, 1, 0)
    )

    acc = tl.zeros((1,1,1,1,NR,), dtype=tl.float32)
    for k_block in range(0, K//QK_K):

        # 每个 block 一块
        extra_data_ptr = tl.advance(iq4k_extra_block_ptr, offsets=(0, 0, k_block, 0)) # [NR]@int16
        iq4k_extra_data = tl.load(extra_data_ptr) # uint16 * NR

        
        q8k_d_data_ptr = tl.advance(q8k_d_block_ptr, offsets=(0, k_block))
        iq4k_d_data_ptr = tl.advance(iq4k_d_block_ptr, offsets=(0, 0, k_block, 0))
        q8k_d_data = tl.load(q8k_d_data_ptr)  # float32, [NR] 
        iq4k_d_data = tl.load(iq4k_d_data_ptr)  # uint16, [NR] 
        d4d8 = q8k_d_data * iq4k_d_data  # float32, [NR] 

        sum = tl.zeros((1,1,1,1,NR,), dtype=tl.int32)
        for k_sub_block in range(0, NUM_SUB_BLOCKS):
            # scale 
            iq4k_scale_l_data_ptr = tl.advance(iq4k_scale_l_block_ptr, offsets=(0, 0, k_block, k_sub_block, 0)) 
            iq4k_scale_l_data = tl.load(iq4k_scale_l_data_ptr)  # uint8 * NR，包含两个 uint4 

            iq4k_scale_h_data_ptr = tl.advance(iq4k_scale_h_block_ptr, offsets=(0, 0, k_block, k_sub_block // 2, 0)) # 每个 subblock 两个 scale_h, uint8 存储 4 个 uint2
            iq4k_scale_h_data = (tl.load(iq4k_scale_h_data_ptr)) >> (4 & (k_sub_block % 2))  # uint8 * NR，包含 4 个 uint2, 这里只要两个

            scale_1 = (iq4k_scale_l_data & 0x0F) | (iq4k_scale_h_data << 4 & 0x30)  # uint4 的低 4 bit + 高 2 bit 
            scale_2 = (iq4k_scale_l_data >> 4) | (iq4k_scale_h_data << 4 & 0xC0)  # uint4 的高 4 bit + 低 2 bit 
            
            iq4k_data_block_ptr = tl.advance(iq4k_block_ptr, offsets=(0, 0, k_block, k_sub_block, 0, 0))
            iq4k_data_packed = tl.load(iq4k_data_block_ptr)  # uint8 * NR, 每个 uint8 包含两个 iq4k 数据 [16, NR]@uint8 => [32, NR]@uint4

            iq4k_data_1 = (iq4k_data_packed & 0x0F).reshape(16, NR).permute(1, 0)  # uint4, [NR, 16] 
            iq4k_data_2 = (iq4k_data_packed >> 4).reshape(16, NR).permute(1, 0)    # uint4, [NR, 16] 

            # dequant by lookup table
            tlb_start_idx = (iq4k_extra_data >> (2 * k_sub_block))   # [NR] 

            # FIXME 不支持 bool 类型
            # tlb_start_idx_1 = (tlb_start_idx & 0x01).cast(tl.bool) # [NR]  
            # tlb_start_idx_2 = (tlb_start_idx & 0x02).cast(tl.bool) # [NR]  
            tlb_start_idx_1 = (tlb_start_idx & 0x01).reshape(NR,1)  # [NR] 
            tlb_start_idx_2 = (tlb_start_idx & 0x02).reshape(NR,1)  # [NR] 

            # # 构造低/高表 (16,1) => 通过广播按列选择，避免不支持的二维切片语法 
            # value_table_lo = tl.reshape(tl.constexpr(value_table[:16]), (16, 1))  # (16,1) 
            # value_table_hi = tl.reshape(tl.constexpr(value_table[16:]), (16, 1))  # (16,1) 

            # tlb_start_idx_1 / tlb_start_idx_2 是 (NR,) -> (NR, 1) 以便广播到 (NR. 16) 
            mask1 = tlb_start_idx_1.broadcast_to((NR, QK_4_K_SUB_BLOCK_DATA_SIZE)).cast(tl.int1)  # (NR,16) 
            mask2 = tlb_start_idx_2.broadcast_to((NR, QK_4_K_SUB_BLOCK_DATA_SIZE)).cast(tl.int1)  # (NR,16) 

            # 按列选择对应表 (16,NR)
            value_tlb1 = tl.where(mask1, value_table1, value_table2)  # (NR, 16)
            value_tlb2 = tl.where(mask2, value_table1, value_table2)  # (NR, 16)

            # 按行维度 (axis=0) gather：iq4k_data_1 / iq4k_data_2 的取值范围 0..15 
            iq4k_data_k0_15_dequant_data = (tl.gather(src=value_tlb1, index=iq4k_data_1, axis=1)) # (NR,16)  
            iq4k_data_k16_31_dequant_data = (tl.gather(src=value_tlb2, index=iq4k_data_2, axis=1))  # (NR,16) 

            sum1 = tl.zeros((NR,), dtype=tl.int16)
            sum2 = tl.zeros((NR,), dtype=tl.int16)
            # 修复: 应该循环 QK_4_K_SUB_BLOCK_DATA_SIZE (16) 次，而不是 QK_4_K_DATA_SIZE (128) 次
            for k in range(QK_4_K_SUB_BLOCK_DATA_SIZE):
                # 创建索引张量 [NR]，所有元素都是 k，用于从 [16,NR] 中提取第 k 行
                k_idx = tl.full((NR,), k, dtype=tl.int32).reshape((NR,1))
                
                # 从 [16,NR] 矩阵中提取第 k 行 -> [NR]
                iq4k_data_lo = tl.reshape(tl.gather(src=iq4k_data_k0_15_dequant_data, index=k_idx, axis=1), (NR,))  # [NR] vector
                iq4k_data_hi = tl.reshape(tl.gather(src=iq4k_data_k16_31_dequant_data, index=k_idx, axis=1), (NR,)) # [NR] vector

                q8k_data_ptr = tl.advance(q8k_block_ptr, offsets=(0, k_block, k_sub_block, k, 0))

                q8k_data = tl.reshape(tl.load(q8k_data_ptr), (1,)) # [1]@int8，标量会自动广播 
                sum1 += q8k_data.cast(tl.int16) * iq4k_data_lo.cast(tl.int16)  

                # 修复: 访问第二组数据时，offset 应该是 k + QK_4_K_SUB_BLOCK_DATA_SIZE (16)，而不是 k + QK_4_K_DATA_SIZE (128) 
                q8k_data_ptr = tl.advance(q8k_block_ptr, offsets=(0, k_block, k_sub_block, k + QK_4_K_SUB_BLOCK_DATA_SIZE, 0)) 
                q8k_data = tl.reshape(tl.load(q8k_data_ptr), (1,)) # [1]@int8，标量会自动广播 
                sum2 += q8k_data.cast(tl.int16) * iq4k_data_hi.cast(tl.int16) 

            sum += scale_1 * sum1.cast(tl.int32) + scale_2 * sum2.cast(tl.int32) 

        acc += sum.cast(tl.float32) * d4d8  # float32 

    # Reshape acc to match output_block_ptr's block_shape (1, 1, NR)
    acc = acc.reshape((1, 1, NR))
    output_ptr = tl.advance(output_block_ptr, offsets=(0, 0, 0)) 
    tl.store(output_ptr, acc)

###############################################
# 追加: iq4k_q8k 调试/带宽测试入口
###############################################
if __name__ == "__main__":
    import argparse, os, sys
    if os.path.dirname(__file__) not in sys.path:
        sys.path.append(os.path.dirname(__file__))
    try:
        import bench_gemv_driver as gemv_driver
    except ImportError:
        gemv_driver = None

    parser = argparse.ArgumentParser(description='iq4k_q8k GEMV 调试/带宽测试')
    parser.add_argument('--n', type=int, default=512, help='输出维度(32 的倍数)')
    parser.add_argument('--k', type=int, default=256*4, help='输入维度(256 的倍数)')
    parser.add_argument('--batch', type=int, default=1, help='batch 维度')
    parser.add_argument('--rounds', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--target-gb', type=float, default=0.5)
    parser.add_argument('--num_threads', type=int, default=4)
    parser.add_argument('--simple-test', action='store_true', help='运行简单功能测试')
    args = parser.parse_args()

    if args.simple_test:
        print('运行 iq4k_q8k 简单功能测试...')
        N = args.n
        K = args.k
        batch = args.batch
        assert N % 32 == 0 and K % 256 == 0
        NR = 32
        QK_K = 256
        QK_K_SUB_BLOCK_SIZE = 32
        NUM_SUB_BLOCKS = 8
        QK_4_K_SUB_BLOCK_DATA_SIZE = 16
        # 修复: q8k_vector 应该是 4D 以匹配 kernel，增加 batch 维度
        q8k_vector = torch.randint(-128,127,(batch, K//QK_K, QK_K//QK_K_SUB_BLOCK_SIZE, QK_K_SUB_BLOCK_SIZE, 1),dtype=torch.int8)
        q8k_d = torch.randn((batch, K//QK_K),dtype=torch.float32)
        iq4k_extra = torch.randint(0,65535,(batch, N//NR,K//QK_K,NR),dtype=torch.uint16)
        iq4k_matrix = torch.randint(0,255,(batch, N//NR,K//QK_K,NUM_SUB_BLOCKS,QK_4_K_SUB_BLOCK_DATA_SIZE,NR),dtype=torch.uint8)
        iq4k_d = torch.randint(0,65535,(batch, N//NR,K//QK_K,NR),dtype=torch.uint16)
        iq4k_scale_l = torch.randint(0,255,(batch, N//NR,K//QK_K,NUM_SUB_BLOCKS,NR),dtype=torch.uint8)
        iq4k_scale_h = torch.randint(0,255,(batch, N//NR,K//QK_K,NUM_SUB_BLOCKS//2,NR),dtype=torch.uint8)
        # 修复: output 应该是 2D 以匹配 kernel，增加 batch 维度
        output = torch.zeros((batch, N//NR, NR),dtype=torch.float32)
        
        grid = (batch, N//NR)


        iq4k_q8k_gemv_kernel[grid](
            q8k_vector_ptr=q8k_vector,
            q8k_d_ptr=q8k_d,
            iq4k_extra_ptr=iq4k_extra,
            iq4k_matrix_ptr=iq4k_matrix,
            iq4k_d_ptr=iq4k_d,
            iq4k_scale_l_ptr=iq4k_scale_l,
            iq4k_scale_h_ptr=iq4k_scale_h,
            output_ptr=output,
            K=K,
            N=N,
            batch=batch,
            num_threads=args.num_threads,
        )
        print('简单测试完成 输出范围:', float(output.min()), float(output.max()))
    else:
        if gemv_driver is None:
            raise RuntimeError('未找到 gemv_driver.py，无法执行带宽测试。')
        gemv_driver.run_bandwidth(
            kernel_name='iq4k_q8k',
            N=args.n,
            K=args.k,
            rounds=args.rounds,
            warmup=args.warmup,
            target_gb=args.target_gb,
            threads=args.num_threads,
        )
            