import torch
import time
import numpy as np

import triton
import triton.language as tl

BLOCK_SIZE_M = 1
BLOCK_SIZE_N = 512
USE_GPU = False


# q8k 为输入激活向量
# q4k 为权重矩阵
@triton.jit
def q4k_q8k_gemv_kernel(
    q8k_vector_ptr,  # int8 
    q8k_d_ptr,       # float 
    q8_bsums_ptr,    # int16 
    q4k_matrix_ptr,  # int4 packed in uint8 
    q4k_scale_l_ptr, # q4k 低位量化比例指针, uint8, 32 个数据一个 4bit 值， 一共 8 个值，打包成 8 bit
    q4k_scale_h_ptr, # q4k 高位量化比例指针, uint8, 4个 uint2 打包成uint8)
    q4k_mins_l_ptr,  # uint6 非对称量化偏移
    q4k_mins_h_ptr,  # uint6 非对称量化偏移 
    q4k_d_ptr,       # uint16 
    q4k_dmin_ptr,    # uint16 
    output_ptr,      # float32 
    K,
    N,
):
    pid_n = tl.program_id(axis=0)
    NR : tl.constexpr = 32
    QK_K : tl.constexpr = 256
    QK_K_SUB_BLOCK_SIZE : tl.constexpr = 32
    QK_4_K_SUB_BLOCK_DATA_SIZE : tl.constexpr = QK_K_SUB_BLOCK_SIZE // 2
    NUM_SUB_BLOCKS : tl.constexpr = QK_K // QK_K_SUB_BLOCK_SIZE  # 每个块有8个子块
    NR_SUB_BLOCKS_NUMS = N // NR
    QK_SUPER_BLOCK_NUMS = K // QK_K

    
    q8k_vector_block_ptr = tl.make_block_ptr(
        base=q8k_vector_ptr,
        shape=(QK_SUPER_BLOCK_NUMS, NUM_SUB_BLOCKS, QK_K_SUB_BLOCK_SIZE, 1),
        strides=(QK_K, QK_K_SUB_BLOCK_SIZE, 1, 1),
        offsets=(0, 0, 0, 0),
        block_shape=(1, 1, 1, 1),
        order=(3, 2, 1, 0),
    )

    q8k_d_block_ptr = tl.make_block_ptr(
        base=q8k_d_ptr,
        shape=(QK_SUPER_BLOCK_NUMS,),
        strides=(1,),
        offsets=(0,),
        block_shape=(1,),
        order=(0,),
    )

    q8k_bsums_block_ptr = tl.make_block_ptr(
        base=q8_bsums_ptr,
        shape=(QK_SUPER_BLOCK_NUMS, NUM_SUB_BLOCKS, 2, 1),
        strides=(NUM_SUB_BLOCKS * 2, 2, 1, 1),
        offsets=(0, 0, 0, 0),
        block_shape=(1, 1, 1, 1),
        order=(3, 2, 1, 0),
    ) # @int16

    q4k_block_ptr = tl.make_block_ptr(
        base=q4k_matrix_ptr,
        shape=(NR_SUB_BLOCKS_NUMS, K//QK_K, NUM_SUB_BLOCKS, QK_4_K_SUB_BLOCK_DATA_SIZE, NR),
        strides=(K*NR//2, QK_K*NR//2, QK_4_K_SUB_BLOCK_DATA_SIZE * NR, NR, 1),
        offsets=(pid_n, 0, 0, 0, 0),
        block_shape=(1, 1, 1, 1, NR),
        order=(4, 3, 2, 1, 0),
    )

    q4k_scale_l_block_ptr = tl.make_block_ptr(
        base=q4k_scale_l_ptr,
        shape=(NR_SUB_BLOCKS_NUMS, K//QK_K, NUM_SUB_BLOCKS // 2, NR),
        strides=(K * NR // QK_K_SUB_BLOCK_SIZE // 2, (NUM_SUB_BLOCKS // 2) * NR, NR, 1),
        offsets=(pid_n, 0, 0, 0),
        block_shape=(1, 1, 1, NR),
        order=(3, 2, 1, 0),
    )

    q4k_scale_h_block_ptr = tl.make_block_ptr(
        base=q4k_scale_h_ptr,
        shape=(NR_SUB_BLOCKS_NUMS, K//QK_K, NUM_SUB_BLOCKS // 4, NR),
        strides =(K * NR // QK_K_SUB_BLOCK_SIZE // 4, (NUM_SUB_BLOCKS // 4) * NR, NR, 1),
        offsets=(pid_n, 0, 0, 0),
        block_shape=(1, 1, 1, NR),
        order=(3, 2, 1, 0),
    )

    q4k_mins_l_block_ptr = tl.make_block_ptr(
        base=q4k_mins_l_ptr,
        shape=(NR_SUB_BLOCKS_NUMS, K//QK_K, NUM_SUB_BLOCKS // 2, NR),
        strides=(K//QK_K * NR // 2, (NUM_SUB_BLOCKS // 2) * NR, NR, 1),
        offsets=(pid_n, 0, 0, 0),
        block_shape=(1, 1, 1, NR),
        order=(3, 2, 1, 0),
    )

    q4k_mins_h_block_ptr = tl.make_block_ptr(
        base=q4k_mins_h_ptr,
        shape=(NR_SUB_BLOCKS_NUMS, K//QK_K, NUM_SUB_BLOCKS // 4, NR),
        strides=(K//QK_K * NR // 4, (NUM_SUB_BLOCKS // 4) * NR, NR, 1),
        offsets=(pid_n, 0, 0, 0),
        block_shape=(1, 1, 1, NR),
        order=(3, 2, 1, 0),
    )

    q4k_d_block_ptr = tl.make_block_ptr(
        base=q4k_d_ptr,
        shape=(NR_SUB_BLOCKS_NUMS, K//QK_K, NR),
        strides=(K//QK_K*NR, NR, 1),
        offsets=(pid_n, 0, 0),
        block_shape=(1, 1, NR),
        order=(2, 1, 0),
    )

    q4k_dmin_block_ptr = tl.make_block_ptr(
        base=q4k_dmin_ptr,
        shape=(NR_SUB_BLOCKS_NUMS, K//QK_K, NR),
        strides=(K//QK_K*NR, NR, 1),
        offsets=(pid_n, 0, 0),
        block_shape=(1, 1, NR),
        order=(2, 1, 0),
    )

    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(NR_SUB_BLOCKS_NUMS, NR),
        strides=(NR, 1),
        offsets=(pid_n, 0),
        block_shape=(1, NR),
        order=(1, 0),
    )

    # TODO 这里一定得写成 （1, NR), 后面 reshape 成 (1,NR,) 无效果
    result = tl.zeros((1, NR), dtype=tl.float32)

    for k_block in range(K // QK_K):

        q4k_d_data_ptr = tl.advance(q4k_d_block_ptr, offsets=(0, k_block, 0))
        q4k_d_data = tl.load(q4k_d_data_ptr).reshape((NR,))  # [NR]@float32

        q4k_dmin_data_ptr = tl.advance(q4k_dmin_block_ptr, offsets=(0, k_block, 0))
        q4k_dmin_data = tl.load(q4k_dmin_data_ptr).reshape((NR,))  # [NR]@float32

        q8k_d_data_ptr = tl.advance(q8k_d_block_ptr, offsets=(k_block,))
        q8k_d_data = tl.load(q8k_d_data_ptr).reshape((1,))  # [1]@float32

        sum = tl.zeros((NR,), dtype=tl.int32)
        bsums_min = tl.zeros((NR,), dtype=tl.int32)
        
        for k_sub_block in range(NUM_SUB_BLOCKS):
            # q8k_vector_data_ptr = tl.advance(q8k_vector_block_ptr, offsets=(k_block, k_sub_block, 0)) 
            # q8k_vector_data = tl.load(q8k_vector_data_ptr)  # [QK_K_SUB_BLOCK_SIZE] @int8

            q8k_bsums_data_ptr = tl.advance(q8k_bsums_block_ptr, offsets=(k_block, k_sub_block, 0, 0))
            q8k_bsums_data_1 = (tl.load(q8k_bsums_data_ptr)).reshape((1,))  # [1] @int32 
            q8k_bsums_data_ptr = tl.advance(q8k_bsums_block_ptr, offsets=(k_block, k_sub_block, 1, 0))
            q8k_bsums_data_2 = (tl.load(q8k_bsums_data_ptr)).reshape((1,))  # [1] @int32 

            # q4k_data_ptr = tl.advance(q4k_block_ptr, offsets=(0, k_block, k_sub_block, 0, 0))
            # q4k_data = tl.load(q4k_data_ptr)  # [QK_4_K_SUB_BLOCK_DATA_SIZE, NR]@uint8 

            # q4k_data_k0_15 = q4k_data & 0x0F  # [16, NR]@uint8 
            # q4k_data_k16_31 = q4k_data >> 4  # [16, NR]@uint8 

            q4k_scale_l_data_ptr = tl.advance(q4k_scale_l_block_ptr, offsets=(0, k_block, k_sub_block // 2, 0))
            q4k_scale_l_data = ((tl.load(q4k_scale_l_data_ptr)) >> (4 * (k_sub_block % 2))) & 0x0f  # [NR]@uint8 

            q4k_scale_h_data_ptr = tl.advance(q4k_scale_h_block_ptr, offsets=(0, k_block, k_sub_block // 4, 0))
            q4k_scale_h_data = ((tl.load(q4k_scale_h_data_ptr)) >> (2 * (k_sub_block % 4))) & 0x03  # [NR]@uint8 
 
            q4k_scale = ((q4k_scale_l_data) | (q4k_scale_h_data << 4)).reshape((NR,))  # [NR]@uint8 

   
            q4k_mins_l_data_ptr = tl.advance(q4k_mins_l_block_ptr, offsets=(0, k_block, k_sub_block // 2, 0))
            q4k_mins_l_data = ((tl.load(q4k_mins_l_data_ptr)) >> (4 * (k_sub_block % 2))) & 0x0f  # [NR]@uint8 

            q4k_mins_h_data_ptr = tl.advance(q4k_mins_h_block_ptr, offsets=(0, k_block, k_sub_block // 4, 0))
            q4k_mins_h_data = ((tl.load(q4k_mins_h_data_ptr)) >> (2 * (k_sub_block % 4))) & 0x03 # [NR]@uint8 

            q4k_mins = ((q4k_mins_l_data) | (q4k_mins_h_data << 4)).reshape((NR,))  # [NR]@uint8 

            bsums_min += q8k_bsums_data_1 * q4k_mins.cast(tl.int32)  # [NR] @int32
            bsums_min += q8k_bsums_data_2 * q4k_mins.cast(tl.int32)  # [NR] @int32

            sum1 = tl.zeros(( NR,), dtype=tl.int16)
            sum2 = tl.zeros(( NR,), dtype=tl.int16)
            
            
            for k in range(QK_4_K_SUB_BLOCK_DATA_SIZE):
                # 自动广播 
                q8k_data_ptr = tl.advance(q8k_vector_block_ptr, offsets=(k_block, k_sub_block, k, 0))
                q8k_data = tl.load(q8k_data_ptr)  # [1] @int8
                q4_k_data_ptr = tl.advance(q4k_block_ptr, offsets=(0, k_block, k_sub_block, k, 0))
                q4_k_data = tl.load(q4_k_data_ptr)  # [NR] @uint8

                q4k_data_k0_15 = q4_k_data & 0x0F  # [NR]@uint8
                q4k_data_k16_31 = q4_k_data >> 4  # [NR]@uint8

                sum1 += (q8k_data.cast(tl.int16) * q4k_data_k0_15.cast(tl.int16)).reshape((NR,))

                q8k_data_ptr = tl.advance(q8k_vector_block_ptr, offsets=(k_block, k_sub_block, k + QK_4_K_SUB_BLOCK_DATA_SIZE, 0))
                q8k_data = tl.load(q8k_data_ptr)  # [1] @int8

                sum2 += (q8k_data.cast(tl.int16) * q4k_data_k16_31.cast(tl.int16)).reshape((NR,))

            
            sum += q4k_scale * (sum1.cast(tl.int32) + sum2.cast(tl.int32))
        
        # superblock
        d4d8 = q8k_d_data * q4k_d_data  # [NR]@float32
        d8d4min = q8k_d_data * q4k_dmin_data  # [NR]@float32

        result += d4d8 * sum.cast(tl.float32) - d8d4min * bsums_min

    # tl.reshape(result, (1, 32))  # shape: [1, 32]
    output_data_ptr = tl.advance(output_block_ptr, offsets=(0,))
    tl.store(output_data_ptr, result)


###############################################
# 追加: 单文件调试入口 & 带宽测试委托到统一驱动
###############################################
if __name__ == "__main__":
    import argparse, os, sys
    if os.path.dirname(__file__) not in sys.path:
        sys.path.append(os.path.dirname(__file__))
    try:
        import gemv_driver
    except ImportError:
        gemv_driver = None

    parser = argparse.ArgumentParser(description='q4k_q8k GEMV 调试/带宽测试')
    parser.add_argument('--n', type=int, default=512, help='输出维度(32 的倍数)')
    parser.add_argument('--k', type=int, default=256*4, help='输入维度(256 的倍数)')
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--target-gb', type=float, default=1.0)
    parser.add_argument('--num_threads', type=int, default=4)
    parser.add_argument('--simple-test', action='store_true', help='运行简单功能测试')
    args = parser.parse_args()

    if args.simple_test:
        print('运行 q4k_q8k 简单功能测试...')
        N = args.n
        K = args.k
        assert N % 32 == 0 and K % 256 == 0
        NR = 32
        QK_K = 256
        QK_K_SUB_BLOCK_SIZE = 32

        NUM_SUB_BLOCKS = 8
        QK_4_K_SUB_BLOCK_DATA_SIZE = 16
        QK_SUPER_BLOCK_NUMS = K // QK_K
        NR_SUB_BLOCKS_NUMS = N // NR

        # q8k 数据准备 (输入激活向量)
        q8k_vector = torch.randint(-128, 127, (QK_SUPER_BLOCK_NUMS, NUM_SUB_BLOCKS, QK_K_SUB_BLOCK_SIZE, 1), dtype=torch.int8)
        q8k_d = torch.randn((QK_SUPER_BLOCK_NUMS,), dtype=torch.float32)
        q8_bsums = torch.randint(-32768, 32767, (QK_SUPER_BLOCK_NUMS, NUM_SUB_BLOCKS, 2, 1), dtype=torch.int16)
        
        # q4k 数据准备 (权重矩阵)
        q4k_matrix = torch.randint(0, 255, (NR_SUB_BLOCKS_NUMS, QK_SUPER_BLOCK_NUMS, NUM_SUB_BLOCKS, QK_4_K_SUB_BLOCK_DATA_SIZE, NR), dtype=torch.uint8)
        q4k_scale_l = torch.randint(0, 255, (NR_SUB_BLOCKS_NUMS, QK_SUPER_BLOCK_NUMS, NUM_SUB_BLOCKS // 2, NR), dtype=torch.uint8)
        q4k_scale_h = torch.randint(0, 255, (NR_SUB_BLOCKS_NUMS, QK_SUPER_BLOCK_NUMS, NUM_SUB_BLOCKS // 4, NR), dtype=torch.uint8)
        q4k_mins_l = torch.randint(0, 255, (NR_SUB_BLOCKS_NUMS, QK_SUPER_BLOCK_NUMS, NUM_SUB_BLOCKS // 2, NR), dtype=torch.uint8)
        q4k_mins_h = torch.randint(0, 255, (NR_SUB_BLOCKS_NUMS, QK_SUPER_BLOCK_NUMS, NUM_SUB_BLOCKS // 4, NR), dtype=torch.uint8)
        q4k_d = torch.randint(0, 65535, (NR_SUB_BLOCKS_NUMS, QK_SUPER_BLOCK_NUMS, NR), dtype=torch.uint16)
        q4k_dmin = torch.randint(0, 65535, (NR_SUB_BLOCKS_NUMS, QK_SUPER_BLOCK_NUMS, NR), dtype=torch.uint16)
        
        # 输出准备
        output = torch.zeros((N//NR,NR), dtype=torch.float32)

        grid = (N//NR,)
        q4k_q8k_gemv_kernel[grid](
            q8k_vector_ptr=q8k_vector,
            q8k_d_ptr=q8k_d,
            q8_bsums_ptr=q8_bsums,
            q4k_matrix_ptr=q4k_matrix,
            q4k_scale_l_ptr=q4k_scale_l,
            q4k_scale_h_ptr=q4k_scale_h,
            q4k_mins_l_ptr=q4k_mins_l,
            q4k_mins_h_ptr=q4k_mins_h,
            q4k_d_ptr=q4k_d,
            q4k_dmin_ptr=q4k_dmin,
            output_ptr=output,
            K=K,
            N=N,
            num_threads=args.num_threads,
        )
        print('简单测试完成 输出范围:', float(output.min()), float(output.max()))
    else:
        if gemv_driver is None:
            raise RuntimeError('未找到 gemv_driver.py，无法执行带宽测试。')
        gemv_driver.run_bandwidth(
            kernel_name='q4k_q8k',
            N=args.n,
            K=args.k,
            rounds=args.rounds,
            warmup=args.warmup,
            target_gb=args.target_gb,
            threads=args.num_threads,
        )
