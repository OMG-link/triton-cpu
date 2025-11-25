import torch
import time
import numpy as np

import triton
import triton.language as tl




@triton.jit
def i8_i8_gemm_kernel(
    a_matrix_ptr,         # 输入 a 矩阵指针, int8
    b_matrix_ptr,         # 输入 b 矩阵指针, int8
    output_ptr,           # 输出指针，intel2
    M,                    # 矩阵行数
    N,                    # 矩阵列数
    K,                    # 矩阵公共维度，这里表示元素数量而不是字节数
):
    QK_8_0 : tl.constexpr = 32
    MR : tl.constexpr = 12
    NR : tl.constexpr = 32
    N_block : tl.constexpr = K // QK_8_0

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    a_block_ptr = tl.make_block_ptr(
        base=a_matrix_ptr,
        shape=(M//MR, K//QK_8_0, QK_8_0, MR),
        strides=(K * MR, QK_8_0 * MR, MR, 1),
        offsets=(pid_m, 0, 0, 0),
        block_shape=(1,1, QK_8_0, MR),
        order=(3,2,1,0)
    )

    b_block_ptr = tl.make_block_ptr(
        base=b_matrix_ptr,
        shape=(N//NR, K//QK_8_0, QK_8_0, NR),
        strides=(K * NR, QK_8_0 * NR, NR, 1),
        offsets=(pid_n, 0, 0, 0),
        block_shape=(1,1,QK_8_0, NR),
        order=(3,2,1,0)
    )


    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(M//MR,N//NR, MR, NR),
        strides=(N * MR, MR * NR, NR, 1),
        offsets=(pid_m, pid_n, 0, 0),
        block_shape=(1, 1,MR, NR),
        order=(3,2,1,0)
    )

    # 精度对齐需求
    acc = tl.zeros((MR, NR), dtype=tl.int32)

    for k_block in range(N_block):
        # 加载 a 数据块: 原始形状 (1, 1, QK_8_0, MR)
        a_data_ptr = tl.advance(a_block_ptr, offsets = (0, k_block, 0, 0))
        a_data = tl.load(a_data_ptr)  # shape: (1, 1, QK_8_0, MR)
        a_data = tl.reshape(a_data, (QK_8_0, MR))  # reshape to (32, 12)
        a_data = tl.trans(a_data)  # transpose to (MR, QK_8_0) = (12, 32)

        # 加载 b 数据块: 原始形状 (1, 1, QK_8_0, NR)
        b_data_ptr = tl.advance(b_block_ptr, offsets = (0, k_block, 0, 0))
        b_data = tl.load(b_data_ptr)  # shape: (1, 1, QK_8_0, NR)
        b_data = tl.reshape(b_data, (QK_8_0, NR))  # reshape to (32, 32)
        
        # 矩阵乘法: (MR, K) @ (K, NR) = (MR, NR) = (12, 32)
        # 直接使用 int32 避免溢出
        tmp_result = tl.dot(a_data.to(tl.int16), b_data.to(tl.int16), out_dtype=tl.int32)

        acc += tmp_result


    # 写回结果: 需要将 (MR, NR) reshape 成 (1, 1, MR, NR) 以匹配 block_ptr
    output_data = tl.reshape(acc, (1, 1, MR, NR))
    output_ptr = tl.advance(output_block_ptr, offsets=(0, 0, 0, 0))
    tl.store(output_ptr, output_data)


# ============================================================================
# 简单驱动程序 - 用于快速测试和调试
# ============================================================================

def simple_test(M=48, N=512, K=32, verbose=True):
    """简单的测试函数，用于验证 kernel 正确性
    
    Args:
        M, N, K: 矩阵维度
        verbose: 是否打印详细信息
    
    Returns:
        是否通过测试
    """
    MR = 12
    NR = 32
    QK_8_0 = 32
    
    # 验证约束
    assert M % MR == 0, f"M={M} 必须是 {MR} 的倍数"
    assert N % NR == 0, f"N={N} 必须是 {NR} 的倍数"
    assert K % QK_8_0 == 0, f"K={K} 必须是 {QK_8_0} 的倍数"
    
    if verbose:
        print(f"测试配置: M={M}, N={N}, K={K}")
        print(f"Grid: ({M//MR}, {N//NR})")
    
    # 生成测试数据
    torch.manual_seed(42)
    a_matrix = torch.randint(-10, 10, (M, K), dtype=torch.int8)
    b_matrix = torch.randint(-10, 10, (K, N), dtype=torch.int8)
    output = torch.zeros((M, N), dtype=torch.int32)
    
    # 计算参考结果 (使用 PyTorch)
    reference = torch.matmul(a_matrix.to(torch.int32), b_matrix.to(torch.int32))
    
    # 执行 Triton kernel
    grid = (M // MR, N // NR)
    i8_i8_gemm_kernel[grid](
        a_matrix_ptr=a_matrix,
        b_matrix_ptr=b_matrix,
        output_ptr=output,
        M=M,
        N=N,
        K=K,
    )
    
    # 验证结果
    max_diff = torch.max(torch.abs(output - reference)).item()
    allclose = torch.allclose(output, reference, atol=0)
    
    if verbose:
        print(f"\n结果验证:")
        print(f"  最大差异: {max_diff}")
        print(f"  结果匹配: {'✓ 通过' if allclose else '✗ 失败'}")
        
        if not allclose:
            # 显示一些不匹配的值
            diff = output - reference
            non_zero = torch.nonzero(diff)
            if len(non_zero) > 0:
                print(f"  不匹配位置数: {len(non_zero)}")
                print(f"  前5个不匹配位置:")
                for i in range(min(5, len(non_zero))):
                    pos = non_zero[i]
                    m, n = pos[0].item(), pos[1].item()
                    print(f"    [{m},{n}]: 输出={output[m,n].item()}, "
                          f"期望={reference[m,n].item()}, "
                          f"差异={diff[m,n].item()}")
    
    return allclose


def simple_benchmark(M=48, N=512, K=32, warmup=10, rep=100):
    """简单的性能测试函数
    
    Args:
        M, N, K: 矩阵维度
        warmup: 预热次数
        rep: 测试次数
    
    Returns:
        性能统计字典
    """
    MR = 12
    NR = 32
    QK_8_0 = 32
    
    # 验证约束
    assert M % MR == 0, f"M={M} 必须是 {MR} 的倍数"
    assert N % NR == 0, f"N={N} 必须是 {NR} 的倍数"
    assert K % QK_8_0 == 0, f"K={K} 必须是 {QK_8_0} 的倍数"
    
    print(f"性能测试: M={M}, N={N}, K={K}")
    
    # 生成测试数据
    torch.manual_seed(42)
    a_matrix = torch.randint(-128, 127, (M, K), dtype=torch.int8)
    b_matrix = torch.randint(-128, 127, (K, N), dtype=torch.int8)
    output = torch.zeros((M, N), dtype=torch.int32)
    
    grid = (M // MR, N // NR)
    
    # 预热
    print(f"预热 {warmup} 次...")
    for _ in range(warmup):
        i8_i8_gemm_kernel[grid](
            a_matrix_ptr=a_matrix,
            b_matrix_ptr=b_matrix,
            output_ptr=output,
            M=M, N=N, K=K,
        )
    
    # 性能测试
    print(f"测试 {rep} 次...")
    times = []
    for _ in range(rep):
        start = time.perf_counter()
        i8_i8_gemm_kernel[grid](
            a_matrix_ptr=a_matrix,
            b_matrix_ptr=b_matrix,
            output_ptr=output,
            M=M, N=N, K=K,
        )
        end = time.perf_counter()
        times.append((end - start) * 1000)  # 转换为毫秒
    
    # 统计
    times = np.array(times)
    median_ms = np.median(times)
    min_ms = np.min(times)
    max_ms = np.max(times)
    mean_ms = np.mean(times)
    
    # 计算 GOPS
    ops = 2 * M * N * K
    gops_median = ops / (median_ms * 1e6)
    
    print(f"\n性能结果:")
    print(f"  时间 (中位数): {median_ms:.4f} ms")
    print(f"  时间范围: [{min_ms:.4f}, {max_ms:.4f}] ms")
    print(f"  计算量: {ops/1e9:.2f} GOps")
    print(f"  性能: {gops_median:.2f} GOPS")
    
    return {
        'median_ms': median_ms,
        'min_ms': min_ms,
        'max_ms': max_ms,
        'mean_ms': mean_ms,
        'gops': gops_median,
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='i8_i8_gemm 简单测试和调试')
    parser.add_argument('--test', action='store_true', help='运行正确性测试')
    parser.add_argument('--bench', action='store_true', help='运行性能测试')
    parser.add_argument('--m', type=int, default=48, help='矩阵 A 的行数 (默认: 48)')
    parser.add_argument('--n', type=int, default=512, help='矩阵 B 的列数 (默认: 512)')
    parser.add_argument('--k', type=int, default=32, help='公共维度 (默认: 32)')
    parser.add_argument('--warmup', type=int, default=10, help='预热次数 (默认: 10)')
    parser.add_argument('--rep', type=int, default=100, help='测试次数 (默认: 100)')
    
    args = parser.parse_args()
    
    # 如果没有指定任何模式，默认运行测试
    if not args.test and not args.bench:
        args.test = True
    
    print("="*70)
    print("i8_i8_gemm 简单驱动程序")
    print("="*70)
    
    if args.test:
        print("\n[正确性测试]")
        print("-"*70)
        success = simple_test(M=args.m, N=args.n, K=args.k, verbose=True)
        if not success:
            print("\n⚠️  测试失败！请检查 kernel 实现")
    
    if args.bench:
        print("\n[性能测试]")
        print("-"*70)
        simple_benchmark(M=args.m, N=args.n, K=args.k, 
                        warmup=args.warmup, rep=args.rep)
    
    print("\n" + "="*70)


