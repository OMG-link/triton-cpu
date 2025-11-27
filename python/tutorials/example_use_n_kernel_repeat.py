"""
演示如何使用 n_kernel_repeat 参数来减少 kernel 启动开销

n_kernel_repeat 参数允许在单次 kernel 调用中重复执行 kernel 多次,
这样可以减少:
1. Python -> C++ 的调用开销
2. OpenMP 线程池的启动/同步开销
3. 提高性能测量的精度

使用场景:
- 性能基准测试时,希望减少启动开销占比
- 需要更精确地测量 kernel 的实际计算性能
"""

import torch
import time
import numpy as np
import triton
import triton.language as tl


@triton.jit
def simple_add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def benchmark_with_repeat(n_elements=1024, n_repeat_values=[1, 10, 100]):
    """对比不同 n_kernel_repeat 值的性能"""
    
    print(f"向量大小: {n_elements} 元素\n")
    
    # 准备数据
    x = torch.randn(n_elements, dtype=torch.float32)
    y = torch.randn(n_elements, dtype=torch.float32)
    output = torch.zeros(n_elements, dtype=torch.float32)
    
    BLOCK_SIZE = 256
    grid = lambda meta: ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    # 预热
    for _ in range(10):
        simple_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    print(f"{'n_kernel_repeat':<20} {'Total Time (ms)':<20} {'Time per iter (μs)':<25} {'Overhead ratio':<15}")
    print("-" * 80)
    
    baseline_time = None
    
    for n_repeat in n_repeat_values:
        times = []
        iterations = 100  # 外部重复次数
        
        for _ in range(iterations):
            start = time.perf_counter()
            simple_add_kernel[grid](
                x, y, output, n_elements, 
                BLOCK_SIZE=BLOCK_SIZE,
                n_kernel_repeat=n_repeat  # 关键参数
            )
            end = time.perf_counter()
            times.append((end - start) * 1000)  # 转换为毫秒
        
        median_ms = np.median(times)
        time_per_iter_us = (median_ms / n_repeat) * 1000  # 每次迭代的时间(微秒)
        
        if baseline_time is None:
            baseline_time = time_per_iter_us
            overhead_ratio = 1.0
        else:
            overhead_ratio = time_per_iter_us / baseline_time
        
        print(f"{n_repeat:<20} {median_ms:<20.4f} {time_per_iter_us:<25.4f} {overhead_ratio:<15.2f}x")
    
    print("\n说明:")
    print("- n_kernel_repeat=1 时测量的时间包含完整的 kernel 启动开销")
    print("- n_kernel_repeat 增大时,启动开销被分摊,每次迭代的时间会减少")
    print("- 理想情况下,随着 n_kernel_repeat 增大,overhead ratio 应该趋近于 1.0")
    print("- overhead ratio 与 1.0 的差距反映了启动开销的占比")


def compare_with_without_repeat():
    """对比使用和不使用 n_kernel_repeat 的性能测试精度"""
    
    n_elements = 4096
    x = torch.randn(n_elements, dtype=torch.float32)
    y = torch.randn(n_elements, dtype=torch.float32)
    output = torch.zeros(n_elements, dtype=torch.float32)
    
    BLOCK_SIZE = 256
    grid = lambda meta: ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    print("\n" + "="*80)
    print("对比测试: 传统方法 vs. 使用 n_kernel_repeat")
    print("="*80)
    
    # 方法1: 传统方法 - 多次调用 kernel
    print("\n[方法1] 传统方法: 外部循环 100 次调用 kernel")
    times_traditional = []
    for _ in range(50):
        start = time.perf_counter()
        for _ in range(100):
            simple_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        end = time.perf_counter()
        times_traditional.append((end - start) * 1000)
    
    median_traditional = np.median(times_traditional)
    time_per_call_traditional = median_traditional / 100
    
    print(f"  总时间(中位数): {median_traditional:.4f} ms")
    print(f"  单次调用时间: {time_per_call_traditional:.4f} ms")
    print(f"  包含启动开销: 是 (每次调用都有)")
    
    # 方法2: 使用 n_kernel_repeat - 单次调用,内部循环
    print("\n[方法2] 新方法: 单次调用 kernel,内部重复 100 次")
    times_repeat = []
    for _ in range(50):
        start = time.perf_counter()
        simple_add_kernel[grid](
            x, y, output, n_elements, 
            BLOCK_SIZE=BLOCK_SIZE,
            n_kernel_repeat=100  # 内部重复
        )
        end = time.perf_counter()
        times_repeat.append((end - start) * 1000)
    
    median_repeat = np.median(times_repeat)
    time_per_call_repeat = median_repeat / 100
    
    print(f"  总时间(中位数): {median_repeat:.4f} ms")
    print(f"  单次执行时间: {time_per_call_repeat:.4f} ms")
    print(f"  包含启动开销: 最小化 (只有一次)")
    
    # 分析
    print("\n[分析]")
    speedup = median_traditional / median_repeat
    overhead_reduction = (1 - time_per_call_repeat / time_per_call_traditional) * 100
    
    print(f"  方法2 总体加速: {speedup:.2f}x")
    print(f"  单次执行开销减少: {overhead_reduction:.1f}%")
    print(f"  启动开销估计: {(time_per_call_traditional - time_per_call_repeat) * 100:.2f} μs")
    
    print("\n推荐使用场景:")
    print("  ✓ 性能基准测试 - 减少测量误差")
    print("  ✓ 小型 kernel - 启动开销占比大")
    print("  ✓ 需要高精度性能数据")
    print("  ✗ 生产环境 - 通常不需要内部重复")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='演示 n_kernel_repeat 参数的使用')
    parser.add_argument('--mode', type=str, default='both', 
                       choices=['benchmark', 'compare', 'both'],
                       help='运行模式')
    parser.add_argument('--size', type=int, default=1024,
                       help='向量大小')
    
    args = parser.parse_args()
    
    if args.mode in ['benchmark', 'both']:
        print("="*80)
        print("测试不同 n_kernel_repeat 值的影响")
        print("="*80)
        benchmark_with_repeat(n_elements=args.size, n_repeat_values=[1, 10, 50, 100])
    
    if args.mode in ['compare', 'both']:
        compare_with_without_repeat()
    
    print("\n" + "="*80)
    print("完成!")
    print("="*80)
