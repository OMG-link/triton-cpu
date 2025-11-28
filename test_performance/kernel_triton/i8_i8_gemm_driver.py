"""
i8_i8_gemm 性能测试驱动程序

功能:
1. 支持不同矩阵形状的性能测试
2. 性能指标: GOPS (Giga Operations Per Second)
3. 输出详细的性能统计 (median/min/max)
4. 支持导出 CSV 结果
5. 支持生成性能对比图

运行示例:
  TRITON_ALWAYS_COMPILE=1  TRITON_CPU_BACKEND=1  python i8_i8_gemm_driver.py --shapes 48x512x32,96x1024x64 --csv-out results.csv --plot-out results.png
  TRITON_ALWAYS_COMPILE=1  TRITON_CPU_BACKEND=1  python i8_i8_gemm_driver.py --sweep  # 使用默认的多组形状测试
  TRITON_ALWAYS_COMPILE=1  TRITON_CPU_BACKEND=1  python i8_i8_gemm_driver.py --num-threads=1 --m 48 --n 512 --k 32  # 单个形状测试

注意:
  - 形状约束: M % 12 == 0, N % 32 == 0, K % 32 == 0
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from typing import List, Tuple, Dict

import numpy as np
import torch
import triton
import triton.testing as tt

# 导入 kernel
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.append(CUR_DIR)

from i8_i8_gemm import i8_i8_gemm_kernel

# ============================================================================
# 配置区域
# ============================================================================

# 原始 (K, N)  权重矩阵 shape 组合
k_n_pairs = [
    (2048, 64),
    (2048, 2048),
    (2048, 8192),
    (8192, 2048),
    (3072, 128),
    (3072, 3072),
    (3072, 8192),
    (8192, 3072),
	# qwen2.5-0.5b 869 不是 256 倍数 
    # (896, 64),
    # (896, 896),
    # (896, 4864),
    # (4864, 896),
    (768, 64),
    (768, 768),
    (768, 4864),
    (4864, 768),
    (1536, 128),
    (1536, 1536),
    (1536, 8960),
    (8960, 1536),
    (2048, 128),
    (2048, 11008),
    (11008, 2048),
    (6144, 128),
    (2048, 6144),
	# gemma-3-1b-it
    # (1152, 256),
    # (1024, 1152),
    # (1152, 6912),
    # (6912, 1152),

    (1024, 256),
    (1024, 1024),
    (1024, 6912),
    (6912, 1024),
]

# M 的取值集合
m_values = [480, 192, 144, 96, 72, 48, 24, 12]

# 根据 k_n_pairs 和 m_values 构造默认测试形状 (M, N, K)
# 对于每个 (K, N) 对，使用所有 M 值构造 (M, N, K) 测试形状
DEFAULT_SHAPES: List[Tuple[int, int, int]] = []
for k, n in k_n_pairs:
    for m in m_values:
        DEFAULT_SHAPES.append((m, n, k))

# 形状约束
MR = 12
NR = 32
QK_8_0 = 32

# 机器峰值性能配置（默认值）
DEFAULT_FREQ = 1.6  # GHz
DEFAULT_VLEN = 256  # bits
DEFAULT_DTYPE_WIDTH = 16  # int16


def calculate_peak_gops(freq: float, vlen: int, dtype_width: int = 16, threads: int = 1) -> float:
    """计算理论峰值 GOPS (考虑线程数)

    Args:
        freq: 芯片频率 (GHz)
        vlen: 向量宽度 (bits)
        dtype_width: 数据类型位宽 (bits)，int8=8
        threads: 实际参与计算的线程(或核心)数量

    Returns:
        理论峰值 GOPS (总和)

    公式推导:
        每线程向量一次 FMA 吞吐: 2 * (VLEN / dtype_width) * Freq
        总峰值: 每线程峰值 * 线程数

        Peak GOPS = 2 * (VLEN / dtype_width) * Freq * Threads

    注意: 若未显式指定线程数, 可使用 os.cpu_count() 作为近似, 或用户通过 --num-threads 指定。
    """
    elements_per_vector = vlen / dtype_width
    peak_per_thread = 2.0 * elements_per_vector * freq
    peak_total = peak_per_thread * max(1, threads)
    return peak_total


def validate_shape(M: int, N: int, K: int) -> Tuple[bool, str]:
    """验证矩阵形状是否满足约束"""
    if M % MR != 0:
        return False, f"M={M} 必须是 {MR} 的倍数"
    if N % NR != 0:
        return False, f"N={N} 必须是 {NR} 的倍数"
    if K % QK_8_0 != 0:
        return False, f"K={K} 必须是 {QK_8_0} 的倍数"
    return True, ""


# ============================================================================
# 数据生成
# ============================================================================

def generate_test_data(M: int, N: int, K: int, seed: int = 42) -> Dict:
    """生成测试数据
    
    Args:
        M: 矩阵 A 的行数
        N: 矩阵 B 的列数
        K: 公共维度
        seed: 随机种子
    
    Returns:
        包含输入矩阵和输出矩阵的字典
    """
    torch.manual_seed(seed)
    
    # 生成 int8 矩阵
    # A: [M, K], B: [K, N]
    a_matrix = torch.randint(-128, 127, (M, K), dtype=torch.int8)
    b_matrix = torch.randint(-128, 127, (K, N), dtype=torch.int8)
    output = torch.zeros((M, N), dtype=torch.int32)

    a_matrix_para = a_matrix.reshape(M//MR, MR, K//QK_8_0, QK_8_0).permute(0,2,3,1).contiguous()
    b_matrix_para = b_matrix.permute(1,0).reshape(N//NR, NR, K//QK_8_0, QK_8_0).permute(0,2,3,1).contiguous()

    output = torch.zeros((M, N), dtype=torch.int32)
    output_para = output.reshape(M//MR, MR, N//NR, NR).permute(0,2,1,3).contiguous() # [M//MR, N//NR, MR, NR]

    return {
        'a_matrix': a_matrix_para,
        'b_matrix': b_matrix_para,
        'output': output_para,
        'M': M,
        'N': N,
        'K': K,
    }


# ============================================================================
# 性能测试核心
# ============================================================================

def benchmark_single_shape(
    M: int, 
    N: int, 
    K: int,
    warmup_ms: int = 25,
    rep_ms: int = 100,
    threads: int = 1,
    peak_gops: float = None,
    n_kernel_repeat: int = 10
) -> Dict:
    """对单个矩阵形状进行性能测试
    
    Args:
        M, N, K: 矩阵维度
        warmup_ms: 预热时间（毫秒）
        rep_ms: 测试时间（毫秒）
        num_threads: CPU 线程数
        peak_gops: 理论峰值 GOPS（用于计算效率）
        n_kernel_repeat: kernel 内部重复次数（减少启动开销）
    
    Returns:
        包含性能统计的字典
    """
    # 验证形状
    valid, msg = validate_shape(M, N, K)
    if not valid:
        raise ValueError(f"无效的矩阵形状: {msg}")
    
    
    # 生成测试数据
    data = generate_test_data(M, N, K)
    a_matrix = data['a_matrix']
    b_matrix = data['b_matrix']
    output = data['output']
    
    # 计算 grid 配置
    grid = (M // MR, N // NR)

    # 根据测试数据大小动态选择 n_kernel_repeat
    # 根据 峰值 GOPS 计算需要考虑重复次数，使得每launch 一次 kernel 最少能跑 10s
    ops_per_call = 2 * M * N * K
    if peak_gops:
        n_kernel_repeat = max(n_kernel_repeat, int(peak_gops * 1e9 / ops_per_call))

    # 定义执行函数
    def run_kernel():
        i8_i8_gemm_kernel[grid](
            a_matrix_ptr=a_matrix,
            b_matrix_ptr=b_matrix,
            output_ptr=output,
            M=M,
            N=N,
            K=K,
            num_threads=threads,
            n_kernel_repeat=n_kernel_repeat
        )
    
    # 使用 triton.testing.do_bench 进行性能测试
    print(f"  使用 Triton do_bench 测试 (warmup={warmup_ms}ms, rep={rep_ms}ms, n_kernel_repeat={n_kernel_repeat})...")
    median_ms, min_ms, max_ms = tt.do_bench(
        run_kernel, 
        warmup=warmup_ms, 
        rep=rep_ms, 
        quantiles=[0.5, 0.2, 0.8]
    )
    
    # 计算 GOPS
    # GEMM 的计算量: 2*M*N*K (乘法 + 加法)
    # 注意: 如果使用了 n_kernel_repeat,需要乘以重复次数来计算总计算量
    total_ops = ops_per_call * n_kernel_repeat
    gops_median = total_ops / (median_ms * 1e6)  # GOPS
    gops_min = total_ops / (max_ms * 1e6)  # 最大时间对应最小GOPS
    gops_max = total_ops / (min_ms * 1e6)  # 最小时间对应最大GOPS
    
    # 计算效率（如果提供了峰值GOPS）
    efficiency_median = (gops_median / peak_gops * 100.0) if peak_gops else None
    efficiency_min = (gops_min / peak_gops * 100.0) if peak_gops else None
    efficiency_max = (gops_max / peak_gops * 100.0) if peak_gops else None
    
    result = {
        'M': M,
        'N': N,
        'K': K,
        'median_ms': median_ms,
        'min_ms': min_ms,
        'max_ms': max_ms,
        'gops_median': gops_median,
        'gops_min': gops_min,
        'gops_max': gops_max,
        'peak_gops': peak_gops,
        'efficiency_median': efficiency_median,
        'efficiency_min': efficiency_min,
        'efficiency_max': efficiency_max,
        'ops': total_ops,
        'n_kernel_repeat': n_kernel_repeat,
    }
    
    return result


def print_result(result: Dict):
    """打印单个测试结果"""
    print(f"\n{'='*70}")
    print(f"矩阵形状: M={result['M']}, N={result['N']}, K={result['K']}")
    print(f"{'='*70}")
    print(f"总计算量: {result['ops']/1e9:.2f} GOps")
    print(f"\n时间统计 (ms):")
    print(f"  Median: {result['median_ms']:.4f}")
    print(f"  Min:    {result['min_ms']:.4f}")
    print(f"  Max:    {result['max_ms']:.4f}")
    print(f"\n性能 (GOPS):")
    print(f"  Median: {result['gops_median']:.2f}")
    print(f"  Min:    {result['gops_min']:.2f}")
    print(f"  Max:    {result['gops_max']:.2f}")
    if result.get('peak_gops'):
        print(f"  Peak:   {result['peak_gops']:.2f}")
        print(f"\n效率 (%):")
        print(f"  Median: {result['efficiency_median']:.2f}%")
        print(f"  Min:    {result['efficiency_min']:.2f}%")
        print(f"  Max:    {result['efficiency_max']:.2f}%")
    print(f"{'='*70}\n")


# ============================================================================
# 批量测试
# ============================================================================

def benchmark_shapes(
    shapes: List[Tuple[int, int, int]],
    warmup_ms: int = 25,
    rep_ms: int = 100,
    num_threads: int = None,
    freq: float = None,
    vlen: int = None,
    n_kernel_repeat: int = 10
) -> List[Dict]:
    """对多个矩阵形状进行性能测试
    
    Args:
        shapes: 要测试的矩阵形状列表 [(M1,N1,K1), (M2,N2,K2), ...]
        warmup_ms: 预热时间（毫秒）
        rep_ms: 测试时间（毫秒）
        num_threads: CPU 线程数
        freq: 芯片频率 (GHz)
        vlen: 向量宽度 (bits)
    
    Returns:
        所有测试结果的列表
    """
    results = []
    
    # 计算理论峰值GOPS
    peak_gops = calculate_peak_gops(freq, vlen, DEFAULT_DTYPE_WIDTH, threads=num_threads) 
    print(f"\n开始批量测试，共 {len(shapes)} 个形状")
    print(f"配置: warmup={warmup_ms}ms, rep={rep_ms}ms, num_threads={num_threads}")
    if peak_gops:
        print(f"机器参数: Freq={freq} GHz, VLEN={vlen} bits")
        print(f"理论峰值: {peak_gops:.2f} GOPS\n")
    else:
        print()
    
    for idx, (M, N, K) in enumerate(shapes, 1):
        print(f"[{idx}/{len(shapes)}] 测试形状: M={M}, N={N}, K={K}")
        
        # 验证形状
        valid, msg = validate_shape(M, N, K)
        if not valid:
            print(f"  ⚠️  跳过: {msg}\n")
            continue
        
        try:
            result = benchmark_single_shape(M, N, K, warmup_ms, rep_ms, num_threads, peak_gops)
            result['threads_used'] = num_threads
            results.append(result)
            perf_str = f"{result['gops_median']:.2f} GOPS"
            if result.get('efficiency_median'):
                perf_str += f" ({result['efficiency_median']:.2f}% eff.)"
            if result.get('median_ms'):
                perf_str += f", Median Time: {result['median_ms']:.4f} ms"
            print(f"  ✓ 完成: {perf_str}")
        except Exception as e:
            print(f"  ✗ 错误: {e}")
        
        print()
    
    return results


# ============================================================================
# 结果导出
# ============================================================================

def save_results_csv(results: List[Dict], csv_path: str):
    """保存结果到 CSV 文件"""
    if not results:
        print("没有结果可保存")
        return
    
    fieldnames = [
        'M', 'N', 'K', 'ops',
        'median_ms', 'min_ms', 'max_ms',
        'gops_median', 'gops_min', 'gops_max',
        'peak_gops', 'efficiency_median', 'efficiency_min', 'efficiency_max',
        'threads_used'
    ]
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({k: result[k] for k in fieldnames})
    
    print(f"✓ CSV 结果已保存到: {csv_path}")


def plot_results(results: List[Dict], output_path: str, metric: str = 'gops_median'):
    """绘制性能对比图
    
    Args:
        results: 测试结果列表
        output_path: 图片输出路径
        metric: 绘图的性能指标 (gops_median, gops_mean 等)
    """
    if not results:
        print("没有结果可绘图")
        return
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  未安装 matplotlib，跳过绘图")
        return
    
    # 准备数据
    shape_labels = [f"{r['M']}x{r['N']}x{r['K']}" for r in results]
    values = [r[metric] for r in results]
    
    # 绘图
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(shape_labels)), values, alpha=0.7, color='steelblue')
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, values)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('矩阵形状 (M×N×K)', fontsize=12, fontweight='bold')
    plt.ylabel('性能 (GOPS)', fontsize=12, fontweight='bold')
    plt.title('i8_i8_gemm 性能测试', fontsize=14, fontweight='bold')
    plt.xticks(range(len(shape_labels)), shape_labels, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 图表已保存到: {output_path}")
    plt.close()


def print_summary(results: List[Dict]):
    """打印测试总结"""
    if not results:
        return
    
    has_efficiency = results[0].get('efficiency_median') is not None
    
    print(f"\n{'='*70}")
    print("测试总结")
    print(f"{'='*70}")
    if has_efficiency:
        print(f"{'形状 (M×N×K)':<20} {'中位数(ms)':<15} {'GOPS':<12} {'效率(%)':<10} {'线程':<6}")
    else:
        print(f"{'形状 (M×N×K)':<20} {'中位数(ms)':<15} {'GOPS(median)':<15} {'线程':<6}")
    print(f"{'-'*70}")
    
    for r in results:
        shape_str = f"{r['M']}×{r['N']}×{r['K']}"
        threads_used = r.get('threads_used', '-')
        if has_efficiency:
            print(f"{shape_str:<20} {r['median_ms']:<15.4f} {r['gops_median']:<12.2f} {r['efficiency_median']:<10.2f} {threads_used:<6}")
        else:
            print(f"{shape_str:<20} {r['median_ms']:<15.4f} {r['gops_median']:<15.2f} {threads_used:<6}")
    
    # 统计最佳性能
    best_result = max(results, key=lambda x: x['gops_median'])
    print(f"\n最佳性能:")
    print(f"  形状: M={best_result['M']}, N={best_result['N']}, K={best_result['K']}")
    print(f"  性能: {best_result['gops_median']:.2f} GOPS")
    if has_efficiency:
        print(f"  效率: {best_result['efficiency_median']:.2f}%")
        if best_result.get('peak_gops'):
            print(f"  峰值: {best_result['peak_gops']:.2f} GOPS")
    print(f"{'='*70}\n")


# ============================================================================
# 命令行接口
# ============================================================================

def parse_shapes_arg(shapes_str: str) -> List[Tuple[int, int, int]]:
    """解析形状参数字符串"""
    shapes = []
    for item in shapes_str.split(','):
        item = item.strip()
        if not item:
            continue
        try:
            parts = item.lower().split('x')
            if len(parts) != 3:
                print(f"⚠️  忽略无效形状 '{item}' (格式应为 MxNxK)")
                continue
            M, N, K = int(parts[0]), int(parts[1]), int(parts[2])
            shapes.append((M, N, K))
        except Exception as e:
            print(f"⚠️  忽略无效形状 '{item}': {e}")
    return shapes


def main():
    parser = argparse.ArgumentParser(
        description='i8_i8_gemm 性能测试驱动程序',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # 测试模式
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--sweep',
        action='store_true',
        help='使用默认形状列表进行批量测试'
    )
    mode_group.add_argument(
        '--shapes',
        type=str,
        help='指定测试形状列表，格式: M1xN1xK1,M2xN2xK2,... 例如: 48x512x32,96x1024x64'
    )
    
    # 单一形状测试参数
    parser.add_argument('--m', type=int, help='矩阵 A 的行数')
    parser.add_argument('--n', type=int, help='矩阵 B 的列数')
    parser.add_argument('--k', type=int, help='公共维度')
    
    # 性能测试参数
    parser.add_argument(
        '--warmup',
        type=int,
        default=25,
        help='预热时间（毫秒） (默认: 25)'
    )
    parser.add_argument(
        '--rep',
        type=int,
        default=100,
        help='测试时间（毫秒） (默认: 100)'
    )
    parser.add_argument(
        '--num-threads',
        type=int,
        default=None,
        help='CPU 线程数 (默认: 使用系统默认)'
    )
    
    # 机器峰值性能参数
    parser.add_argument(
        '--freq',
        type=float,
        default=DEFAULT_FREQ,
        help=f'芯片频率 GHz (默认: {DEFAULT_FREQ})'
    )
    parser.add_argument(
        '--vlen',
        type=int,
        default=DEFAULT_VLEN,
        help=f'向量宽度 bits (默认: {DEFAULT_VLEN})'
    )
    
    # 输出参数
    parser.add_argument(
        '--csv-out',
        type=str,
        help='CSV 结果输出路径'
    )
    parser.add_argument(
        '--plot-out',
        type=str,
        help='性能图输出路径'
    )

    parser.add_argument(
        '--n-kernel-repeat',
        type=int,
        default=20,
        help='kernel repeat nums to reduce jit launch overhead'
    )
    
    args = parser.parse_args()
    
    # 确定测试形状
    shapes = []
    if args.sweep:
        shapes = DEFAULT_SHAPES
        print(f"使用默认形状列表 (共 {len(shapes)} 个)")
    elif args.shapes:
        shapes = parse_shapes_arg(args.shapes)
        if not shapes:
            print("错误: 未提供有效的形状")
            sys.exit(1)
    elif args.m and args.n and args.k:
        shapes = [(args.m, args.n, args.k)]
    else:
        print("错误: 请指定测试模式 (--sweep, --shapes, 或 --m/--n/--k)")
        parser.print_help()
        sys.exit(1)
    
    import os as _os
    # 计算有效线程数: 优先使用 --num-threads，其次测量，再次回退为 1
    if args.num_threads:
        effective_threads = args.num_threads
    else:
        effective_threads = _os.cpu_count() or 1  # 默认用全部可用

    peak_gops = calculate_peak_gops(args.freq, args.vlen, DEFAULT_DTYPE_WIDTH, threads=effective_threads)
    
    # 执行测试
    print(f"\n{'='*70}")
    print("i8_i8_gemm 性能测试")
    print(f"{'='*70}")
    print(f"测试形状数量: {len(shapes)}")
    print(f"预热时间: {args.warmup} ms")
    print(f"测试时间: {args.rep} ms")
    print(f"线程数(有效): {effective_threads}  (用户指定: {args.num_threads if args.num_threads else '未指定'})")
    print(f"\n机器配置:")
    print(f"  频率: {args.freq} GHz")
    print(f"  向量宽度: {args.vlen} bits")
    print(f"  数据类型: int8 ({DEFAULT_DTYPE_WIDTH} bits)")
    print(f"  理论峰值: {peak_gops:.2f} GOPS  (公式: 2*(VLEN/{DEFAULT_DTYPE_WIDTH})*Freq*Threads)")
    print(f"{'='*70}\n")
    
    results = benchmark_shapes(
        shapes=shapes,
        warmup_ms=args.warmup,
        rep_ms=args.rep,
        num_threads=effective_threads,
        freq=args.freq,
        vlen=args.vlen,
        n_kernel_repeat=args.n_kernel_repeat,
    )
    
    # 输出结果
    if results:
        print_summary(results)
        
        # 保存 CSV
        if args.csv_out:
            save_results_csv(results, args.csv_out)
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            default_csv = f'i8_i8_gemm_results_{timestamp}.csv'
            save_results_csv(results, default_csv)
        
        # 绘图
        if args.plot_out:
            plot_results(results, args.plot_out)
        elif len(results) > 1:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            default_plot = f'i8_i8_gemm_results_{timestamp}.png'
            plot_results(results, default_plot)
    else:
        print("⚠️  没有成功完成的测试")


if __name__ == '__main__':
    main()
