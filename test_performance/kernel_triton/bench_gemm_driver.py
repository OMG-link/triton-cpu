"""统一内核性能测试驱动

功能:
1. 注册不同 GEMM 内核 (目前: q4k_q8k, iq4k_q8k, q40_q80)
2. 注册需要测试的矩阵形状 (可命令行覆盖)
3. 执行性能测试 (median/min/max ms, GFLOPS, 利用率)
4. 导出 CSV 结果
5. 生成性能折线图 (x: shape, y: GFLOPS 或利用率)

运行示例:

# 测试所有 kernel 是否可以正常运行 
TRITON_ALWAYS_COMPILE=1 TRITON_CPU_BACKEND=1 python bench_gemm_driver.py  --shapes 192x2048x2048 --kernels q40_q80,q4k_q8k,iq4k_q8k,iq4k_q8k_stlb,iq4k_q8k_stlb_wogather --metric gflops --num-threads 8 --warmup 10 --rounds 100 --freq 1.6 --vlen 256 

# 测试所有 kernel 在所有默认shape 下性能，不指定默认全部 
TRITON_ALWAYS_COMPILE=1 TRITON_CPU_BACKEND=1 python bench_gemm_driver.py --kernels q40_q80,q4k_q8k,iq4k_q8k,iq4k_q8k_stlb,iq4k_q8k_stlb_wogather --metric gflops --csv-out bench.csv --png-out bench.png --num-threads 1 --warmup 10 --rounds 100 --freq 1.6 --vlen 256 



注意:
  - 可用的 kernel: q40_q80, q4k_q8k, iq4k_q8k
  - q40_q80 约束: M%12==0, N%32==0, K%32==0
  - q4k_q8k/iq4k_q8k 约束: M%12==0, N%32==0, K%256==0
  - 使用 triton.testing.do_bench 进行精确的性能测试
  - warmup: 预热次数，rounds: 测试轮数
  - num-threads: CPU核心/线程数，用于多核理论峰值计算 (峰值 = 2 * VLEN/dtype_width * Freq * num_cores)
  - n_kernel_repeat: 在单次 kernel 调用中重复执行的次数 
    * 减少 Python->C++ 调用开销和 OpenMP 线程池启动开销 
    * 提高小规模 kernel 的性能测量精度 
    * 建议值: 小型 kernel 用 10-100, 大型 kernel 用 1 即可 
"""

from __future__ import annotations
import argparse
import os
import sys
import csv
from typing import List, Tuple, Callable, Dict, Any
import time

import torch
import triton
import triton.testing as tt

# 保证可以从当前目录导入内核脚本
CUR_DIR = os.path.dirname(os.path.abspath(__file__))

if CUR_DIR not in sys.path:
    sys.path.append(CUR_DIR)

# 直接导入 kernel 函数
from q4k_q8k_gemm import q4k_q8k_matmul_kernel  # noqa: E402
from iq4k_q8k_gemm import iq4k_q8k_matmul_kernel  # noqa: E402
from q40_q80_gemm import q40_q80_gemm_kernel  # noqa: E402
from iq4k_q8k_gemm_stlb import iq4k_q8k_matmul_kernel as iq4k_q8k_matmul_kernel_stlb
from iq4k_q8k_gemm_stlb_without_gather import iq4k_q8k_matmul_kernel as iq4k_q8k_matmul_kernel_stlb_wogather

try:
    from dataclasses import dataclass
except ImportError:
    raise RuntimeError("需要 Python 3.7+ 支持 dataclasses")

# -----------------------
# Kernel 规范定义
# -----------------------

@dataclass
class KernelSpec:
    name: str
    kernel_fn: Callable
    gen_dataset: Callable
    estimate_ops: Callable
    grid_fn: Callable
    param_builder: Callable
    constraints: Dict[str, int]
    compute_dtype: str = 'int16'
    dtype_width: int = 16  # 位宽

# -----------------------
# q40_q80 kernel 相关函数
# -----------------------

def q40_gen_dataset(M, K, N, seed=42):
    """生成 q40_q80 测试数据"""
    torch.manual_seed(seed)
    MR, NR, QK_8_0 = 12, 32, 32
    Mb, Nb, Kblocks = M // MR, N // NR, K // QK_8_0
    QK_4_0_DATA_SIZE = QK_8_0 // 2
    
    q8_0_matrix = torch.randint(-128, 127, (Mb, Kblocks, 2, QK_8_0//2, MR), dtype=torch.int8)
    q8_0_scale = torch.rand((Mb, Kblocks, MR), dtype=torch.float32)
    q4_0_matrix = torch.randint(0, 255, (Nb, Kblocks, QK_4_0_DATA_SIZE, NR), dtype=torch.uint8)
    q4_0_scale = torch.rand((Nb, Kblocks, NR), dtype=torch.float32)
    output = torch.empty((M, N), dtype=torch.float32)
    
    return {
        'q4_0_matrix': q4_0_matrix,
        'q4_0_scale': q4_0_scale,
        'q8_0_matrix': q8_0_matrix,
        'q8_0_scale': q8_0_scale,
        'output': output
    }

def q40_estimate_ops(M, N, K):
    """估算 q40_q80 的操作数"""
    return 2.0 * M * N * K

def q40_grid(M, N, K):
    """返回 q40_q80 的 grid 配置"""
    return (M // 12, N // 32)

def q40_param_builder(data, M, N, K, threads):
    """构建 q40_q80 kernel 参数"""
    return {
        'q4_0_matrix_ptr': data['q4_0_matrix'],
        'q4_0_scale_ptr': data['q4_0_scale'],
        'q8_0_matrix_ptr': data['q8_0_matrix'],
        'q8_0_scale_ptr': data['q8_0_scale'],
        'output_ptr': data['output'],
        'M': M, 'N': N, 'K': K,
        'num_threads': threads
    }

# -----------------------
# q4k_q8k kernel 相关函数
# -----------------------

def q4k_gen_dataset(M, K, N, seed=42):
    """生成 q4k_q8k 测试数据"""
    torch.manual_seed(seed)
    MR, NR, QK_K = 12, 32, 256
    Mb, Nb, Ksup = M // MR, N // NR, K // QK_K
    NUM_SUB_BLOCKS, SUB_BLOCK_SIZE = 8, 32
    QK_SB_K = 32
    
    # q8k 数据: (M//MR, K//QK_K, QK_K//QK_SB_K, QK_SB_K, MR)
    q8k_matrix = torch.randint(-128, 127, (Mb, Ksup, NUM_SUB_BLOCKS, QK_SB_K, MR), dtype=torch.int8)
    q8k_d = torch.rand((Mb, Ksup, MR), dtype=torch.float32)
    q8k_bsums = torch.randint(-32768, 32767, (Mb, Ksup, NUM_SUB_BLOCKS, MR), dtype=torch.int16)
    
    # q4k 数据: (N//NR, K//QK_K, QK_K//QK_SB_K, QK_SB_K, NR)
    q4k_matrix = torch.randint(0, 15, (Nb, Ksup, NUM_SUB_BLOCKS, QK_SB_K, NR), dtype=torch.int8)
    # b_scales: int8, shape (N//NR, K//QK_K, QK_K//QK_SB_K, NR)
    q4k_scales = torch.randint(-32, 31, (Nb, Ksup, NUM_SUB_BLOCKS, NR), dtype=torch.int8)
    # b_mins: int8, shape (N//NR, K//QK_K, QK_K//QK_SB_K, NR)
    q4k_mins = torch.randint(-128, 127, (Nb, Ksup, NUM_SUB_BLOCKS, NR), dtype=torch.int8)
    # b_d, b_dmin: float16
    q4k_d = torch.rand((Nb, Ksup, NR), dtype=torch.float16)
    q4k_dmin = torch.rand((Nb, Ksup, NR), dtype=torch.float16)
    
    output = torch.empty((M, N), dtype=torch.float32)
    
    return {
        'q8k_matrix': q8k_matrix, 'q8k_d': q8k_d, 'q8k_bsums': q8k_bsums,
        'q4k_matrix': q4k_matrix, 'q4k_scale': q4k_scales, 'q4k_mins': q4k_mins,
        'q4k_d': q4k_d, 'q4k_dmin': q4k_dmin, 'output': output
    }

def q4k_estimate_ops(M, N, K):
    """估算 q4k_q8k 的操作数"""
    return 2.0 * M * N * K

def q4k_grid(M, N, K):
    """返回 q4k_q8k 的 grid 配置"""
    return (M // 12, N // 32)

def q4k_param_builder(data, M, N, K, threads):
    """构建 q4k_q8k kernel 参数"""
    return {
        'q8k_matrix_ptr': data['q8k_matrix'],
        'q8k_bsums_ptr': data['q8k_bsums'],
        'q8k_d_ptr': data['q8k_d'],
        'q4k_matrix_ptr': data['q4k_matrix'],
        'q4k_scale_ptr': data['q4k_scale'],
        'q4k_mins_ptr': data['q4k_mins'],
        'q4k_d_ptr': data['q4k_d'],
        'q4k_dmin_ptr': data['q4k_dmin'],
        'output_ptr': data['output'],
        'M': M, 'N': N, 'K': K,
        'num_threads': threads
    }

# -----------------------
# iq4k_q8k kernel 相关函数
# -----------------------

def iq4k_gen_dataset(M, K, N, seed=42):
    """生成 iq4k_q8k 测试数据"""
    torch.manual_seed(seed)
    MR, NR, QK_K = 12, 32, 256
    Mb, Nb, Ksup = M // MR, N // NR, K // QK_K
    NUM_SUB_BLOCKS, SUB_BLOCK_SIZE = 8, 32
    QK_4_K_SUB_BLOCK_DATA_SIZE = 16
    
    q8k_matrix = torch.randint(-128, 127, (Mb, Ksup, NUM_SUB_BLOCKS, 2, SUB_BLOCK_SIZE//2, MR), dtype=torch.int8)
    q8k_d = torch.rand((Mb, Ksup, MR), dtype=torch.float32)
    iq4k_matrix = torch.randint(0, 255, (Nb, Ksup, NUM_SUB_BLOCKS, QK_K//2, NR), dtype=torch.uint8)
    iq4k_d = torch.rand((Nb, Ksup, NR), dtype=torch.float32)
    iq4k_extra = torch.randint(0, 0xFFFF, (Nb, Ksup, NR), dtype=torch.int32).to(torch.uint16)
    iq4k_scale_l = torch.randint(0, 255, (Nb, Ksup, NUM_SUB_BLOCKS, NR), dtype=torch.uint8)
    iq4k_scale_h = torch.randint(0, 255, (Nb, Ksup, (NUM_SUB_BLOCKS * 2)//4, NR), dtype=torch.uint8)
    output = torch.empty((M, N), dtype=torch.float32)
    
    return {
        'iq4k_matrix': iq4k_matrix, 'iq4k_d': iq4k_d, 'iq4k_extra': iq4k_extra,
        'iq4k_scale_l': iq4k_scale_l, 'iq4k_scale_h': iq4k_scale_h,
        'q8k_matrix': q8k_matrix, 'q8k_d': q8k_d, 'output': output
    }

def iq4k_estimate_ops(M, N, K):
    """估算 iq4k_q8k 的操作数"""
    return 2.0 * M * N * K

def iq4k_grid(M, N, K):
    """返回 iq4k_q8k 的 grid 配置"""
    return (M // 12, N // 32)

def iq4k_param_builder(data, M, N, K, threads):
    """构建 iq4k_q8k kernel 参数"""
    return {
        'iq4k_matrix_ptr': data['iq4k_matrix'], 'iq4k_d_ptr': data['iq4k_d'],
        'iq4k_extra_ptr': data['iq4k_extra'], 'iq4k_scale_l_ptr': data['iq4k_scale_l'],
        'iq4k_scale_h_ptr': data['iq4k_scale_h'], 'q8k_matrix_ptr': data['q8k_matrix'],
        'q8k_d_ptr': data['q8k_d'], 'output_ptr': data['output'],
        'M': M, 'N': N, 'K': K, 'num_threads': threads
    }

# -----------------------
# 注册内核
# -----------------------

KERNEL_SPECS = [
    KernelSpec(
        name='q40_q80',
        kernel_fn=q40_q80_gemm_kernel,
        gen_dataset=q40_gen_dataset,
        estimate_ops=q40_estimate_ops,
        grid_fn=q40_grid,
        param_builder=q40_param_builder,
        constraints={'M': 12, 'N': 32, 'K': 32},
        compute_dtype='int16',
        dtype_width=16
    ),
    KernelSpec(
        name='q4k_q8k',
        kernel_fn=q4k_q8k_matmul_kernel,
        gen_dataset=q4k_gen_dataset,
        estimate_ops=q4k_estimate_ops,
        grid_fn=q4k_grid,
        param_builder=q4k_param_builder,
        constraints={'M': 12, 'N': 32, 'K': 256},
        compute_dtype='int16',
        dtype_width=16
    ),
    KernelSpec(
        name='iq4k_q8k',
        kernel_fn=iq4k_q8k_matmul_kernel,
        gen_dataset=iq4k_gen_dataset,
        estimate_ops=iq4k_estimate_ops,
        grid_fn=iq4k_grid,
        param_builder=iq4k_param_builder,
        constraints={'M': 12, 'N': 32, 'K': 256},
        compute_dtype='int16',
        dtype_width=16
    ),
    KernelSpec(
        name='iq4k_q8k_stlb',
        kernel_fn=iq4k_q8k_matmul_kernel_stlb,
        gen_dataset=iq4k_gen_dataset,
        estimate_ops=iq4k_estimate_ops,
        grid_fn=iq4k_grid,
        param_builder=iq4k_param_builder,
        constraints={'M': 12, 'N': 32, 'K': 256},
        compute_dtype='int16',
        dtype_width=16
    ),
    KernelSpec(
        name='iq4k_q8k_stlb_wogather',
        kernel_fn=iq4k_q8k_matmul_kernel_stlb_wogather,
        gen_dataset=iq4k_gen_dataset,
        estimate_ops=iq4k_estimate_ops,
        grid_fn=iq4k_grid,
        param_builder=iq4k_param_builder,
        constraints={'M': 12, 'N': 32, 'K': 256},
        compute_dtype='int16',
        dtype_width=16
    ),
]# -----------------------

# 性能计算工具函数
# -----------------------
def calc_gops_per_second(ms: float, M: int, N: int, K: int, dtype_width: int) -> float:
    """
    计算 OPS (Operations Per Second)
    对于 GEMM: 总操作数 = 2 * M * N * K (每个输出元素需要 K 次乘加)
    返回单位：GOPS (Giga Operations Per Second)
    """
    total_ops = 2.0 * M * N * K
    ops_per_sec = total_ops / (ms * 1e-3)  # ops/s
    return ops_per_sec * 1e-9  # GOPS

def calc_peak_ops(freq: float, vlen: int, dtype_width: int, num_cores: int = 1) -> float:
    """
    计算理论峰值 OPS（考虑多核）
    峰值 = 2 * (VLEN / dtype_width) * Freq * num_cores
    返回单位：GOPS
    
    Args:
        freq: 芯片频率 (GHz)
        vlen: 向量宽度 (bits)
        dtype_width: 数据类型宽度 (bits)
        num_cores: CPU 核心数
    """
    return 2.0 * (vlen / dtype_width) * freq * num_cores

# -----------------------
# 默认形状注册区 (可在此扩展)
# -----------------------

# 原始 (K, N)  权重矩阵 shape 组合
k_n_pairs = [
    (2048, 64),
    (2048, 2048),
    (2048, 8192),
    (8192, 2048),
    # (3072, 128),
    # (3072, 3072),
    # (3072, 8192),
    # (8192, 3072),
	# qwen2.5-0.5b 869 不是 256 倍数 
    # (896, 64),
    # (896, 896),
    # (896, 4864),
    # (4864, 896),
    # (768, 64),
    # (768, 768),
    # (768, 4864),
    # (4864, 768),
    # (1536, 128),
    # (1536, 1536),
    # (1536, 8960),
    # (8960, 1536),
    # (2048, 128),
    # (2048, 11008),
    # (11008, 2048),
    # (6144, 128),
    # (2048, 6144),
	# gemma-3-1b-it
    # (1152, 256),
    # (1024, 1152),
    # (1152, 6912),
    # (6912, 1152),
    # (1024, 256),
    # (1024, 1024),
    # (1024, 6912),
    # (6912, 1024),
]

# M 的取值集合
m_values = [480, 192, 48]

DEFAULT_SHAPES: List[Tuple[int,int,int]] = [
]

for (K,N) in k_n_pairs:
    for M in m_values:
        DEFAULT_SHAPES.append((M, K, N))

def parse_shapes_arg(shapes_str: str) -> List[Tuple[int,int,int]]:
    res = []
    for item in shapes_str.split(','):
        item = item.strip()
        if not item:
            continue
        try:
            m,k,n = item.lower().split('x')
            res.append((int(m), int(k), int(n)))
        except Exception as e:
            print(f"忽略无效形状 '{item}': {e}")
    return res

def run_single_benchmark(spec: KernelSpec, M: int, K: int, N: int, 
                         warmup: int, rounds: int, num_threads: int, n_kernel_repeat: int = 1) -> Tuple[float, float, float]:
    """运行单个形状的性能测试,返回 (median_ms, min_ms, max_ms)
    
    参数:
        n_kernel_repeat: 在单次 kernel 调用中重复执行的次数，用于减少启动开销影响
    """
    # 生成数据
    data = spec.gen_dataset(M, K, N)
    grid = spec.grid_fn(M, N, K)
    params = spec.param_builder(data, M, N, K, num_threads)
    
    # 定义执行函数
    def run_kernel():
        spec.kernel_fn[grid](**params, n_kernel_repeat=n_kernel_repeat)
    
    # 使用 tt.do_bench 进行性能测试
    # do_bench 返回 [median, min, max] 对应 quantiles=[0.5, 0.2, 0.8]
    median_ms, min_ms, max_ms = tt.do_bench(
        run_kernel,
        warmup=warmup,
        rep=rounds,
        quantiles=[0.5, 0.2, 0.8]
    )
    
    # 除以 n_kernel_repeat 得到单次执行的时间
    median_ms /= n_kernel_repeat
    min_ms /= n_kernel_repeat
    max_ms /= n_kernel_repeat
    
    return median_ms, min_ms, max_ms

def run_bench(shapes: List[Tuple[int,int,int]], warmup: int, rounds: int, 
              num_threads: int, freq: float, vlen: int, selected_kernels: List[str] = None, n_kernel_repeat: int = 1):
    all_results = []
    
    # 如果指定了 kernel 列表，过滤出匹配的 kernel
    kernels_to_test = KERNEL_SPECS
    if selected_kernels:
        kernels_to_test = [spec for spec in KERNEL_SPECS if spec.name in selected_kernels]
        if not kernels_to_test:
            print(f"错误: 未找到匹配的 kernel。可用的 kernel: {[s.name for s in KERNEL_SPECS]}")
            return []
        print(f"已选择 kernel: {[s.name for s in kernels_to_test]}")
    
    for kernel_idx, spec in enumerate(kernels_to_test, 1):
        name = spec.name
        dtype_width = spec.dtype_width
        compute_dtype = spec.compute_dtype
        constraints = spec.constraints
        
        # 计算该 kernel 的理论峰值（考虑多核）
        peak_gops = calc_peak_ops(freq, vlen, dtype_width, num_threads)
        
        # 过滤不满足约束的形状
        valid_shapes = []
        for (M,K,N) in shapes:
            if (M % constraints.get('M', 1) or 
                N % constraints.get('N', 1) or 
                K % constraints.get('K', 1)):
                print(f"[跳过] kernel={name} shape=({M},{K},{N}) 不满足块约束")
                continue
            valid_shapes.append((M,K,N))
        
        if not valid_shapes:
            print(f"[警告] kernel={name} 无有效形状可测试")
            continue
        
        print(f"\n{'='*80}")
        print(f"[Kernel {kernel_idx}/{len(kernels_to_test)}] {name}")
        print(f"  计算数据类型: {compute_dtype} ({dtype_width} bits)")
        print(f"  理论峰值: {peak_gops:.2f} GOPS@{compute_dtype} ({num_threads} cores)")
        print(f"  块约束: M%{constraints.get('M',1)}==0, N%{constraints.get('N',1)}==0, K%{constraints.get('K',1)}==0")
        print(f"  测试形状数量: {len(valid_shapes)}")
        if n_kernel_repeat > 1:
            print(f"  内核重复次数: {n_kernel_repeat}x (减少启动开销影响)")
        print(f"{'='*80}")
        
        # 逐个测试并实时显示进度
        for idx, (M, K, N) in enumerate(valid_shapes, 1):
            # 根据测试数据大小动态选择 n_kernel_repeat
            # 根据 峰值 GOPS 计算需要考虑重复次数，使得每launch 一次 kernel 最少能跑 10s
            ops_per_call = 2 * M * N * K
            n_kernel_repeat = max(n_kernel_repeat, int(peak_gops * 1e9 / ops_per_call))
            
            # 调用 benchmark 函数
            median_ms, min_ms, max_ms = run_single_benchmark(
                spec, M, K, N, warmup, rounds, num_threads, n_kernel_repeat
            )
            
            # 计算性能指标
            gops_med = calc_gops_per_second(median_ms, M, N, K, dtype_width)
            gops_min = calc_gops_per_second(max_ms, M, N, K, dtype_width)
            gops_max = calc_gops_per_second(min_ms, M, N, K, dtype_width)
            
            util_med = (gops_med / peak_gops * 100.0) if peak_gops > 0 else 0.0
            util_min = (gops_min / peak_gops * 100.0) if peak_gops > 0 else 0.0
            util_max = (gops_max / peak_gops * 100.0) if peak_gops > 0 else 0.0
            
            # 实时输出进度
            print(f"  [{idx:3d}/{len(valid_shapes)}] M={M:4d} K={K:5d} N={N:5d} n_kernel_repeat={n_kernel_repeat} | "
                  f"时间: {median_ms:7.3f}ms | "
                  f"性能: {gops_med:6.2f} GOPS@{compute_dtype} ({util_med:5.1f}%)")
            
            # 保存结果
            all_results.append({
                'kernel': name,
                'compute_dtype': compute_dtype,
                'dtype_width': dtype_width,
                'M': M, 'K': K, 'N': N,
                'median_ms': median_ms, 'min_ms': min_ms, 'max_ms': max_ms,
                'gops_med': gops_med, 'gops_min': gops_min, 'gops_max': gops_max,
                'util_med': util_med, 'util_min': util_min, 'util_max': util_max,
                'peak_gops': peak_gops,
            })
        
        print(f"\n[完成] {name}: 平均利用率 {sum(r['util_med'] for r in all_results if r['kernel']==name)/len([r for r in all_results if r['kernel']==name]):.1f}%")
    
    return all_results

def write_csv(results, path: str):
    fieldnames = [
        'kernel','compute_dtype','dtype_width','M','K','N','median_ms','min_ms','max_ms',
        'gops_med','gops_min','gops_max','util_med','util_min','util_max','peak_gops'
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"CSV 写入完成: {path}")

def plot_results(results, metric: str, path: str):
    # metric: 'gflops' 或 'util'
    import importlib
    if importlib.util.find_spec("matplotlib.pyplot") is None:
        print("警告: 未找到 matplotlib, 跳过绘图。")
        return
    import matplotlib.pyplot as plt  # type: ignore
    import matplotlib.cm as cm
    import numpy as np
    
    metric_key = 'gops_med' if metric == 'gflops' else 'util_med'
    
    # 提取所有唯一的 (K, N) 组合、M 值和 kernel
    # 按照权重数据大小 N*K 从小到大排序
    k_n_pairs = sorted(set((r['K'], r['N']) for r in results), key=lambda x: x[0] * x[1])
    m_values = sorted(set(r['M'] for r in results))
    
    # 为每个 M 值分配颜色（使用色谱）
    colors = cm.get_cmap('tab20' if len(m_values) <= 20 else 'hsv')(np.linspace(0, 1, len(m_values)))
    m_color_map = {m: colors[i] for i, m in enumerate(m_values)}
    
    # 为每个 kernel 分配标记形状
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']
    kernel_marker_map = {spec.name: markers[i % len(markers)] for i, spec in enumerate(KERNEL_SPECS)}
    
    # 创建 x 轴位置和标签
    x_labels = [f"{K}×{N}" for (K, N) in k_n_pairs]
    x_positions = np.arange(len(k_n_pairs))
    
    # 动态调整图像大小
    fig_width = max(12, len(k_n_pairs) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_width, 7))
    
    # 用于图例的句柄
    legend_elements_kernel = []
    legend_elements_m = []
    
    # 绘制数据点
    for r in results:
        k_n = (r['K'], r['N'])
        if k_n not in k_n_pairs:
            continue
        x_idx = k_n_pairs.index(k_n)
        x_pos = x_positions[x_idx]
        
        # 为每个 kernel 添加小的随机偏移，避免重叠
        kernel_idx = [spec.name for spec in KERNEL_SPECS].index(r['kernel'])
        offset = (kernel_idx - len(KERNEL_SPECS)/2) * 0.15
        
        marker = kernel_marker_map[r['kernel']]
        color = m_color_map[r['M']]
        
        ax.scatter(x_pos + offset, r[metric_key], 
                  marker=marker, c=[color], s=80, alpha=0.7, 
                  edgecolors='black', linewidths=0.5)
    
    # 创建图例
    # Kernel 图例（不同形状）
    for spec in KERNEL_SPECS:
        name = spec.name
        legend_elements_kernel.append(
            plt.Line2D([0], [0], marker=kernel_marker_map[name], color='gray', 
                      label=name, markersize=8, linestyle='None', 
                      markeredgecolor='black', markeredgewidth=0.5)
        )
    
    # M 值图例（不同颜色）- 只显示部分代表性的 M 值
    m_display = m_values if len(m_values) <= 10 else m_values[::max(1, len(m_values)//10)]
    for m in m_display:
        legend_elements_m.append(
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=m_color_map[m], label=f'M={m}',
                      markersize=8, markeredgecolor='black', markeredgewidth=0.5)
        )
    
    # 绘制每个 kernel 的理论峰值线
    if metric == 'gflops':
        # 从结果中提取每个 kernel 的峰值
        kernel_peaks = {}
        kernel_dtypes = {}
        for r in results:
            if r['kernel'] not in kernel_peaks:
                kernel_peaks[r['kernel']] = r.get('peak_gops', 0)
                kernel_dtypes[r['kernel']] = r.get('compute_dtype', 'int8')
        
        # 为峰值线分配样式
        peak_line_styles = ['-', '--', '-.', ':']
        for idx, (kernel_name, peak_gops) in enumerate(kernel_peaks.items()):
            if peak_gops > 0:
                line_style = peak_line_styles[idx % len(peak_line_styles)]
                dtype_name = kernel_dtypes.get(kernel_name, 'int8')
                ax.axhline(y=peak_gops, color='red', linestyle=line_style, 
                          linewidth=2, alpha=0.6, 
                          label=f'{kernel_name} peak: {peak_gops:.1f} GOPS@{dtype_name}')
    elif metric == 'util':
        # Utilization模式下绘制 100% 参考线
        ax.axhline(y=100, color='red', linestyle='--', 
                  linewidth=2, alpha=0.6, label='100% Utilization')
    
    # 设置坐标轴
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
    
    # Y轴标签根据 metric 显示不同单位
    if metric == 'gflops':
        # 获取第一个结果的 compute_dtype 作为参考
        dtype_label = results[0].get('compute_dtype', 'int8') if results else 'int8'
        ax.set_ylabel(f'GOPS@{dtype_label}', fontsize=11)
    else:
        ax.set_ylabel('Utilization (%)', fontsize=11)

    ax.set_xlabel('Weight Matrix Shapes (K × N)', fontsize=11)
    ax.set_title(f'Kernel Performance by Weight Shape\n'
                 f'({len(KERNEL_SPECS)} kernels × {len(m_values)} M values × {len(k_n_pairs)} K×N shapes)', 
                 fontsize=12)
    ax.grid(alpha=0.3, axis='y', linewidth=0.5)
    
    # 添加三个图例：Kernel形状 + M值颜色 + 峰值线
    legend1 = ax.legend(handles=legend_elements_kernel, title='Kernel', 
                       loc='upper left', fontsize=9, framealpha=0.9)
    ax.add_artist(legend1)  # 保持第一个图例

    legend2 = ax.legend(handles=legend_elements_m, title='Input Length M', 
                       loc='upper right', fontsize=8, framealpha=0.9, ncol=max(1, len(m_display)//8))
    ax.add_artist(legend2)  # 保持第二个图例
    
    # 如果有峰值线，添加第三个图例
    if metric == 'gflops' and any(r.get('peak_gops', 0) > 0 for r in results):
        # 收集峰值线的图例句柄
        handles, labels = ax.get_legend_handles_labels()
        peak_handles = [(h, l) for h, l in zip(handles, labels) if 'peak' in l]
        if peak_handles:
            ax.legend([h for h, l in peak_handles], [l for h, l in peak_handles],
                     title='Peak Performance', loc='lower left', fontsize=8, framealpha=0.9)
    elif metric == 'util':
        # Utilization模式显示 100% 线的图例
        handles, labels = ax.get_legend_handles_labels()
        util_handles = [(h, l) for h, l in zip(handles, labels) if 'Utilization' in l]
        if util_handles:
            ax.legend([h for h, l in util_handles], [l for h, l in util_handles],
                     title='Reference Line', loc='lower left', fontsize=8, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"图像已保存: {path} (K×N组合: {len(k_n_pairs)}, M值: {len(m_values)}, 总数据点: {len(results)})")

def main():
    parser = argparse.ArgumentParser(description='统一 GEMM Kernel 性能测试驱动')
    parser.add_argument('--shapes', type=str, default='', help='逗号分隔形状列表，例如 48x512x32,96x512x32')
    parser.add_argument('--kernels', type=str, default='', help='逗号分隔的 kernel 名称列表，例如 q40_q80,q4k_q8k。留空表示测试所有 kernel')
    parser.add_argument('--warmup', type=int, default=10, help='预热时间 ms')
    parser.add_argument('--rounds', type=int, default=100, help='测试时间 ms')
    parser.add_argument('--num-threads', type=int, default=8, help='CPU核心/线程数，用于多核理论峰值计算和 torch 线程设置')
    parser.add_argument('--n-kernel-repeat', type=int, default=1, help='单次 kernel 调用中的重复执行次数，用于减少启动开销影响 (默认 1)')
    parser.add_argument('--freq', type=float, default=1.6, help='芯片频率 GHz (默认 1.6)')
    parser.add_argument('--vlen', type=int, default=256, help='向量宽度 bits (默认 256)')
    parser.add_argument('--csv-out', type=str, default='kernel_bench_results.csv', help='CSV 输出路径')
    parser.add_argument('--png-out', type=str, default='kernel_bench_results.png', help='性能图输出路径')
    parser.add_argument('--metric', type=str, default='gflops', choices=['gflops','util'], help='绘图纵轴')
    args = parser.parse_args()

    if args.num_threads is not None:
        try:
            torch.set_num_threads(args.num_threads)
        except Exception as e:
            print(f"警告: 无法设置线程数: {e}")

    # 如果命令行指定了形状，使用指定的形状；否则使用默认形状
    if args.shapes:
        shapes = parse_shapes_arg(args.shapes)
        print(f"\n使用命令行指定的形状 (共 {len(shapes)} 个)")
    else:
        shapes = list(DEFAULT_SHAPES)
        print(f"\n使用默认形状 (共 {len(shapes)} 个)")
    
    # 解析 kernel 选择
    selected_kernels = None
    if args.kernels:
        selected_kernels = [k.strip() for k in args.kernels.split(',') if k.strip()]
        available_kernels = [spec.name for spec in KERNEL_SPECS]
        print(f"可用的 kernel: {', '.join(available_kernels)}")
        print(f"选择的 kernel: {', '.join(selected_kernels)}")

    print(f"\n{'='*80}")
    print(f"性能测试配置")
    print(f"{'='*80}")
    print(f"芯片参数: Freq={args.freq} GHz, VLEN={args.vlen} bits, Cores={args.num_threads}")
    print(f"测试参数: warmup={args.warmup}, rounds={args.rounds}, threads={args.num_threads}, n_kernel_repeat={args.n_kernel_repeat}")
    print(f"总形状数: {len(shapes)}")
    print(f"注册 Kernel 数: {len(KERNEL_SPECS)}")
    if selected_kernels:
        print(f"测试 Kernel 数: {len(selected_kernels)}")
    print(f"{'='*80}\n")
    
    results = run_bench(shapes, warmup=args.warmup, rounds=args.rounds, 
                        num_threads=args.num_threads, freq=args.freq, 
                        vlen=args.vlen, selected_kernels=selected_kernels, 
                        n_kernel_repeat=args.n_kernel_repeat)
    if not results:
        print("\n无结果产生, 退出")
        return
    
    print(f"\n{'='*80}")
    print(f"测试完成")
    print(f"{'='*80}")
    print(f"总测试数: {len(results)}")
    print(f"平均性能: {sum(r['gops_med'] for r in results)/len(results):.2f} GOPS")
    print(f"平均利用率: {sum(r['util_med'] for r in results)/len(results):.1f}%")
    print(f"{'='*80}\n")
    
    write_csv(results, args.csv_out)
    plot_results(results, args.metric, args.png_out)

if __name__ == '__main__':
    main()
