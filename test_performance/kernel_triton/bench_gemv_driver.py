'''
# 测试所有 kernel 在所有预定义形状上的性能
TRITON_ALWAYS_COMPILE=1  TRITON_CPU_BACKEND=1  python test_performance/kernel_triton/gemv_driver.py --sweep --kernel all --rounds 5

# 只测试特定 kernel
TRITON_ALWAYS_COMPILE=1  TRITON_CPU_BACKEND=1  python test_performance/kernel_triton/gemv_driver.py --sweep --kernel q40_q80 --rounds 3

# 测试多个 kernel（逗号分隔会被解释为单一字符串，需通过修改代码支持或多次运行）
TRITON_ALWAYS_COMPILE=1  TRITON_CPU_BACKEND=1  python test_performance/kernel_triton/gemv_driver.py --sweep --kernel q40_q80 --rounds 3
TRITON_ALWAYS_COMPILE=1  TRITON_CPU_BACKEND=1  python test_performance/kernel_triton/gemv_driver.py --sweep --kernel q4k_q8k --rounds 3

# 指定机器的理论峰值带宽（用于计算利用率）
TRITON_ALWAYS_COMPILE=1  TRITON_CPU_BACKEND=1  python test_performance/kernel_triton/gemv_driver.py --sweep --kernel all --peak-bw 10.0

TRITON_ALWAYS_COMPILE=1  TRITON_CPU_BACKEND=1  python test_performance/kernel_triton/gemv_driver.py --sweep --kernel all --csv my_results.csv

# 自动生成时间戳命名的图片
TRITON_ALWAYS_COMPILE=1  TRITON_CPU_BACKEND=1  python test_performance/kernel_triton/gemv_driver.py --sweep --kernel all --plot

# 指定图片文件名
TRITON_ALWAYS_COMPILE=1  TRITON_CPU_BACKEND=1  python test_performance/kernel_triton/gemv_driver.py --sweep --kernel all --plot bandwidth_comparison.png

# 同时指定 CSV 和图片，以及峰值带宽 (推荐)
TRITON_ALWAYS_COMPILE=1  TRITON_CPU_BACKEND=1  python test_performance/kernel_triton/gemv_driver.py --sweep --kernel all --num_threads=8 \
    --csv bandwidth_comparison.csv --plot bandwidth_comparison.png --rounds 5 --target-gb 1.5 --peak-bw 8.5
'''



import os
import sys
import time
import csv
from datetime import datetime
import numpy as np
import torch
import triton
import triton.testing as tt
import matplotlib
matplotlib.use('Agg')  # 无需显示，直接保存
import matplotlib.pyplot as plt

# 确保可以被其它同目录脚本导入
CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from q40_q80_gemv import q40_q80_gemv_kernel  # type: ignore
from q4k_q8k_gemv import q4k_q8k_gemv_kernel  # type: ignore
from iq4k_q8k_gemv import iq4k_q8k_gemv_kernel  # type: ignore

try:
    from dataclasses import dataclass
except ImportError:
    # 极端情况下 Python 版本过低
    raise RuntimeError("需要 Python 3.7+ 支持 dataclasses")

@dataclass
class KernelSpec:
    name: str
    fn: object  # triton kernel
    constraint: object  # lambda N,K -> None / raise
    gen_dataset: object  # lambda N,K,seed -> dict(str,tensor)
    estimate_bytes: object  # lambda N,K -> int (单批次总字节)
    grid_fn: object  # lambda N,K -> tuple
    param_builder: object  # lambda data,N,K,threads -> (kernel_args_dict)


DEFAULT_PEAK_BW = 5.83  # 默认理论峰值带宽 (GB/s)


REGISTRY = {}

def register_kernel(spec: KernelSpec):
    REGISTRY[spec.name] = spec


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
# m_values = [480, 192, 144, 96, 72, 48, 24, 12]

# 根据 k_n_pairs 和 m_values 构造默认测试形状 (M, N, K)
# 对于每个 (K, N) 对，使用所有 M 值构造 (M, N, K) 测试形状
SHAPE_CONFIGS: list[tuple[int, int, int]] = [
]

for k, n in k_n_pairs:
    SHAPE_CONFIGS.append((n, k))


def register_shape(N, K):
    """允许用户动态添加测试形状"""
    if (N, K) not in SHAPE_CONFIGS:
        SHAPE_CONFIGS.append((N, K))

def get_registered_shapes():
    """获取所有注册的测试形状"""
    return SHAPE_CONFIGS.copy()


# ------------------------------ q40_q80 ----------------------------------

def _q40_constraint(N, K):
    assert N % 32 == 0, "q40_q80: N 必须是32的倍数"
    assert K % 32 == 0, "q40_q80: K 必须是32的倍数"


def _q40_gen_dataset(N, K, seed, batch=1):
    torch.manual_seed(seed)
    q8_0_vector = torch.randint(-128, 127, (batch, K), dtype=torch.int8)
    q8_0_scale = torch.randint(1, 255, (batch, K // 32), dtype=torch.uint16)
    q4_0_matrix = torch.randint(0, 255, (batch, N * (K // 2)), dtype=torch.uint8)
    q4_0_scale = torch.randint(1, 255, (batch, N * (K // 32)), dtype=torch.uint16)
    output = torch.zeros((batch, N), dtype=torch.float32)
    return {
        'q8_0_vector': q8_0_vector.contiguous(),
        'q8_0_scale': q8_0_scale.contiguous(),
        'q4_0_matrix': q4_0_matrix.contiguous(),
        'q4_0_scale': q4_0_scale.contiguous(),
        'output': output.contiguous(),
    }


def _q40_estimate_bytes(N, K):
    weight_bytes = N * (K // 2) + N * (K // 32) * 2  # q4 matrix + scale
    activation_bytes = K + (K // 32) * 2  # q8 vector + scale
    output_bytes = N * 4
    return weight_bytes + activation_bytes + output_bytes


def _q40_grid(N, K, batch=1):
    return (batch, N // 32)


def _q40_params(data, N, K, threads, batch=1):
    return dict(
        q8_0_vector_ptr=data['q8_0_vector'],
        q8_0_scale_ptr=data['q8_0_scale'],
        q4_0_matrix_ptr=data['q4_0_matrix'],
        q4_0_scale_ptr=data['q4_0_scale'],
        output_ptr=data['output'],
        K=K,
        N=N,
        batch=batch,
        num_threads=threads,
    )

# ------------------------------ q4k_q8k ----------------------------------

def _q4k_constraint(N, K):
    assert N % 32 == 0, "q4k_q8k: N 必须是32的倍数"
    assert K % 256 == 0, "q4k_q8k: K 必须是256的倍数"


def _q4k_gen_dataset(N, K, seed, batch=1):
    torch.manual_seed(seed)
    NR = 32
    QK_K = 256
    QK_K_SUB_BLOCK_SIZE = 32

    NUM_SUB_BLOCKS = 8
    QK_4_K_SUB_BLOCK_DATA_SIZE = 16
    NUM_SUB_BLOCKS = QK_K // QK_K_SUB_BLOCK_SIZE 

    NR_SUB_BLOCKS_NUMS = N // NR
    QK_SUPER_BLOCK_NUMS = K // QK_K
    
    # 添加形状验证
    assert NR_SUB_BLOCKS_NUMS > 0, f"N={N} 太小，必须至少为 {NR}"
    assert QK_SUPER_BLOCK_NUMS > 0, f"K={K} 太小，必须至少为 {QK_K}" 

    # q8k 数据准备 (输入激活向量) - 按照 kernel 中的 make_block_ptr shape 定义，增加 batch 维度
    q8k_vector = torch.randint(-128, 127, (batch, QK_SUPER_BLOCK_NUMS, NUM_SUB_BLOCKS, QK_K_SUB_BLOCK_SIZE, 1), dtype=torch.int8)
    q8k_d = torch.randn((batch, QK_SUPER_BLOCK_NUMS), dtype=torch.float32)
    q8_bsums = torch.randint(-32768, 32767, (batch, QK_SUPER_BLOCK_NUMS, NUM_SUB_BLOCKS, 2, 1), dtype=torch.int16)

    # q4k 数据准备 (权重矩阵) - 按照 kernel 中的 make_block_ptr shape 定义，增加 batch 维度
    q4k_matrix = torch.randint(0, 255, (batch, NR_SUB_BLOCKS_NUMS, QK_SUPER_BLOCK_NUMS, NUM_SUB_BLOCKS, QK_4_K_SUB_BLOCK_DATA_SIZE, NR), dtype=torch.uint8)
    q4k_scale_l = torch.randint(0, 255, (batch, NR_SUB_BLOCKS_NUMS, QK_SUPER_BLOCK_NUMS, NUM_SUB_BLOCKS // 2, NR), dtype=torch.uint8)
    q4k_scale_h = torch.randint(0, 255, (batch, NR_SUB_BLOCKS_NUMS, QK_SUPER_BLOCK_NUMS, NUM_SUB_BLOCKS // 4, NR), dtype=torch.uint8)
    q4k_mins_l = torch.randint(0, 255, (batch, NR_SUB_BLOCKS_NUMS, QK_SUPER_BLOCK_NUMS, NUM_SUB_BLOCKS // 2, NR), dtype=torch.uint8)
    q4k_mins_h = torch.randint(0, 255, (batch, NR_SUB_BLOCKS_NUMS, QK_SUPER_BLOCK_NUMS, NUM_SUB_BLOCKS // 4, NR), dtype=torch.uint8)
    q4k_d = torch.randint(0, 65535, (batch, NR_SUB_BLOCKS_NUMS, QK_SUPER_BLOCK_NUMS, NR), dtype=torch.uint16)
    q4k_dmin = torch.randint(0, 65535, (batch, NR_SUB_BLOCKS_NUMS, QK_SUPER_BLOCK_NUMS, NR), dtype=torch.uint16)
    
    output = torch.zeros((batch, NR_SUB_BLOCKS_NUMS, NR), dtype=torch.float32)
    
    # 确保所有张量内存连续
    tensors_dict = {
        'q8k_vector': q8k_vector.contiguous(),
        'q8k_d': q8k_d.contiguous(),
        'q8_bsums': q8_bsums.contiguous(),
        'q4k_matrix': q4k_matrix.contiguous(),
        'q4k_scale_l': q4k_scale_l.contiguous(),
        'q4k_scale_h': q4k_scale_h.contiguous(),
        'q4k_mins_l': q4k_mins_l.contiguous(),
        'q4k_mins_h': q4k_mins_h.contiguous(),
        'q4k_d': q4k_d.contiguous(),
        'q4k_dmin': q4k_dmin.contiguous(),
        'output': output.contiguous(),
    }

    return tensors_dict


def _q4k_estimate_bytes(N, K):
    NR = 32
    QK_K = 256
    NUM_SUB_BLOCKS = 8
    QK_4_K_SUB_BLOCK_DATA_SIZE = 16
    # 逐个估算字节数
    size = 0
    size += K  # q8k_vector int8
    size += (K // QK_K) * 4  # q8k_d float32
    size += (K // QK_K) * NUM_SUB_BLOCKS * 2 * 2  # q8_bsums int16 *2
    size += (N // NR) * (K // QK_K) * NUM_SUB_BLOCKS * QK_4_K_SUB_BLOCK_DATA_SIZE * NR  # q4k_matrix uint8
    size += (N // NR) * (K // QK_K) * (NUM_SUB_BLOCKS // 2) * NR  # scale_l uint8
    size += (N // NR) * (K // QK_K) * (NUM_SUB_BLOCKS // 4) * NR  # scale_h uint8
    size += (N // NR) * (K // QK_K) * (NUM_SUB_BLOCKS // 2) * NR  # mins_l
    size += (N // NR) * (K // QK_K) * (NUM_SUB_BLOCKS // 4) * NR  # mins_h
    size += (N // NR) * (K // QK_K) * NR * 2  # q4k_d uint16
    size += (N // NR) * (K // QK_K) * NR * 2  # q4k_dmin uint16
    size += N * 4  # output float32
    return size


def _q4k_grid(N, K, batch=1):
    return (batch, N // 32)


def _q4k_params(data, N, K, threads, batch=1):
    return dict(
        q8k_vector_ptr=data['q8k_vector'],
        q8k_d_ptr=data['q8k_d'],
        q8_bsums_ptr=data['q8_bsums'],
        q4k_matrix_ptr=data['q4k_matrix'],
        q4k_scale_l_ptr=data['q4k_scale_l'],
        q4k_scale_h_ptr=data['q4k_scale_h'],
        q4k_mins_l_ptr=data['q4k_mins_l'],
        q4k_mins_h_ptr=data['q4k_mins_h'],
        q4k_d_ptr=data['q4k_d'],
        q4k_dmin_ptr=data['q4k_dmin'],
        output_ptr=data['output'],
        K=K,
        N=N,
        batch=batch,
        num_threads=threads,
    )

# ------------------------------ iq4k_q8k ----------------------------------

def _iq4k_constraint(N, K):
    assert N % 32 == 0, "iq4k_q8k: N 必须是32的倍数"
    assert K % 256 == 0, "iq4k_q8k: K 必须是256的倍数"


def _iq4k_gen_dataset(N, K, seed, batch=1):
    torch.manual_seed(seed)
    NR = 32
    QK_K = 256
    NUM_SUB_BLOCKS = 8
    QK_K_SUB_BLOCK_SIZE = 32
    QK_4_K_SUB_BLOCK_DATA_SIZE = 16

    # 修复: q8k_vector 应该是 4D 以匹配 kernel 中的 block_ptr，增加 batch 维度
    q8k_vector = torch.randint(-128, 127, (batch, K // QK_K, QK_K // QK_K_SUB_BLOCK_SIZE, QK_K_SUB_BLOCK_SIZE, 1), dtype=torch.int8)
    q8k_d = torch.randn((batch, K // QK_K), dtype=torch.float32)
    
    iq4k_extra = torch.randint(0, 65535, (batch, N // NR, K // QK_K, NR), dtype=torch.uint16)
    iq4k_matrix = torch.randint(0, 255, (batch, N // NR, K // QK_K, NUM_SUB_BLOCKS, QK_4_K_SUB_BLOCK_DATA_SIZE, NR), dtype=torch.uint8)
    iq4k_d = torch.randint(0, 65535, (batch, N // NR, K // QK_K, NR), dtype=torch.uint16)
    iq4k_scale_l = torch.randint(0, 255, (batch, N // NR, K // QK_K, NUM_SUB_BLOCKS, NR), dtype=torch.uint8)
    iq4k_scale_h = torch.randint(0, 255, (batch, N // NR, K // QK_K, NUM_SUB_BLOCKS // 2, NR), dtype=torch.uint8)
    # 修复: output 应该是 2D 以匹配 kernel 中的 block_ptr，增加 batch 维度
    output = torch.zeros((batch, N // NR, NR), dtype=torch.float32)

    return {
        'q8k_vector': q8k_vector.contiguous(),
        'q8k_d': q8k_d.contiguous(),
        'iq4k_extra': iq4k_extra.contiguous(),
        'iq4k_matrix': iq4k_matrix.contiguous(),
        'iq4k_d': iq4k_d.contiguous(),
        'iq4k_scale_l': iq4k_scale_l.contiguous(),
        'iq4k_scale_h': iq4k_scale_h.contiguous(),
        'output': output.contiguous(),
    }


def _iq4k_estimate_bytes(N, K):
    NR = 32
    QK_K = 256
    NUM_SUB_BLOCKS = 8
    QK_4_K_SUB_BLOCK_DATA_SIZE = 16
    size = 0
    size += K  # q8k_vector int8
    size += (K // QK_K) * 4  # q8k_d float32
    size += (N // NR) * (K // QK_K) * NR * 2  # iq4k_extra uint16
    size += (N // NR) * (K // QK_K) * NUM_SUB_BLOCKS * QK_4_K_SUB_BLOCK_DATA_SIZE * NR  # iq4k_matrix uint8
    size += (N // NR) * (K // QK_K) * NR * 2  # iq4k_d uint16
    size += (N // NR) * (K // QK_K) * NUM_SUB_BLOCKS * NR  # scale_l uint8
    size += (N // NR) * (K // QK_K) * (NUM_SUB_BLOCKS // 2) * NR  # scale_h uint8
    size += N * 4  # output float32
    return size


def _iq4k_grid(N, K, batch=1):
    return (batch, N // 32)


def _iq4k_params(data, N, K, threads, batch=1):
    return dict(
        q8k_vector_ptr=data['q8k_vector'],
        q8k_d_ptr=data['q8k_d'],
        iq4k_extra_ptr=data['iq4k_extra'],
        iq4k_matrix_ptr=data['iq4k_matrix'],
        iq4k_d_ptr=data['iq4k_d'],
        iq4k_scale_l_ptr=data['iq4k_scale_l'],
        iq4k_scale_h_ptr=data['iq4k_scale_h'],
        output_ptr=data['output'],
        K=K,
        N=N,
        batch=batch,
        num_threads=threads,
    )

# 注册三个 kernel
register_kernel(KernelSpec(
    name='q40_q80',
    fn=q40_q80_gemv_kernel,
    constraint=_q40_constraint,
    gen_dataset=_q40_gen_dataset,
    estimate_bytes=_q40_estimate_bytes,
    grid_fn=_q40_grid,
    param_builder=_q40_params,
))

register_kernel(KernelSpec(
    name='q4k_q8k',
    fn=q4k_q8k_gemv_kernel,
    constraint=_q4k_constraint,
    gen_dataset=_q4k_gen_dataset,
    estimate_bytes=_q4k_estimate_bytes,
    grid_fn=_q4k_grid,
    param_builder=_q4k_params,
))

register_kernel(KernelSpec(
    name='iq4k_q8k',
    fn=iq4k_q8k_gemv_kernel,
    constraint=_iq4k_constraint,
    gen_dataset=_iq4k_gen_dataset,
    estimate_bytes=_iq4k_estimate_bytes,
    grid_fn=_iq4k_grid,
    param_builder=_iq4k_params,
))

# ------------------------------ 通用带宽测试 ------------------------------

def calculate_num_batches(spec: KernelSpec, N, K, target_gb):
    bytes_per_batch = spec.estimate_bytes(N, K)
    target_bytes = target_gb * (1024 ** 3)
    batches = int(target_bytes / bytes_per_batch)
    return max(batches, 1)



def run_bandwidth(kernel_name: str, N: int, K: int, rounds: int = 5, warmup: int = 3, target_gb: float = 2.0, threads: int = 4, peak_bw: float = DEFAULT_PEAK_BW, verbose: bool = True, batch: int = None):
    assert kernel_name in REGISTRY, f"未注册 kernel: {kernel_name}"
    spec = REGISTRY[kernel_name]
    spec.constraint(N, K)

    # 计算 batch 数量
    if batch is None:
        num_batches = calculate_num_batches(spec, N, K, target_gb)
    else:
        num_batches = batch
    if verbose:
        print(f"Kernel: {kernel_name}")
        print(f"维度: [batch x {K}] x [{K} x {N}]  (N={N}, K={K}, batch={num_batches})")
        print(f"数据批次数: {num_batches} (目标 ~{target_gb} GB)")

    # 生成带 batch 维度的数据
    data = spec.gen_dataset(N, K, seed=12345, batch=num_batches)
    total_bytes_all = spec.estimate_bytes(N, K) * num_batches
    total_mb = total_bytes_all / (1024 ** 2)
    total_gb = total_mb / 1024
    if verbose:
        print(f"总数据量估算: {total_mb:.2f} MB ({total_gb:.2f} GB)\n")

    # grid 增加 batch 维度
    grid = spec.grid_fn(N, K, batch=num_batches)

    def run_kernel():
        try:
            params = spec.param_builder(data, N, K, threads, batch=num_batches)
            spec.fn[grid](**params)
        except Exception as e:
            print(f"\n[ERROR] 执行失败:")
            print(f"  形状: N={N}, K={K}, batch={num_batches}")
            print(f"  异常: {e}")
            import traceback
            traceback.print_exc()
            raise

    if verbose:
        print("开始性能测试 (使用 triton.testing.do_bench)...")

    median_ms, min_ms, max_ms = tt.do_bench(
        run_kernel,
        warmup=warmup,
        rep=rounds,
        quantiles=[0.5, 0.2, 0.8]
    )

    bw_median = total_gb / (median_ms / 1000.0)
    bw_min = total_gb / (max_ms / 1000.0)
    bw_max = total_gb / (min_ms / 1000.0)
    std_bw = (bw_max - bw_min) / 4.0
    util = bw_median / peak_bw * 100

    if verbose:
        print("="*60)
        print(f"{kernel_name} 测试总结")
        print("="*60)
        print(f"中位数带宽: {bw_median:.2f} GB/s (±{std_bw:.2f})")
        print(f"带宽范围: [{bw_min:.2f}, {bw_max:.2f}] GB/s")
        print(f"中位数延迟: {median_ms:.3f} ms")
        print(f"延迟范围: [{min_ms:.3f}, {max_ms:.3f}] ms")
        print(f"理论峰值带宽: {peak_bw:.2f} GB/s | 利用率: {util:.2f}%")
        print(f"batch: {num_batches}")
        print("="*60)

    return {
        'kernel_name': kernel_name,
        'N': N,
        'K': K,
        'avg_bandwidth': bw_median,
        'std_bandwidth': std_bw,
        'min_bandwidth': bw_min,
        'max_bandwidth': bw_max,
        'avg_latency': median_ms,
        'bandwidth_util': util,
        'bandwidths': [bw_median],
        'latencies': [median_ms],
        'rounds': rounds,
        'target_gb': target_gb,
        'num_batches': num_batches,
        'peak_bw': peak_bw,
        'batch': num_batches,
    }


def run_all(N, K, rounds=5, warmup=3, target_gb=2.0, threads=4, peak_bw=DEFAULT_PEAK_BW):
    results = {}
    for name in REGISTRY.keys():
        print("\n\n" + "#"*70)
        print(f"运行 {name} 带宽测试")
        print("#"*70 + "\n")
        results[name] = run_bandwidth(name, N, K, rounds, warmup, target_gb, threads, peak_bw)
    return results


def run_sweep(kernel_names, shapes=None, rounds=5, warmup=3, target_gb=2.0, threads=4, peak_bw=DEFAULT_PEAK_BW, csv_path=None):
    """对指定 kernel 在多个 shape 上进行测试
    
    Args:
        kernel_names: 要测试的 kernel 列表，如 ['q40_q80', 'q4k_q8k'] 或 'all'
        shapes: 测试形状列表 [(N1,K1), (N2,K2), ...]，默认使用 SHAPE_CONFIGS
        peak_bw: 理论峰值带宽 (GB/s)
        csv_path: CSV 输出路径，若为 None 则自动生成
    
    Returns:
        results: {kernel_name: [result_dict1, result_dict2, ...]}
    """
    if shapes is None:
        shapes = SHAPE_CONFIGS
    
    if kernel_names == 'all':
        kernel_names = list(REGISTRY.keys())
    elif isinstance(kernel_names, str):
        kernel_names = [kernel_names]
    
    results = {name: [] for name in kernel_names}
    
    total_tests = len(kernel_names) * len(shapes)
    current_test = 0
    
    for name in kernel_names:
        print(f"\n[INFO] 开始测试 kernel: {name}")
        
        for N, K in shapes:
            current_test += 1
            print(f"  [{current_test}/{total_tests}] 测试 {name} @ N={N}, K={K} ...", end=' ', flush=True)
            
            # 添加形状验证
            try:
                REGISTRY[name].constraint(N, K)
            except AssertionError as e:
                print(f"⊗ 跳过 (形状不满足约束: {e})")
                continue
                
            try:
                # 强制垃圾回收，减少内存碎片
                import gc
                gc.collect()
                
                result = run_bandwidth(name, N, K, rounds, warmup, target_gb, threads, peak_bw, verbose=False)
                results[name].append(result)
                print(f"✓ {result['avg_bandwidth']:.2f} GB/s")
            except Exception as e:
                import traceback
                error_msg = str(e)
                print(f"✗ 失败: {error_msg}")
                print(f"     详细错误信息:")
                traceback.print_exc()
                
                # 记录失败但继续
                results[name].append({
                    'kernel_name': name,
                    'N': N,
                    'K': K,
                    'avg_bandwidth': 0.0,
                    'error': error_msg
                })
    
    # 保存 CSV
    if csv_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = f'gemv_bandwidth_sweep_{timestamp}.csv'
    
    save_results_csv(results, csv_path)
    print(f"\n[INFO] 结果已保存到: {csv_path}")
    
    # 打印汇总表格
    print_summary_table(results)
    
    return results


def print_summary_table(results):
    """打印所有测试结果的汇总表格
    
    Args:
        results: {kernel_name: [result_dict1, result_dict2, ...]}
    """
    print("\n" + "="*120)
    print(" " * 45 + "测试结果汇总")
    print("="*120)
    
    # 表头
    header = f"{'Kernel':<12} {'Shape (NxK)':<15} {'Avg BW':<10} {'Std BW':<10} {'Min BW':<10} {'Max BW':<10} {'Latency':<12} {'Util %':<8}"
    print(header)
    print("-"*120)
    
    # 按 kernel 分组打印
    for kernel_name in sorted(results.keys()):
        result_list = results[kernel_name]
        valid_results = [r for r in result_list if 'error' not in r]
        
        if not valid_results:
            print(f"{kernel_name:<12} {'(无有效结果)'}")
            continue
        
        # 打印该 kernel 的所有结果
        for idx, res in enumerate(valid_results):
            shape_str = f"{res['N']}x{res['K']}"
            kernel_col = kernel_name if idx == 0 else ""
            
            line = f"{kernel_col:<12} {shape_str:<15} "
            line += f"{res['avg_bandwidth']:>8.2f}  "
            line += f"{res['std_bandwidth']:>8.2f}  "
            line += f"{res['min_bandwidth']:>8.2f}  "
            line += f"{res['max_bandwidth']:>8.2f}  "
            line += f"{res['avg_latency']:>10.2f}  "
            line += f"{res['bandwidth_util']:>6.1f}"
            print(line)
        
        # kernel 之间的分隔线
        if kernel_name != sorted(results.keys())[-1]:
            print("-"*120)
    
    print("="*120)
    
    # 统计信息
    total_tests = sum(len([r for r in rlist if 'error' not in r]) for rlist in results.values())
    total_failed = sum(len([r for r in rlist if 'error' in r]) for rlist in results.values())
    
    print(f"\n总测试数: {total_tests + total_failed} | 成功: {total_tests} | 失败: {total_failed}")
    
    # 每个 kernel 的平均性能
    print("\n各 Kernel 平均带宽:")
    for kernel_name in sorted(results.keys()):
        valid_results = [r for r in results[kernel_name] if 'error' not in r]
        if valid_results:
            avg_bw = np.mean([r['avg_bandwidth'] for r in valid_results])
            print(f"  {kernel_name:<12}: {avg_bw:>6.2f} GB/s")
    print()


def save_results_csv(results, csv_path):
    """保存测试结果到 CSV
    
    Args:
        results: {kernel_name: [result_dict1, result_dict2, ...]}
        csv_path: CSV 文件路径
    """
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # 表头
        writer.writerow([
            'kernel_name', 'N', 'K', 'shape_str',
            'avg_bandwidth_GBps', 'std_bandwidth_GBps',
            'min_bandwidth_GBps', 'max_bandwidth_GBps',
            'avg_latency_ms', 'bandwidth_util_percent',
            'rounds', 'target_gb', 'num_batches'
        ])
        
        for kernel_name, result_list in results.items():
            for res in result_list:
                if 'error' in res:
                    continue
                writer.writerow([
                    res['kernel_name'],
                    res['N'],
                    res['K'],
                    f"{res['N']}x{res['K']}",
                    res['avg_bandwidth'],
                    res['std_bandwidth'],
                    res['min_bandwidth'],
                    res['max_bandwidth'],
                    res['avg_latency'],
                    res['bandwidth_util'],
                    res['rounds'],
                    res['target_gb'],
                    res['num_batches'],
                ])


def plot_bandwidth_comparison(results, output_path=None, title='GEMV Bandwidth Comparison'):
    """绘制不同 kernel 在不同 shape 下的带宽对比图
    
    Args:
        results: {kernel_name: [result_dict1, result_dict2, ...]}
        output_path: 图片保存路径，默认自动生成
        title: 图表标题
    """
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'gemv_bandwidth_plot_{timestamp}.png'
    
    plt.figure(figsize=(12, 7))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    for idx, (kernel_name, result_list) in enumerate(results.items()):
        # 过滤掉失败的结果
        valid_results = [r for r in result_list if 'error' not in r]
        if not valid_results:
            continue
        
        # 提取数据
        shape_labels = [f"{r['N']}x{r['K']}" for r in valid_results]
        bandwidths = [r['avg_bandwidth'] for r in valid_results]
        std_bws = [r['std_bandwidth'] for r in valid_results]
        
        x_pos = np.arange(len(shape_labels))
        
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        plt.errorbar(x_pos, bandwidths, yerr=std_bws,
                    label=kernel_name, marker=marker, markersize=8,
                    linestyle='-', linewidth=2, capsize=5,
                    color=color, alpha=0.8)
    
    # 设置图表
    if valid_results:  # 使用最后一个 kernel 的 shape 标签
        shape_labels = [f"{r['N']}x{r['K']}" for r in valid_results]
        plt.xticks(np.arange(len(shape_labels)), shape_labels, rotation=45, ha='right')
    
    plt.xlabel('Matrix Shape (NxK)', fontsize=12, fontweight='bold')
    plt.ylabel('Bandwidth (GB/s)', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] 图表已保存到: {output_path}")
    plt.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='统一 GEMV kernel 驱动带宽测试')
    parser.add_argument('--kernel', type=str, default='all', help='选择 kernel: all|q40_q80|q4k_q8k|iq4k_q8k')
    parser.add_argument('--n', type=int, default=512)
    parser.add_argument('--k', type=int, default=1024)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--target-gb', type=float, default=2.0)
    parser.add_argument('--num_threads', type=int, default=4)
    parser.add_argument('--peak-bw', type=float, default=DEFAULT_PEAK_BW, help=f'理论峰值带宽 (GB/s)，默认: {DEFAULT_PEAK_BW}')
    parser.add_argument('--sweep', action='store_true', help='对多个预定义 shape 进行批量测试')
    parser.add_argument('--csv', type=str, default=None, help='CSV 输出路径（仅在 --sweep 模式有效）')
    parser.add_argument('--plot', type=str, default=None, help='图表输出路径（仅在 --sweep 模式有效）')
    args = parser.parse_args()

    if args.sweep:
        # 批量测试模式
        results = run_sweep(
            kernel_names=args.kernel,
            shapes=None,  # 使用默认 SHAPE_CONFIGS
            rounds=args.rounds,
            warmup=args.warmup,
            target_gb=args.target_gb,
            threads=args.num_threads,
            peak_bw=args.peak_bw,
            csv_path=args.csv,
        )
        # 绘图
        plot_bandwidth_comparison(results, output_path=args.plot)
    else:
        # 单一 shape 测试
        if args.kernel == 'all':
            run_all(args.n, args.k, args.rounds, args.warmup, args.target_gb, args.num_threads, args.peak_bw)
        else:
            run_bandwidth(args.kernel, args.n, args.k, args.rounds, args.warmup, args.target_gb, args.num_threads, args.peak_bw)
