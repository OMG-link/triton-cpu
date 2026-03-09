"""
GGUF q4k x q8k
=====================
In this test, matmul on CPU with k-quantize is tested.

Transpose is done by 'tl.transpose'(A.T) operand.

"""

import argparse
import math
import os
import torch

import subprocess
import sys

import numpy as np
import triton
import triton.language as tl
import triton.testing as tt
import time

MR = tl.constexpr(4)
NR = tl.constexpr(32) # 如何获取CPU的VLEN作为NR?

QK_K = tl.constexpr(256)
QK_SB_K = tl.constexpr(32)


def cdiv(a, b):
    return (a + b - 1) // b

def quantize_q8_K(a: torch.Tensor):
    """
    a: torch.Tensor float32 shape (M, K)
    returns: a_q_ptr (int8), a_bsums_ptr (int16), a_d_ptr (float32)
    shapes:
      a_q_ptr:   (M//MR, K//QK_K, QK_K//QK_SB_K, 2, QK_SB_K//2, MR)   dtype=int8
      a_bsums:   (M//MR, K//QK_K, QK_K//QK_SB_K, MR)           dtype=int16
      a_d_ptr:   (M//MR, K//QK_K, MR)                          dtype=float32
    """
    M, K = a.shape
    device = a.device
    Mb = cdiv(M, MR)
    Ksup = cdiv(K, QK_K)
    Ksub = QK_K // QK_SB_K  # typically 8

    # pad rows & cols to block multiples
    M_p = Mb * MR
    K_p = Ksup * QK_K
    a_pad = torch.zeros((M_p, K_p), dtype=a.dtype, device=device)
    a_pad[:M, :K] = a

    # allocate outputs on CPU with layout matching kernel's make_block_ptr expectation:
    # shape: (Mb, Ksup, Ksub, 2, QK_SB_K//2, MR) - 6D tensor
    a_q = torch.zeros((Mb, Ksup, Ksub, 2, QK_SB_K//2, MR), dtype=torch.int8, device='cpu')
    a_bsums = torch.zeros((Mb, Ksup, Ksub, MR), dtype=torch.int16, device='cpu')
    a_d = torch.zeros((Mb, Ksup, MR), dtype=torch.float32, device='cpu')

    # iterate mb, ksup
    for mb in range(Mb):
        row_start = mb * MR
        rows = a_pad[row_start: row_start + MR, :]  # shape (MR, K_p)
        for ksup in range(Ksup):
            k0 = ksup * QK_K
            seg = rows[:, k0: k0 + QK_K]   # shape (MR, QK_K)
            # compute per-row scale (symmetric int8 -> range [-127,127])
            max_abs = seg.abs().amax(dim=1)  # (MR,)
            # avoid zero scale
            scale = torch.where(max_abs > 0, max_abs / 127.0, torch.ones_like(max_abs))
            a_d[mb, ksup, :] = scale.to(torch.float32)
            # quantize per row
            # q = round(x / scale)
            # expand scale to shape (MR, QK_K)
            scale_exp = scale.unsqueeze(1).expand(-1, QK_K)
            q = torch.round(seg / scale_exp).clamp(-127, 127).to(torch.int8)  # (MR, QK_K)
            # fill a_q and a_bsums (store as (2, QK_SB_K//2, MR) per subblock)
            for s in range(Ksub):
                s0 = s * QK_SB_K
                q_sub = q[:, s0:s0 + QK_SB_K]  # (MR, QK_SB_K)
                # split into low/high halves and transpose each
                q_low = q_sub[:, :QK_SB_K//2].t().contiguous()   # (QK_SB_K//2, MR)
                q_high = q_sub[:, QK_SB_K//2:].t().contiguous()  # (QK_SB_K//2, MR)
                a_q[mb, ksup, s, 0, :, :] = q_low
                a_q[mb, ksup, s, 1, :, :] = q_high
                # bsums: sum across the 32 elements for each MR row
                sums = q_sub.sum(dim=1).to(torch.int16)  # (MR,)
                a_bsums[mb, ksup, s, :] = sums

    # bring outputs to CPU (kernel's block_ptr expects host tensors)
    return a_q, a_bsums, a_d

def quantize_q4_K(b: torch.Tensor):
    """
    非对称 q4_K 实现（解码为:  orig ≈ d * (s_q * q) - dmin * m_q）
    b: (K, N)
    returns: b_q (Nb, Ksup, Ksub, QK_SB_K//2, NR) uint8 (两个int4打包)
             b_scales (Nb, Ksup, Ksub, NR)   int8  (signed, -32..31)
             b_mins (Nb, Ksup, Ksub, NR)     int16 (signed, we store -m_q)
             b_d (Nb, Ksup, NR)              float16
             b_dmin (Nb, Ksup, NR)           float16
    """
    K, N = b.shape
    device = b.device
    Nb = cdiv(N, NR)
    Ksup = cdiv(K, QK_K)
    Ksub = QK_K // QK_SB_K

    # pad
    K_p = Ksup * QK_K
    N_p = Nb * NR
    b_pad = torch.zeros((K_p, N_p), dtype=b.dtype, device=device)
    b_pad[:K, :N] = b

    # outputs - b_q 现在是打包格式，每个 uint8 存两个 4bit
    b_q = torch.zeros((Nb, Ksup, Ksub, QK_SB_K//2, NR), dtype=torch.uint8, device='cpu')   # packed int4x2
    b_scales = torch.zeros((Nb, Ksup, Ksub, NR), dtype=torch.int8, device='cpu')      # signed 6-bit stored in int8
    b_mins = torch.zeros((Nb, Ksup, ksub := Ksub, NR), dtype=torch.int16, device='cpu')  # store -m_q (int16)
    b_d = torch.zeros((Nb, Ksup, NR), dtype=torch.float16, device='cpu')
    b_dmin = torch.zeros((Nb, Ksup, NR), dtype=torch.float16, device='cpu')

    # iterate nb (NR-chunk), ksup
    for nb in range(Nb):
        col_start = nb * NR
        cols = b_pad[:, col_start: col_start + NR]  # (K_p, NR)
        for ksup in range(Ksup):
            k0 = ksup * QK_K
            seg = cols[k0: k0 + QK_K, :]   # (QK_K, NR)

            # compute per-subblock real scales and mins
            s_reals = torch.zeros((Ksub, NR), dtype=torch.float32, device=device)
            m_reals = torch.zeros((Ksub, NR), dtype=torch.float32, device=device)
            for s in range(Ksub):
                s0 = s * QK_SB_K
                sub = seg[s0:s0 + QK_SB_K, :]    # (QK_SB_K, NR)
                sub_min = sub.amin(dim=0)       # (NR,)
                sub_max = sub.amax(dim=0)       # (NR,)
                s_real = (sub_max - sub_min) / 15.0    # per-column scale for this subblock
                # guard zeros
                s_real = torch.where(s_real > 0, s_real, torch.zeros_like(s_real))
                s_reals[s, :] = s_real
                m_reals[s, :] = sub_min

            # choose super-block d and dmin so that s_q and m_q fit into 6-bit signed [-31..31]
            max_s_real = s_reals.abs().amax(dim=0)   # (NR,)
            max_m_abs = m_reals.abs().amax(dim=0)    # (NR,)

            # avoid zero -> set to 1.0 to prevent division by zero (will make s_q/m_q == 0)
            d = torch.where(max_s_real > 0, max_s_real / 31.0, torch.ones_like(max_s_real))
            dmin = torch.where(max_m_abs > 0, max_m_abs / 31.0, torch.ones_like(max_m_abs))

            # store d / dmin as float16
            b_d[nb, ksup, :] = d.to(torch.float16)
            b_dmin[nb, ksup, :] = dmin.to(torch.float16)

            # compute per-subblock integer s_q and m_q
            # s_q = round(s_real / d)  (should be non-negative, clamp to [-32,31])
            # m_q = round(m_real / dmin)
            for s in range(Ksub):
                s_real = s_reals[s, :]   # (NR,)
                m_real = m_reals[s, :]   # (NR,)

                # s_q (signed 6-bit). Use division by d chosen above.
                # if s_real is zero, s_q becomes 0.
                s_q = torch.round(torch.where(d > 0, s_real / d, torch.zeros_like(s_real))).to(torch.int32)
                s_q = torch.clamp(s_q, -32, 31).to(torch.int8)

                # m_q (signed)
                m_q = torch.round(torch.where(dmin > 0, m_real / dmin, torch.zeros_like(m_real))).to(torch.int16)
                # store -m_q so that kernel doing sum_row - sum_min_row recovers +min
                b_scales[nb, ksup, s, :] = s_q
                b_mins[nb, ksup, s, :] = (-m_q).to(torch.int16)

            # now quantize per element inside each subblock using the real subblock scale/min
            # 并打包两个 4bit 到一个 uint8
            for s in range(Ksub):
                s0 = s * QK_SB_K
                sub = seg[s0:s0 + QK_SB_K, :]  # (QK_SB_K, NR)
                s_real = s_reals[s, :].unsqueeze(0).expand(QK_SB_K, -1)  # (QK_SB_K, NR)
                m_real = m_reals[s, :].unsqueeze(0).expand(QK_SB_K, -1)

                # avoid divide by zero: when s_real == 0, set q = 0
                zero_mask = (s_real == 0)
                q_sub = torch.zeros_like(sub, dtype=torch.int32)
                nonzero_mask = ~zero_mask
                if nonzero_mask.any():
                    q_real = torch.round((sub - m_real) / torch.where(s_real != 0, s_real, torch.ones_like(s_real)))
                    q_sub = q_real.clamp(0, 15).to(torch.uint8)

                # 打包两个 4bit 到一个 uint8: low 4bit = q[0..15], high 4bit = q[16..31]
                q_packed = torch.zeros((QK_SB_K//2, NR), dtype=torch.uint8, device=device)
                for i in range(QK_SB_K // 2):
                    q_low = q_sub[i, :] & 0x0F                 # 低 4 bit (k0..k15)
                    q_high = q_sub[i + QK_SB_K//2, :] & 0x0F   # 高 4 bit (k16..k31)
                    q_packed[i, :] = q_low | (q_high << 4)     # 打包

                b_q[nb, ksup, s, :, :] = q_packed.contiguous()

    return b_q, b_scales, b_mins, b_d, b_dmin

# -----------------------
# Triton kernel
# -----------------------
@triton.jit
def q4k_q8k_matmul_kernel(
    # q8k 输入
    q8k_matrix_ptr,         # int8     A        (M//MR, K//QK_K, QK_K//QK_SB_K, QK_SB_K, MR)
    q8k_bsums_ptr,          # int16    a_bsums  (M//MR, K//QK_K, QK_K//QK_SB_K, MR)
    q8k_d_ptr,              # float32  a_d      (M//MR, K//QK_K, MR)
    
    q4k_matrix_ptr,         # int8     Bpacked  (N//NR, K//QK_K, QK_K//QK_SB_K, 2, QK_SB_K // 2, NR)
    q4k_scale_ptr,         # int8     b_scales (N//NR, K//QK_K, QK_K//QK_SB_K, NR)
    q4k_mins_ptr,           # int8     b_scales (N//NR, K//QK_K, QK_K//QK_SB_K, NR)
    q4k_d_ptr,              # float16  b_d      (N//NR, K//QK_K, NR)
    q4k_dmin_ptr,           # float16  b_dmin   (N//NR, K//QK_K, NR)
    output_ptr,                # float32  C        (M, N)
    M, N, K,
):
    pid_m = tl.program_id(axis=0) 
    pid_n = tl.program_id(axis=1) 

    i_mr = pid_m * MR 
    i_nr = pid_n * NR 

    a_q_ptr_start = tl.make_block_ptr(
        base=q8k_matrix_ptr,
        shape=(M//MR, K//QK_K, QK_K//QK_SB_K, 2, QK_SB_K//2, MR),
        strides=(K*MR, QK_K*MR, QK_SB_K*MR, QK_SB_K*MR // 2, MR, 1),
        offsets=(pid_m, 0, 0, 0, 0, 0),
        block_shape=(1, 1, 1, 1, QK_SB_K//2, MR),
        order=(5, 4, 3, 2, 1, 0),
    )

    a_bsums_ptr_start = tl.make_block_ptr(
        base=q8k_bsums_ptr,
        shape=(M//MR, K//QK_K, QK_K//QK_SB_K, MR),
        strides=(K//QK_SB_K*MR, QK_K//QK_SB_K*MR, MR, 1),
        offsets=(pid_m, 0, 0, 0),
        block_shape=(1, 1, QK_K//QK_SB_K, MR),
        order=(3, 2, 1, 0),
    )

    a_d_ptr_start = tl.make_block_ptr(
        base=q8k_d_ptr,
        shape=(M//MR, K//QK_K, MR),
        strides=(K//QK_K*MR, MR, 1),
        offsets=(pid_m, 0, 0),
        block_shape=(1, 1, MR),
        order=(2, 1, 0),
    )

    b_q_ptr_start = tl.make_block_ptr(
        base=q4k_matrix_ptr,
        shape=(N//NR, K//QK_K, QK_K//QK_SB_K, QK_SB_K//2, NR),
        strides=(K//2*NR, QK_K//2*NR, QK_SB_K//2*NR, NR, 1),
        offsets=(pid_n, 0, 0, 0, 0),
        block_shape=(1, 1, 1, QK_SB_K//2, NR),
        order=(4, 3, 2, 1, 0),
    )

    b_mins_ptr_start = tl.make_block_ptr(
        base=q4k_mins_ptr,
        shape=(N//NR, K//QK_K, QK_K//QK_SB_K, NR),
        strides=(K//QK_SB_K*NR, QK_K//QK_SB_K*NR, NR, 1),
        offsets=(pid_n, 0, 0, 0),
        block_shape=(1, 1, QK_K//QK_SB_K, NR),
        order=(3, 2, 1, 0),
    )

    b_scales_ptr_start = tl.make_block_ptr(
        base=q4k_scale_ptr,
        shape=(N//NR, K//QK_K, QK_K//QK_SB_K, NR),
        strides=(K//QK_SB_K*NR, QK_K//QK_SB_K*NR, NR, 1),
        offsets=(pid_n, 0, 0, 0),
        block_shape=(1, 1, 1, NR),
        order=(3, 2, 1, 0),
    )
    
    b_d_ptr_start = tl.make_block_ptr(
        base=q4k_d_ptr,
        shape=(N//NR, K//QK_K, NR),
        strides=(K//QK_K*NR, NR, 1),
        offsets=(pid_n, 0, 0),
        block_shape=(1, 1, NR),
        order=(2, 1, 0),
    )

    b_dmin_ptr_start = tl.make_block_ptr(
        base=q4k_dmin_ptr,
        shape=(N//NR, K//QK_K, NR),
        strides=(K//QK_K*NR, NR, 1),
        offsets=(i_nr//NR, 0, 0),
        block_shape=(1, 1, NR),
        order=(2, 1, 0),
    )

    c_ptr_start = tl.make_block_ptr(
        base=output_ptr,
        shape=(M, N),
        strides=(N, 1),
        offsets=(i_mr, i_nr),
        block_shape=(MR, NR),
        order=(1, 0),
    )

    # sum_row
    sum_row = tl.zeros((MR, NR), dtype=tl.float32)

    # sum_min_row
    sum_min_row = tl.zeros((MR, NR), dtype=tl.float32) 
    c_output = tl.zeros((MR, NR), dtype=tl.float32) 

    for i_supb in range(0, tl.cdiv(K, QK_K)):
        sum_block = tl.zeros((MR, NR), dtype=tl.int32)
        submin = tl.zeros((MR, NR), dtype=tl.int32)
        a_q_ptr = tl.advance(a_q_ptr_start, (0, i_supb, 0, 0, 0, 0))
        b_q_ptr = tl.advance(b_q_ptr_start, (0, i_supb, 0, 0, 0))
        a_d_ptr     = tl.advance(a_d_ptr_start, (0, i_supb, 0)) 
        b_d_ptr     = tl.advance(b_d_ptr_start, (0, i_supb, 0))
        b_dmin_ptr  = tl.advance(b_dmin_ptr_start, (0, i_supb, 0)) 
        a_bsums_ptr = tl.advance(a_bsums_ptr_start,  (0, i_supb, 0, 0))
        b_mins_ptr = tl.advance(b_mins_ptr_start, (0, i_supb, 0, 0))


        for i_subb in range(0, tl.cdiv(QK_K, QK_SB_K)):
            # 加载 packed int4 数据 (16, NR)，每个 uint8 包含两个 int4
            b_q_packed = tl.load(b_q_ptr).reshape((QK_SB_K//2, NR))
            # 解压两个 int4: 低4位和高4位
            b_q_low = tl.cast(b_q_packed & 0x0F, tl.int8)   # (16, NR) 对应 k=0..15
            b_q_high = tl.cast((b_q_packed >> 4), tl.int8)  # (16, NR) 对应 k=16..31
            # 加载 a_q 的低半部分 (第3维=0) 与高4位做 dot
            a_q_low_ptr = tl.advance(a_q_ptr, (0, 0, i_subb, 0, 0, 0))
            a_q_low = tl.load(a_q_low_ptr).reshape((QK_SB_K//2, MR))
            suml = tl.dot(a_q_low.T, b_q_low, out_dtype=tl.int16)
            # 加载 a_q 的高半部分 (第3维=1) 与低4位做 dot
            a_q_high_ptr = tl.advance(a_q_ptr, (0, 0, i_subb, 1, 0, 0))
            a_q_high = tl.load(a_q_high_ptr).reshape((QK_SB_K//2, MR))
            suml += tl.dot(a_q_high.T, b_q_high, out_dtype=tl.int16)
            b_scales_ptr = tl.advance(b_scales_ptr_start, (0, i_supb, i_subb, 0))
            b_scales = tl.load(b_scales_ptr).reshape((1, NR))
            sum_block += tl.cast(suml, tl.int32) * tl.cast(b_scales, tl.int32)
            # i_subb++: 在 subblock 维度 (第2维) 上前进，同时重置第3维
            a_q_ptr = tl.advance(a_q_ptr, (0, 0, 1, 0, 0, 0))
            b_q_ptr = tl.advance(b_q_ptr, (0, 0, 1, 0, 0))

        a_d = tl.load(a_d_ptr).reshape((1, MR)) 
        b_d = tl.cast(tl.load(b_d_ptr).reshape((1, NR)), tl.float32) 
        # 超块边界 
        #TODO 需要 spill sum_block
        sum_row += tl.cast(sum_block, tl.float32) * tl.dot(a_d.T, b_d, out_dtype=tl.float32) # [12, 32] @ float32 

        a_bsums = tl.load(a_bsums_ptr).reshape((QK_K//QK_SB_K, MR)) #[8, 12] 
        b_mins = tl.cast(tl.load(b_mins_ptr).reshape((QK_K//QK_SB_K, NR)), tl.int16) #[8, 32] 
        a_d = tl.load(a_d_ptr).reshape((1, MR)) 
        b_dmin = tl.cast(tl.load(b_dmin_ptr).reshape((1, NR)), tl.float32) 
        # submin = tl.cast(tl.dot(a_bsums.T, b_mins, out_dtype=tl.int32) ，超块边界 
        # TODO 需要 spill sum_row
        sum_min_row += tl.dot(a_d.T, b_dmin, out_dtype=tl.float32) * tl.cast(tl.dot(a_bsums.T, b_mins, out_dtype=tl.int32), tl.float32) 

    
    # compute final result
    c_ptr = c_ptr_start 
    tl.store(c_ptr, sum_row - sum_min_row) 
 
def gflo_ps_from_ms(ms, M, N, K):
    # total flops assumed 2*M*N*K
    return 2.0 * M * N * K * 1e-9 / (ms * 1e-3)

def bench_kernel_only_case(M, K, N, rep_ms=100, warmup_ms=10, num_threads=None, n_kernel_repeat=1):
    """
    Pre-quantize inputs once, then benchmark only the Triton kernel invocation.
    Returns timing only - performance calculation moved to driver.
    """
    device = 'cpu'

    # create random original tensors (only to create quantized inputs)
    torch.manual_seed(0)
    a = torch.randn((M, K), device=device, dtype=torch.float32)
    b = torch.randn((K, N), device=device, dtype=torch.float32)

    # Precompute quantized representations (these are CPU tensors and are reused)
    a_q_ptr, a_bsums_ptr, a_d_ptr = quantize_q8_K(a)
    b_q_ptr, b_scales_ptr, b_mins_ptr, b_d_ptr, b_dmin_ptr = quantize_q4_K(b)

    # pre-allocate output
    c = torch.empty((M, N), device='cpu', dtype=torch.float32)

    # grid size (same pattern your matmul used)
    grid = (cdiv(M, MR), cdiv(N, NR))

    # function to benchmark: only the kernel launch
    def fn():
        q4k_q8k_matmul_kernel[grid](
            a_q_ptr, a_bsums_ptr, a_d_ptr,
            b_q_ptr, b_scales_ptr, b_mins_ptr, b_d_ptr, b_dmin_ptr,
            c, M, N, K, num_threads=num_threads
        )

    # run benchmark
    ms, min_ms, max_ms = tt.do_bench(fn, warmup=warmup_ms, rep=rep_ms, quantiles=[0.5, 0.2, 0.8])
    return ms, min_ms, max_ms


# Integrate with argparse: add a --bench-kernel flag
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run kernel-only benchmarks.")
    parser.add_argument('--rep-ms', type=int, default=100, help='do_bench rep time in ms')
    parser.add_argument('--warmup-ms', type=int, default=10, help='do_bench warmup time in ms')
    parser.add_argument('--num-threads', type=int, default=1, help='torch.set_num_threads() (None = unchanged)')
    parser.add_argument('--shapes', type=str, default='', help='额外形状, 逗号分隔, 例如 48x512x32,96x256x32')
    parser.add_argument('--n-kernel-repeat', type=int, default=30, help='每次调用内核时重复的次数')
    args, unknown = parser.parse_known_args()

    # run the original correctness tests (keeps previous behavior)
    tests = [
        (1200, 2048, 2048),
    ]
    
    if args.shapes:
        for item in args.shapes.split(','):
            item = item.strip()
            if not item:
                continue
            try:
                m,k,n = item.lower().split('x')
                tests.append((int(m), int(k), int(n)))
            except Exception as e:
                print(f"忽略无效形状 '{item}': {e}")

    for M, K, N in tests:
        mean_ms, max_ms, min_ms = bench_kernel_only_case(
            M=M,
            N=N,
            K=K,
            rep_ms=args.rep_ms,
            warmup_ms=args.warmup_ms,
            num_threads=args.num_threads,
            n_kernel_repeat=args.n_kernel_repeat
        )
        
        # 使用默认芯片参数计算性能（独立运行时）
        freq, vlen = 1.6, 256
        peak_flops = 2.0 * vlen / 16 * freq
        gflops = gflo_ps_from_ms(mean_ms, M, N, K)
        gflops_max = gflo_ps_from_ms(min_ms, M, N, K)
        gflops_min = gflo_ps_from_ms(max_ms, M, N, K)
        utilization = gflops / peak_flops * 100.0
        utilization_max = gflops_max / peak_flops * 100.0
        utilization_min = gflops_min / peak_flops * 100.0
        print(f"[KERNEL BENCH] M={M} K={K} N={N} | median_ms={mean_ms:.3f} min_ms={min_ms:.3f} max_ms={max_ms:.3f} | "
            f"GFLOPS(med)={gflops:.2f} GFLOPS(min)={gflops_min:.2f} GFLOPS(max)={gflops_max:.2f} | "
            f"Utilization(med)={utilization:.2f}% Utilization(min)={utilization_min:.2f}% Utilization(max)={utilization_max:.2f}%") 

# -----------------------
# 统一驱动注册接口 - 简化版本，逻辑移到 driver
# -----------------------

def get_kernel_info():
    """返回 kernel 的基本信息，供 driver 调用"""
    return {
        'name': 'q4k_q8k',
        'bench_fn': bench_kernel_only_case,
        'constraints': {'M': 12, 'N': 32, 'K': 256},  # 块大小约束
        'compute_dtype': 'int8',  # 实际计算使用的数据类型
        'dtype_width': 8,  # bits
    }
        


