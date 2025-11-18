import torch
import time
import numpy as np

import triton
import triton.language as tl

MR = tl.constexpr(12)
NR = tl.constexpr(32) # 如何获取CPU的VLEN作为NR?

QK_K = tl.constexpr(256)
QK_SB_K = tl.constexpr(32)

@triton.jit
def q40_q80_gemm_kernel(
    q4_0_matrix_ptr,         # q4_0 矩阵指针
    q4_0_scale_ptr,          # q4_0 量化比例指针
    q8_0_matrix_ptr,         # q8_0 矩阵指
    q8_0_scale_ptr,          # q8_0 量化比例指针
    output_ptr,              # 输出指针
    M,                       # 矩阵行数
    N,                       # 矩阵列数
    K,                       # 矩阵公共维度
):
