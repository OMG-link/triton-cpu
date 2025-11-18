import torch
import time
import numpy as np

import triton
import triton.language as tl

BLOCK_SIZE_M = 1
BLOCK_SIZE_N = 512
USE_GPU = False


@triton.jit
def q40_q80_gemv_kernel(
    q8_k_vector_ptr,  # int8 
    q8_k_scale_ptr,   # uint16 
    q4_k_matrix_ptr,  # int4 packed in uint8 
    q4_k_scale_ptr,   # uint16 
    output_ptr,       # float32 
    K,
    N,
):
    pid_n = tl.program_id(0)
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    k_offsets = tl.arange(0, K)

    q8_k_vector = tl.load(q8_k_vector_ptr + k_offsets)
    q8_k_scale = tl.load(q8_k_scale_ptr + k_offsets)

    q4_k_matrix = tl.load(q4_k_matrix_ptr + k_offsets[:, None] * (N // 2) + n_offsets[None, :] // 2)
    q4_k_scale = tl.load(q4_k_scale_ptr + k_offsets)

    q4_low = tl.cast(tl.bitwise_and(q4_k_matrix, 0x0F), tl.int32)
    q4_high = tl.cast(tl.bitwise_and(tl.shift_right(q4_k_matrix, 4), 0x0F), tl.int32)
    q4_values = tl.where(n_offsets[None, :] % 2 == 0, q4_low, q4_high)

    q4_dequantized = tl.cast(q4_values, tl.float32) * tl.cast(q4_k_scale[:, None], tl.float32)
    q8_dequantized = tl.cast(q8_k_vector, tl.float32) * tl.cast(q8_k_scale, tl.float32)

    prod = q4_dequantized * q8_dequantized[:, None]
    sum_prod = tl.sum(prod, axis=0)

    output_ptr_offset = output_ptr + n_offsets
    tl.store(output_ptr_offset, sum_prod)
