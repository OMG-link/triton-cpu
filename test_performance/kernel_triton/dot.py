"""
DOT test + Benchmark (fixed)
"""

import argparse
import math
import torch
import traceback

import triton
import triton.language as tl
import triton.testing as tt

# --- dtype maps ---
DTYPE_CONFIG = {
    "i8":   (torch.int8,   tl.int8),
    "i16":  (torch.int16,  tl.int16),
    "i32":  (torch.int32,  tl.int32),
    "fp16": (torch.float16, tl.float16),
    "bf16": (torch.bfloat16, tl.bfloat16),
    "fp32": (torch.float32, tl.float32),
}
BITWIDTH = {
    "i8": 8,
    "i16": 16,
    "i32": 32,
    "fp16": 16,
    "bf16": 16,
    "fp32": 32,
}

# --- Python side integer constants (for packing / grid calculation) ---
MR_IN_VAL = 4
NR_IN_VAL = 4
KC_IN_VAL = 256
MR_OUT_VAL = 4
NR_OUT_VAL = 32
KC_OUT_VAL = 256

def cdiv(a, b):
    return (a + b - 1) // b

# --- Triton kernels keep using tl.constexpr ---
MR_IN = tl.constexpr(MR_IN_VAL)
NR_IN = tl.constexpr(NR_IN_VAL)
KC_IN = tl.constexpr(KC_IN_VAL)

@triton.jit
def matmul_inner_kernel(a_ptr_raw, b_ptr_raw, c_ptr_raw, M, K, N):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    out_dtype = TL_OUT_DTYPE

    i_mr = pid_m * MR_IN
    i_nr = pid_n * NR_IN

    a_ptr = tl.make_block_ptr(
        base=a_ptr_raw,
        shape=(M, K),
        strides=(K, 1),
        offsets=(i_mr, 0),
        block_shape=(MR_IN, KC_IN),
        order=(1, 0),
    )
    b_ptr = tl.make_block_ptr(
        base=b_ptr_raw,
        shape=(N, K),
        strides=(K, 1),
        offsets=(i_nr, 0),
        block_shape=(NR_IN, KC_IN),
        order=(1, 0),
    )
    c_ptr = tl.make_block_ptr(
        base=c_ptr_raw,
        shape=(M, N),
        strides=(N, 1),
        offsets=(i_mr, i_nr),
        block_shape=(MR_IN, NR_IN),
        order=(1, 0),
    )
    c = tl.zeros((MR_IN, NR_IN), dtype=out_dtype)
    for i_kc in range(0, tl.cdiv(K, KC_IN)):
        a = tl.load(a_ptr)
        b = tl.load(b_ptr)
        c += tl.dot(a, b.T, out_dtype=out_dtype)
        a_ptr = tl.advance(a_ptr, (0, KC_IN))
        b_ptr = tl.advance(b_ptr, (0, KC_IN))
    tl.store(c_ptr, c)

MR_OUT = tl.constexpr(MR_OUT_VAL)
NR_OUT = tl.constexpr(NR_OUT_VAL)
KC_OUT = tl.constexpr(KC_OUT_VAL)

@triton.jit
def matmul_outer_kernel(a_ptr_raw, b_ptr_raw, c_ptr_raw, M, K, N):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    out_dtype = TL_OUT_DTYPE

    i_mr = pid_m * MR_OUT
    i_nr = pid_n * NR_OUT

    a_ptr = tl.make_block_ptr(
        base=a_ptr_raw,
        shape=(M//MR_OUT, K, MR_OUT),
        strides=(MR_OUT*K, MR_OUT, 1),
        offsets=(i_mr//MR_OUT, 0, 0),
        block_shape=(1, KC_OUT, MR_OUT),
        order=(2, 1, 0),
    )
    b_ptr = tl.make_block_ptr(
        base=b_ptr_raw,
        shape=(N//NR_OUT, K, NR_OUT),
        strides=(NR_OUT*K, NR_OUT, 1),
        offsets=(i_nr//NR_OUT, 0, 0),
        block_shape=(1, KC_OUT, NR_OUT),
        order=(2, 1, 0),
    )
    c_ptr = tl.make_block_ptr(
        base=c_ptr_raw,
        shape=(M, N),
        strides=(N, 1),
        offsets=(i_mr, i_nr),
        block_shape=(MR_OUT, NR_OUT),
        order=(1, 0),
    )
    c = tl.zeros((MR_OUT, NR_OUT), dtype=out_dtype)
    for i_kc in range(0, tl.cdiv(K, KC_OUT)):
        a = tl.load(a_ptr).reshape((KC_OUT, MR_OUT))
        b = tl.load(b_ptr).reshape((KC_OUT, NR_OUT))
        c += tl.dot(a.T, b, out_dtype=out_dtype)
        a_ptr = tl.advance(a_ptr, (0, KC_OUT, 0))
        b_ptr = tl.advance(b_ptr, (0, KC_OUT, 0))
    tl.store(c_ptr, c)

# --- prepare functions ---
def prepare_inner(a, b):
    """
    For inner kernel:
      - a: (M, K) layout, keep as-is
      - b: kernel expects (N, K) (so we transpose b: (K, N) -> (N, K))
    Returns tensors ready to be passed as a_ptr_raw and b_ptr_raw.
    """
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "A and B inner dims disagree"
    assert M % MR_IN_VAL == 0, f"M ({M}) must be divisible by MR_IN ({MR_IN_VAL})"
    assert N % NR_IN_VAL == 0, f"N ({N}) must be divisible by NR_IN ({NR_IN_VAL})"
    assert K % KC_IN_VAL == 0, f"N ({K}) must be divisible by KC_IN ({KC_IN_VAL})"

    # ensure contiguous
    a_c = a.contiguous()
    b_t = b.t().contiguous()
    return a_c, b_t

def prepare_outer(a, b):
    """
    Pack:
      A: [M, K] -> [M//MR_OUT, K, MR_OUT] where blocks are A[i_block, :, :] = A[i*MR:(i+1)*MR, :].T
      B: [K, N] -> [N//NR_OUT, K, NR_OUT] where blocks are B[j_block, :, :] = B[:, j*NR:(j+1)*NR].T
    """
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "A and B inner dims disagree"
    assert M % MR_OUT_VAL == 0, f"M ({M}) must be divisible by MR_OUT ({MR_OUT_VAL})"
    assert N % NR_OUT_VAL == 0, f"N ({N}) must be divisible by NR_OUT ({NR_OUT_VAL})"
    assert K % KC_OUT_VAL == 0, f"N ({K}) must be divisible by KC_OUT ({KC_OUT_VAL})"

    # pack A
    a_out = torch.empty((M // MR_OUT_VAL, K, MR_OUT_VAL), dtype=a.dtype, device=a.device)
    for i in range(M // MR_OUT_VAL):
        block = a[i * MR_OUT_VAL:(i + 1) * MR_OUT_VAL, :].contiguous().T
        a_out[i, :, :] = block

    # pack B
    b_out = torch.empty((N // NR_OUT_VAL, K, NR_OUT_VAL), dtype=b.dtype, device=b.device)
    for j in range(N // NR_OUT_VAL):
        block = b[:, j * NR_OUT_VAL:(j + 1) * NR_OUT_VAL].contiguous()  # (K, NR)
        b_out[j, :, :] = block

    return a_out.contiguous(), b_out.contiguous()

# --- Correctness test ---
def test_correctness(M, K, N):
    in_dtype = TORCH_IN_DTYPE
    out_dtype = TORCH_OUT_DTYPE

    torch.manual_seed(0)
    triton.runtime.driver.set_active_to_cpu()

    # gen data
    a = torch.randint(0, 128, (M, K), device='cpu').to(in_dtype)
    b = torch.randint(0, 128, (K, N), device='cpu').to(in_dtype)
    output_outer = torch.empty((M, N), device='cpu', dtype=out_dtype)
    output_inner = torch.empty((M, N), device='cpu', dtype=out_dtype)

    # pack data
    a_in, b_in = prepare_inner(a, b)
    a_out, b_out = prepare_outer(a, b)

    print(a_out.shape)

    # do matmul
    grid_outer = (cdiv(M, MR_OUT_VAL), cdiv(N, NR_OUT_VAL))
    matmul_outer_kernel[grid_outer](
        a_ptr_raw=a_out,
        b_ptr_raw=b_out,
        c_ptr_raw=output_outer,
        M=M, K=K, N=N
    )
    
    grid_inner = (cdiv(M, MR_IN_VAL), cdiv(N, NR_IN_VAL))
    matmul_inner_kernel[grid_inner](
        a_ptr_raw=a_in,
        b_ptr_raw=b_in,
        c_ptr_raw=output_inner,
        M=M, K=K, N=N
    )

    output_torch = torch.matmul(a.to(out_dtype), b.to(out_dtype))

    if torch.allclose(output_outer, output_torch, rtol=1e-5, atol=1e-5):
        ok_outer = True
    else:
        diff = output_outer - output_torch
        print("❌ OUTER: TritonCPU and TorchCPU differ, the maximum difference is "
              f'{torch.max(torch.abs(diff/output_torch)) * 100}' "%")
        ok_outer = False

    return (ok_inner, ok_outer)

    # diff output
    if torch.allclose(output_inner, output_torch, rtol=1e-5, atol=1e-5):
        ok_inner = True
    else:
        diff = output_inner - output_torch
        print("❌ INNER: TritonCPU and TorchCPU differ, the maximum difference is "
              f'{torch.max(torch.abs(diff/output_torch)) * 100}' "%")
        ok_inner = False

# -----------------------
# Benchmark helpers (refactored)
# -----------------------

def gflo_ps_from_ms(ms, M, N, K):
    # total flops assumed 2*M*N*K
    return 2.0 * M * N * K * 1e-9 / (ms * 1e-3)

def bench_kernel(kernel: str, M: int, K: int, N: int, rep_ms=200, warmup_ms=50, num_threads=None):
    """
    Generic benchmark wrapper for 'inner' or 'outer' kernel.
    kernel: 'inner' or 'outer'
    """
    device = 'cpu'
    if num_threads is not None:
        torch.set_num_threads(num_threads)

    torch.manual_seed(0)
    triton.runtime.driver.set_active_to_cpu()

    if kernel == 'outer':
        # prepare packed inputs expected by outer kernel
        # a: [M, K] -> packed a_out as prepare_outer returns
        a = torch.randn((M, K), device=device).to(TORCH_IN_DTYPE)
        b = torch.randn((K, N), device=device).to(TORCH_IN_DTYPE)
        a_in, b_in = prepare_outer(a, b)  # shapes: (M//MR_OUT, K, MR_OUT), (N//NR_OUT, K, NR_OUT)
        c = torch.empty((M, N), device=device, dtype=TORCH_OUT_DTYPE)
        grid = (cdiv(M, MR_OUT_VAL), cdiv(N, NR_OUT_VAL))

        def fn():
            matmul_outer_kernel[grid](a_ptr_raw=a_in, b_ptr_raw=b_in, c_ptr_raw=c, M=M, K=K, N=N)

    elif kernel == 'inner':
        # inner expects a (M, K) and b (N, K) (we'll use prepare_inner to get b transposed)
        a = torch.randn((M, K), device=device).to(TORCH_IN_DTYPE)
        b = torch.randn((K, N), device=device).to(TORCH_IN_DTYPE)
        a_in, b_in = prepare_inner(a, b)  # returns a, b.t()
        c = torch.empty((M, N), device=device, dtype=TORCH_OUT_DTYPE)
        grid = (cdiv(M, MR_IN_VAL), cdiv(N, NR_IN_VAL))

        def fn():
            matmul_inner_kernel[grid](a_ptr_raw=a_in, b_ptr_raw=b_in, c_ptr_raw=c, M=M, K=K, N=N)
    else:
        raise ValueError("kernel must be 'inner' or 'outer'")

    # run benchmark
    ms, _, _ = tt.do_bench(fn, warmup=warmup_ms, rep=rep_ms, quantiles=[0.5, 0.2, 0.8])
    gflops = gflo_ps_from_ms(ms, M, N, K)

    bitwidth = BITWIDTH[arg_in_dtype]
    denom = (256 / bitwidth) * 1.6e9
    utilization = gflops * 1e9 / denom

    print(f"[{kernel.upper()} UTILIZATION] M={M} K={K} N={N} | Utilization={utilization*100:.2f}%")
    return utilization

def run_kernel_bench_cases(cases=None, rep_ms=200, warmup_ms=50, num_threads=None):
    if cases is None:
        cases = [
            (32, 256, 32),
            (32, 2048, 32),
            (256, 256, 256),
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
        ]
    print("Running kernel-only benchmarks for OUTER and INNER kernels...")
    for M, K, N in cases:
        try:
            bench_kernel('outer', M, K, N, rep_ms=rep_ms, warmup_ms=warmup_ms, num_threads=num_threads)
        except Exception as e:
            print(f"[ERROR] OUTER Kernel bench failed for M={M},K={K},N={N}: {e}")
            print("Traceback:")
            traceback.print_tb(e.__traceback__)
        try:
            bench_kernel('inner', M, K, N, rep_ms=rep_ms, warmup_ms=warmup_ms, num_threads=num_threads)
        except Exception as e:
            print(f"[ERROR] INNER Kernel bench failed for M={M},K={K},N={N}: {e}")
            print("Traceback:")
            traceback.print_tb(e.__traceback__)

# -----------------------
# Main
# -----------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run correctness tests or kernel-only benchmarks for DOT tests.")
    parser.add_argument('--bench-kernel', action='store_true', help='Run kernel-only benchmarks')
    parser.add_argument('--rep-ms', type=int, default=200, help='do_bench rep time in ms')
    parser.add_argument('--warmup-ms', type=int, default=50, help='do_bench warmup time in ms')
    parser.add_argument('--num-threads', type=int, default=None, help='torch.set_num_threads() (None = unchanged)')
    parser.add_argument("--in-dtype", type=str, default="fp32", choices=DTYPE_CONFIG.keys(),
                        help="Input data type (i8, i16, i32, fp16, bf16, fp32)")
    parser.add_argument("--out-dtype", type=str, default="fp32", choices=DTYPE_CONFIG.keys(),
                        help="Output/accumulator data type (i8, i16, i32, fp16, bf16, fp32)")
    args, unknown = parser.parse_known_args()
    arg_in_dtype = args.in_dtype
    arg_out_dtype = args.out_dtype
    TORCH_IN_DTYPE, TL_IN_DTYPE = DTYPE_CONFIG[args.in_dtype]
    TORCH_OUT_DTYPE, TL_OUT_DTYPE = DTYPE_CONFIG[args.out_dtype]

    correctness_tests = (
        (12, 256, 32),
    )

    is_inner_good = True
    is_outer_good = True
    for (M, K, N) in correctness_tests:
        ok_inner, ok_outer = test_correctness(M=M, K=K, N=N)
        is_inner_good = is_inner_good and ok_inner
        is_outer_good = is_outer_good and ok_outer

    if is_inner_good:
        print("✅ INNER: TritonCPU and TorchCPU match")

    if is_outer_good:
        print("✅ OUTER: TritonCPU and TorchCPU match")

    if args.bench_kernel:
        if is_outer_good and is_inner_good:
            run_kernel_bench_cases(rep_ms=args.rep_ms, warmup_ms=args.warmup_ms, num_threads=args.num_threads)
        else:
            print("Skipping kernel benchmark because correctness tests failed.")
