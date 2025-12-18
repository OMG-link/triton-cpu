# transpose.py
import time
import torch
import triton
import triton.language as tl

from gemm_base import GEMMKernelBase
from abc import ABC

def cdiv(a, b):
    return (a + b - 1) // b


@triton.jit
def transpose_kernel(
    in_ptr_raw, out_ptr_raw,
    M, N,
    TM: tl.constexpr, TN: tl.constexpr,
    dtype: tl.constexpr
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    i_tm = pid_m * TM
    i_tn = pid_n * TN

    in_ptr = tl.make_block_ptr(
        base=in_ptr_raw,
        shape=(M, N),
        strides=(N, 1),
        offsets=(i_tm, i_tn),
        block_shape=(TM, TN),
        order=(1, 0),
    )

    out_ptr = tl.make_block_ptr(
        base=out_ptr_raw,
        shape=(N, M),
        strides=(M, 1),
        offsets=(i_tn, i_tm),
        block_shape=(TN, TM),
        order=(1, 0),
    )

    a = tl.load(in_ptr)
    tl.store(out_ptr, a.T)


class TransposeKernel(GEMMKernelBase):
    def __init__(self, dtype: str, TM: int, TN: int, device: str = "cpu"):
        assert dtype in GEMMKernelBase.DTYPE_CONFIG
        self.dtype_name = dtype
        self.device = device
        self.TM = TM
        self.TN = TN

    def prepare(self, m: int, k: int, n: int = 1, should_gen_data: bool = True) -> dict:
        TORCH_DTYPE, TL_DTYPE = GEMMKernelBase.DTYPE_CONFIG[self.dtype_name]

        assert n == 1, "Argument 'n' is fixed to 1 in this kernel."

        M, N = m, k
        TM, TN = self.TM, self.TN

        assert M % TM == 0, f"M ({M}) must be divisible by TM ({TM})"
        assert N % TN == 0, f"N ({N}) must be divisible by TN ({TN})"

        if should_gen_data:
            torch.manual_seed(0)
            inp = torch.randint(0, 128, (M, N), device=self.device).to(TORCH_DTYPE).contiguous()
        else:
            inp = torch.empty((M, N), device=self.device, dtype=TORCH_DTYPE).contiguous()

        out = torch.empty((N, M), device=self.device, dtype=TORCH_DTYPE)

        return {
            'in': inp,
            'out': out,
            'M': M, 'N': N,
            'TM': TM, 'TN': TN,
            'torch_dtype': TORCH_DTYPE,
            'tl_dtype': TL_DTYPE,
        }

    def run(self, params: dict, repeats: int = 1) -> float:
        TL_DTYPE = params['tl_dtype']

        inp = params['in']
        out = params['out']

        M, N = params['M'], params['N']
        TM, TN = params['TM'], params['TN']

        grid = (cdiv(M, TM), cdiv(N, TN))

        t0 = time.perf_counter()
        transpose_kernel[grid](
            inp, out,
            M, N,
            TM=TM, TN=TN,
            dtype=TL_DTYPE,
            n_kernel_repeat=repeats,
        )
        t1 = time.perf_counter()
        return (t1 - t0) / repeats

    def verify(self, params: dict) -> bool:
        inp = params['in']
        out = params['out']

        ref = inp.t().contiguous()
        return bool(torch.equal(out, ref))

    def get_name(self) -> str:
        return f"{self.__class__.__name__}_{self.dtype_name}_{self.TM}x{self.TN}"

    def expected_cycles(
        self,
        m: int,
        k: int,
        n: int,
        peak_bandwidth_bytes_per_sec: float = 6.4e9,
        cpu_frequency_hz: float = 1.6e9
    ) -> int:
        M = m
        N = k

        bits = GEMMKernelBase.BITWIDTH[self.dtype_name]
        bytes_per_elem = bits / 8.0

        bytes_moved = 2.0 * M * N * bytes_per_elem
        time_seconds = bytes_moved / peak_bandwidth_bytes_per_sec
        cycles = time_seconds * cpu_frequency_hz

        return int(cycles)
