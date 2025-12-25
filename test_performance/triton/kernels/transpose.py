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
    MR: tl.constexpr, TM: tl.constexpr, TN: tl.constexpr,
    dtype: tl.constexpr
):
    pid_m = tl.program_id(axis=0)

    i_mr = pid_m

    in_ptr_base = tl.make_block_ptr(
        base=in_ptr_raw,
        shape=(M // MR, MR, N),
        strides=(MR*N, N, 1),
        offsets=(i_mr, 0, 0),
        block_shape=(1, TM, TN),
        order=(2, 1, 0),
    )

    out_ptr_base = tl.make_block_ptr(
        base=out_ptr_raw,
        shape=(M // MR, N, MR),
        strides=(N*MR, MR, 1),
        offsets=(i_mr, 0, 0),
        block_shape=(1, TN, TM),
        order=(2, 1, 0),
    )

    for i_tn in range(0, N // TN):
        for i_tm in range(0, MR // TM):
            in_ptr = in_ptr_base.advance((0, i_tm*TM, i_tn*TN))
            out_ptr = out_ptr_base.advance((0, i_tn*TN, i_tm*TM))
            tl.store(out_ptr, tl.load(in_ptr).reshape(TM, TN).T.reshape(1, TN, TM))


class TransposeKernel(GEMMKernelBase):
    def __init__(self, dtype: str, MR: int, TM: int, TN: int, device: str = "cpu"):
        assert dtype in GEMMKernelBase.DTYPE_CONFIG
        self.dtype_name = dtype
        self.device = device
        self.MR = MR
        self.TM = TM
        self.TN = TN

    def prepare(self, m: int, k: int, n: int = 1, should_gen_data: bool = True) -> dict:
        TORCH_DTYPE, TL_DTYPE = GEMMKernelBase.DTYPE_CONFIG[self.dtype_name]

        assert n == 1, "Argument 'n' is fixed to 1 in this kernel."

        M, N = m, k
        MR, TM, TN = self.MR, self.TM, self.TN

        assert M % MR == 0, f"M ({M}) must be divisible by MR ({MR})"
        assert MR % TM == 0, f"MR ({MR}) must be divisible by TM ({TM})"
        assert N % TN == 0, f"N ({N}) must be divisible by TN ({TN})"

        if should_gen_data:
            torch.manual_seed(0)
            inp = torch.randint(0, 128, (M, N), device=self.device).to(TORCH_DTYPE).contiguous()
        else:
            inp = torch.empty((M ), device=self.device, dtype=TORCH_DTYPE).contiguous()

        out = torch.empty((M // MR, N, MR), device=self.device, dtype=TORCH_DTYPE)

        return {
            'in': inp,
            'out': out,
            'M': M, 'N': N,
            'MR': MR, 'TM': TM, 'TN': TN,
            'torch_dtype': TORCH_DTYPE,
            'tl_dtype': TL_DTYPE,
        }

    def run(self, params: dict, repeats: int = 1) -> float:
        TL_DTYPE = params['tl_dtype']

        inp = params['in']
        out = params['out']

        M, N = params['M'], params['N']
        MR, TM, TN = params['MR'], params['TM'], params['TN']

        grid = (cdiv(M, MR), )

        t0 = time.perf_counter()
        transpose_kernel[grid](
            inp, out,
            M, N,
            MR=MR, TM=TM, TN=TN,
            dtype=TL_DTYPE,
            n_kernel_repeat=repeats,
        )
        t1 = time.perf_counter()
        return (t1 - t0) / repeats

    def verify(self, params: dict) -> bool:
        inp = params['in']
        out = params['out']

        M, N = params['M'], params['N']
        MR = params['MR']

        ref = inp.reshape(M // MR, MR, N).permute(0, 2, 1).contiguous()
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
