# outer.py
import time
import torch
import triton
import triton.language as tl

from gemm_base import GEMMKernelBase


def cdiv(a, b):
    return (a + b - 1) // b


@triton.jit
def matmul_outer_kernel(
    a_ptr_raw, b_ptr_raw, c_ptr_raw,
    M, K, N,
    MR: tl.constexpr, NR: tl.constexpr, KC: tl.constexpr,
    out_dtype: tl.constexpr
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    i_mr = pid_m * MR
    i_nr = pid_n * NR

    a_ptr = tl.make_block_ptr(
        base=a_ptr_raw,
        shape=(M // MR, K, MR),
        strides=(MR * K, MR, 1),
        offsets=(i_mr // MR, 0, 0),
        block_shape=(1, KC, MR),
        order=(2, 1, 0),
    )
    b_ptr = tl.make_block_ptr(
        base=b_ptr_raw,
        shape=(N // NR, K, NR),
        strides=(NR * K, NR, 1),
        offsets=(i_nr // NR, 0, 0),
        block_shape=(1, KC, NR),
        order=(2, 1, 0),
    )
    c_ptr = tl.make_block_ptr(
        base=c_ptr_raw,
        shape=(M, N),
        strides=(N, 1),
        offsets=(i_mr, i_nr),
        block_shape=(MR, NR),
        order=(1, 0),
    )

    c = tl.zeros((MR, NR), dtype=out_dtype)

    # loop over KC chunks
    for i_kc in range(0, tl.cdiv(K, KC)):
        a = tl.load(a_ptr).reshape((KC, MR))
        b = tl.load(b_ptr).reshape((KC, NR))
        c += tl.dot(a.T, b, out_dtype=out_dtype)
        a_ptr = tl.advance(a_ptr, (0, KC, 0))
        b_ptr = tl.advance(b_ptr, (0, KC, 0))

    tl.store(c_ptr, c)


class OuterGEMM(GEMMKernelBase):
    def __init__(self, in_dtype: str, out_dtype: str,
                 MR: int, NR: int, KC: int = 256,
                 device: str = "cpu"):
        assert in_dtype in GEMMKernelBase.DTYPE_CONFIG
        assert out_dtype in GEMMKernelBase.DTYPE_CONFIG

        self.in_dtype_name = in_dtype
        self.out_dtype_name = out_dtype
        self.device = device

        self.MR = MR
        self.NR = NR
        self.KC = KC

    def _pack_outer(self, a: torch.Tensor, b: torch.Tensor):
        M, K = a.shape
        K2, N = b.shape
        MR = self.MR
        NR = self.NR

        a_out = a.reshape((M // MR, MR, K)).permute((0, 2, 1))
        b_out = b.reshape((K, N // NR, NR)).permute((1, 0, 2))

        return a_out.contiguous(), b_out.contiguous()

    def prepare(self, m: int, k: int, n: int, should_gen_data: bool = True) -> dict:
        TORCH_IN_DTYPE, TL_IN_DTYPE = GEMMKernelBase.DTYPE_CONFIG[self.in_dtype_name]
        TORCH_OUT_DTYPE, TL_OUT_DTYPE = GEMMKernelBase.DTYPE_CONFIG[self.out_dtype_name]

        M, K, N = m, k, n
        MR, NR, KC = self.MR, self.NR, self.KC

        assert M % MR == 0, f"M ({M}) must be divisible by MR ({MR})"
        assert N % NR == 0, f"N ({N}) must be divisible by NR ({NR})"
        assert K % KC == 0, f"K ({K}) must be divisible by KC ({KC})"

        if should_gen_data:
            torch.manual_seed(0)
            a = torch.randint(0, 128, (M, K), device=self.device).to(TORCH_IN_DTYPE)
            b = torch.randint(0, 128, (K, N), device=self.device).to(TORCH_IN_DTYPE)
            a_packed, b_packed = self._pack_outer(a, b)
        else:
            a = torch.empty((M, K), device=self.device, dtype=TORCH_IN_DTYPE)
            b = torch.empty((K, N), device=self.device, dtype=TORCH_IN_DTYPE)
            a_packed, b_packed = a, b

        out = torch.empty((M, N), device=self.device, dtype=TORCH_OUT_DTYPE)

        return {
            'a_orig': a,
            'b_orig': b,
            'a': a_packed,
            'b': b_packed,
            'out': out,
            'M': M, 'K': K, 'N': N,
            'MR': MR, 'NR': NR, 'KC': KC,
            'torch_in_dtype': TORCH_IN_DTYPE,
            'torch_out_dtype': TORCH_OUT_DTYPE,
            'tl_in_dtype': TL_IN_DTYPE,
            'tl_out_dtype': TL_OUT_DTYPE,
        }

    def run(self, params: dict, repeats: int = 1) -> float:
        a = params['a']
        b = params['b']
        out = params['out']
        M, K, N = params['M'], params['K'], params['N']
        MR, NR, KC = params['MR'], params['NR'], params['KC']
        TL_OUT_DTYPE = params['tl_out_dtype']

        grid = (cdiv(M, MR), cdiv(N, NR), repeats)

        t0 = time.perf_counter()
        matmul_outer_kernel[grid](
            a_ptr_raw=a, b_ptr_raw=b, c_ptr_raw=out,
            M=M, K=K, N=N,
            MR=MR, NR=NR, KC=KC,
            out_dtype=TL_OUT_DTYPE,
        )
        t1 = time.perf_counter()
        return (t1 - t0) / repeats

    def verify(self, params: dict) -> bool:
        a = params['a_orig']
        b = params['b_orig']
        out = params['out']
        ref = torch.matmul(a.to(params['torch_out_dtype']), b.to(params['torch_out_dtype']))
        return bool(torch.allclose(out, ref, rtol=1e-5, atol=1e-5))

    def get_name(self) -> str:
        return f"{self.__class__.__name__}_{self.in_dtype_name}_{self.out_dtype_name}_{self.MR}x{self.NR}"

    def expected_cycles(self, m: int, k: int, n: int) -> int:
        total_mac = m * k * n
        mac_per_cycle = 256 // GEMMKernelBase.BITWIDTH[self.out_dtype_name]
        return total_mac // mac_per_cycle
