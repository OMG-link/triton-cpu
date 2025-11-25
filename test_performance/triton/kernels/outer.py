# outer.py
import time
import torch
import triton
import triton.language as tl

from gemm_base import GEMMKernelBase

# blocking constants
MR_OUT_VAL = 16
NR_OUT_VAL = 16
KC_OUT_VAL = 256

def cdiv(a, b):
    return (a + b - 1) // b

# Triton outer-product style kernel
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


class OuterGEMM(GEMMKernelBase):
    """
    Outer-product GEMM 接口实现（需要 packed A/B）。
    """

    def __init__(self, in_dtype: str, out_dtype: str, device: str = "cpu"):
        assert in_dtype in GEMMKernelBase.DTYPE_CONFIG and out_dtype in GEMMKernelBase.DTYPE_CONFIG
        self.in_dtype_name = in_dtype
        self.out_dtype_name = out_dtype
        self.device = device

    def _pack_outer(self, a: torch.Tensor, b: torch.Tensor):
        """
        Pack A 和 B 为 kernel 期望的形状：
          A_out: (M//MR_OUT, K, MR_OUT) with blocks A_out[i] = A[i*MR:(i+1)*MR, :].T
          B_out: (N//NR_OUT, K, NR_OUT) with blocks B_out[j] = B[:, j*NR:(j+1)*NR].contiguous()
        """
        M, K = a.shape
        K2, N = b.shape

        a_out = torch.empty((M // MR_OUT_VAL, K, MR_OUT_VAL), dtype=a.dtype, device=a.device)
        for i in range(M // MR_OUT_VAL):
            block = a[i * MR_OUT_VAL:(i + 1) * MR_OUT_VAL, :].contiguous().T  # (K,MR)
            a_out[i, :, :] = block

        b_out = torch.empty((N // NR_OUT_VAL, K, NR_OUT_VAL), dtype=b.dtype, device=b.device)
        for j in range(N // NR_OUT_VAL):
            block = b[:, j * NR_OUT_VAL:(j + 1) * NR_OUT_VAL].contiguous()  # (K,NR)
            b_out[j, :, :] = block

        return a_out.contiguous(), b_out.contiguous()

    def prepare(self, m: int, k: int, n: int, should_gen_data: bool = True) -> dict:
        TORCH_IN_DTYPE, TL_IN_DTYPE = GEMMKernelBase.DTYPE_CONFIG[self.in_dtype_name]
        TORCH_OUT_DTYPE, TL_OUT_DTYPE = GEMMKernelBase.DTYPE_CONFIG[self.out_dtype_name]

        M, K, N = m, k, n
        assert M % MR_OUT_VAL == 0, f"M ({M}) must be divisible by MR_OUT ({MR_OUT_VAL})"
        assert N % NR_OUT_VAL == 0, f"N ({N}) must be divisible by NR_OUT ({NR_OUT_VAL})"
        assert K % KC_OUT_VAL == 0, f"K ({K}) must be divisible by KC_OUT ({KC_OUT_VAL})"

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

        params = {
            'a_orig': a,
            'b_orig': b,
            'a': a_packed,
            'b': b_packed,
            'out': out,
            'M': M, 'K': K, 'N': N,
            'torch_in_dtype': TORCH_IN_DTYPE,
            'torch_out_dtype': TORCH_OUT_DTYPE,
            'tl_in_dtype': TL_IN_DTYPE,
            'tl_out_dtype': TL_OUT_DTYPE,
        }
        return params

    def run(self, params: dict, repeats: int = 1) -> float:
        global TORCH_IN_DTYPE, TORCH_OUT_DTYPE, TL_OUT_DTYPE, TL_IN_DTYPE
        TORCH_IN_DTYPE = params['torch_in_dtype']
        TORCH_OUT_DTYPE = params['torch_out_dtype']
        TL_OUT_DTYPE = params['tl_out_dtype']
        TL_IN_DTYPE = params['tl_in_dtype']

        triton.runtime.driver.set_active_to_cpu()

        a = params['a']
        b = params['b']
        out = params['out']
        M = params['M']; K = params['K']; N = params['N']
        grid = (cdiv(M, MR_OUT_VAL), cdiv(N, NR_OUT_VAL), repeats)

        t0 = time.perf_counter()
        matmul_outer_kernel[grid](a_ptr_raw=a, b_ptr_raw=b, c_ptr_raw=out, M=M, K=K, N=N)
        t1 = time.perf_counter()
        return (t1 - t0) / repeats

    def verify(self, params: dict, rtol: float = 1e-5, atol: float = 1e-5) -> bool:
        a = params['a_orig']
        b = params['b_orig']
        out = params['out']
        ref = torch.matmul(a.to(params['torch_out_dtype']), b.to(params['torch_out_dtype']))
        ok = torch.allclose(out, ref, rtol=rtol, atol=atol)
        return bool(ok)

    def get_name(self) -> int:
        return f"{self.__class__.__name__}_{self.in_dtype_name}_{self.out_dtype_name}"

    def expected_cycles(self, m: int, k: int, n: int) -> int:
        total_mac = m * k * n
        mac_per_cycle = 256 // GEMMKernelBase.BITWIDTH[self.out_dtype_name]
        total_cycles = total_mac // mac_per_cycle
        return total_cycles
