# inner.py
import time
import torch
import triton
import triton.language as tl

from gemm_base import GEMMKernelBase

# blocking constants
MR_IN_VAL = 4
NR_IN_VAL = 4
KC_IN_VAL = 256

# helper
def cdiv(a, b):
    return (a + b - 1) // b

# Triton kernel (inner-product style)
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


class InnerGEMM(GEMMKernelBase):
    """
    Inner-product GEMM 接口实现。
    构造函数接受可选的输入/输出 dtype 名称（如 'fp32', 'fp16', ...）。
    """

    def __init__(self, in_dtype: str, out_dtype: str, device: str = "cpu"):
        assert in_dtype in GEMMKernelBase.DTYPE_CONFIG and out_dtype in GEMMKernelBase.DTYPE_CONFIG
        self.in_dtype_name = in_dtype
        self.out_dtype_name = out_dtype
        self.device = device

    def prepare(self, m: int, k: int, n: int, should_gen_data: bool = True) -> dict:
        """
        返回参数字典：
          {
            'a': a_tensor (M,K) contiguous,
            'b': b_t_tensor (N,K) contiguous (note: b is transposed as kernel expects),
            'out': out_tensor (M,N),
            'M': M, 'K': K, 'N': N,
            'grid': (gM, gN),
            'torch_in_dtype': torch dtype object,
            'torch_out_dtype': torch dtype object,
            'tl_out_dtype': triton dtype object,
          }
        """
        TORCH_IN_DTYPE, TL_IN_DTYPE = GEMMKernelBase.DTYPE_CONFIG[self.in_dtype_name]
        TORCH_OUT_DTYPE, TL_OUT_DTYPE = GEMMKernelBase.DTYPE_CONFIG[self.out_dtype_name]

        M, K, N = m, k, n
        assert M % MR_IN_VAL == 0, f"M ({M}) must be divisible by MR_IN ({MR_IN_VAL})"
        assert N % NR_IN_VAL == 0, f"N ({N}) must be divisible by NR_IN ({NR_IN_VAL})"
        assert K % KC_IN_VAL == 0, f"K ({K}) must be divisible by KC_IN ({KC_IN_VAL})"

        # allocate
        if should_gen_data:
            torch.manual_seed(0)
            a = torch.randint(0, 128, (M, K), device=self.device).to(TORCH_IN_DTYPE).contiguous()
            b = torch.randint(0, 128, (K, N), device=self.device).to(TORCH_IN_DTYPE)
        else:
            a = torch.empty((M, K), device=self.device, dtype=TORCH_IN_DTYPE).contiguous()
            b = torch.empty((K, N), device=self.device, dtype=TORCH_IN_DTYPE)

        b_t = b.t().contiguous()  # kernel expects (N, K)
        out = torch.empty((M, N), device=self.device, dtype=TORCH_OUT_DTYPE)

        params = {
            'a': a,
            'b': b_t,
            'out': out,
            'M': M, 'K': K, 'N': N,
            'torch_in_dtype': TORCH_IN_DTYPE,
            'torch_out_dtype': TORCH_OUT_DTYPE,
            'tl_in_dtype': TL_IN_DTYPE,
            'tl_out_dtype': TL_OUT_DTYPE,
        }
        return params

    def run(self, params: dict, repeats: int = 1) -> float:
        """
        执行 inner kernel。返回平均耗时（秒）。
        """
        # bind dtype global names used inside the Triton kernel
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
        grid = (cdiv(M, MR_IN_VAL), cdiv(N, NR_IN_VAL), repeats)

        # warmup not done here; caller can call run multiple times if desired
        t0 = time.perf_counter()
        matmul_inner_kernel[grid](a_ptr_raw=a, b_ptr_raw=b, c_ptr_raw=out, M=M, K=K, N=N)
        t1 = time.perf_counter()
        return (t1 - t0) / repeats

    def verify(self, params: dict) -> bool:
        a = params['a']
        # original b was (K,N) then we stored b_t in params['b']
        b_t = params['b']
        b = b_t.t().contiguous()
        out = params['out']
        # compute reference
        ref = torch.matmul(a.to(params['torch_out_dtype']), b.to(params['torch_out_dtype']))
        ok = torch.allclose(out, ref, rtol=1e-5, atol=1e-5)
        return bool(ok)

    def get_name(self) -> int:
        return f"{self.__class__.__name__}_{self.in_dtype_name}_{self.out_dtype_name}"

    def expected_cycles(self, m: int, k: int, n: int) -> int:
        total_mac = m * k * n
        mac_per_cycle = 256 // GEMMKernelBase.BITWIDTH[self.out_dtype_name]
        total_cycles = total_mac // mac_per_cycle
        return total_cycles