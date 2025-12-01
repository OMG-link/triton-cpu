# k1_cache_fix_tool.py
import time
import torch
import triton
import triton.language as tl

###################################
# Simplified version of OuterGEMM #
###################################

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


class OuterGEMM:
    def __init__(self):
        self.device = 'cpu'

        self.MR = 12
        self.NR = 32
        self.KC = 32

    def prepare(self, m: int, k: int, n: int) -> dict:
        TORCH_IN_DTYPE, TL_IN_DTYPE = (torch.int8, tl.int8)
        TORCH_OUT_DTYPE, TL_OUT_DTYPE = (torch.int16, tl.int16)

        M, K, N = m, k, n
        MR, NR, KC = self.MR, self.NR, self.KC

        assert M % MR == 0, f"M ({M}) must be divisible by MR ({MR})"
        assert N % NR == 0, f"N ({N}) must be divisible by NR ({NR})"
        assert K % KC == 0, f"K ({K}) must be divisible by KC ({KC})"

        a = torch.randint(0, 128, (M, K), device=self.device).to(TORCH_IN_DTYPE)
        b = torch.randint(0, 128, (K, N), device=self.device).to(TORCH_IN_DTYPE)
        out = torch.empty((M, N), device=self.device, dtype=TORCH_OUT_DTYPE)

        return {
            'a': a,
            'b': b,
            'out': out,
            'M': M, 'K': K, 'N': N,
            'MR': MR, 'NR': NR, 'KC': KC,
        }

    def run(self, params: dict, repeats: int = 1) -> float:
        a = params['a']
        b = params['b']
        out = params['out']
        M, K, N = params['M'], params['K'], params['N']
        MR, NR, KC = params['MR'], params['NR'], params['KC']

        grid = (M // MR, N // NR)

        t0 = time.perf_counter()
        matmul_outer_kernel[grid](
            a_ptr_raw=a, b_ptr_raw=b, c_ptr_raw=out,
            M=M, K=K, N=N,
            MR=MR, NR=NR, KC=KC,
            out_dtype=tl.int16,
            n_kernel_repeat=repeats,
        )
        t1 = time.perf_counter()
        return (t1 - t0) / repeats

    def expected_cycles(self, m: int, k: int, n: int) -> int:
        total_mac = m * k * n
        mac_per_cycle = 256 // 16
        return total_mac // mac_per_cycle


#############################
# Cache state detect helper #
#############################

def is_cache_corrupted():
    instance = OuterGEMM()
    m, k, n = 192, 2048, 2048
    params = instance.prepare(m, k, n)
    # Warmup
    # FIXME:
    # We should repeat only one time.
    # Currently, we do this 10 times, because 'repeats' are considered as part of kernel hash.
    # Change it to 1 after resolving the kernel-hash issue.
    instance.run(params, repeats=10)
    # Measure
    elapsed = instance.run(params, repeats=10)
    expected_cycles = instance.expected_cycles(m, k, n)
    perf_ratio = elapsed * 1.6e9 / expected_cycles

    if 1.2 <= perf_ratio <= 1.4:
        # Cache is in good state.
        return False
    elif 1.4 < perf_ratio <= 1.6:
        # Cache is in corrupted state.
        return True
    else:
        # Unstable performance detected, possibly due to heavy system load.
        raise RuntimeError(f"Unstable performance detected: perf_ratio={perf_ratio:.4f}")
    
def fix_cache_state():
    _ = torch.empty((1024, 1024), dtype=torch.int8)

def fix_cache_if_corrupted():
    # Actually, running 'is_cache_corrupted' itself fixes corrupted state in most cases.
    # We are not even sure if 'fix_cache_state' actually works.
    # What's worse, we even see cases that we didn't detect any problem here, but perf_ratio
    # of following run shows the cache is corrupted.
    # MAD (X.X)
    if is_cache_corrupted():
        print("⚠️ Cache is in corrupted state. Fixing...")
        fix_cache_state()
        if is_cache_corrupted():
            raise RuntimeError("Failed to fix cache state.")
        else:
            print("✅ Cache state fixed.")
    else:
        pass
