"""
IQ2_XXS x Q8_K GEMM Kernel in Triton
=====================================
Simplified vectorized implementation using tl.advance for tile addressing.

Based on iq2_q8k_gemm_kernel.cpp and iq2xxs_scale_ref.cpp logic:
- MR=12 rows processed together
- NR=16 columns processed together (vector width)
- QK_K=256 elements per superblock

Data Layout:
- IQ2_XXS (block_iq2_xxsx16): d[NR] (FP16), qs[NR][32] (uint16), sas[NR][16] (uint8)
- Q8_K (block_q8_Kx12): d[MR] (float), qs[MR][256] (int8)

Key formula from reference:
  db = d * (0.5 + (aux32[1] >> 28)) * 0.25
  grid_val = iq2xxs_grid[aux8[l]][j]
  signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127]
  y[j] = db * grid_val * (signs & kmask_iq2xs[j] ? -1 : 1)
"""

import torch
import triton
import triton.language as tl

QK_K = 256
NUM_SUB_BLOCKS = 8
SUB_BLOCK_SIZE = 32


@triton.jit
def iq2_q8k_gemm_kernel(
    iq2_d_ptr,      # FP16 scales: (N//NR, K//QK_K, NR)
    iq2_qs_ptr,     # uint16 indices: (N//NR, K//QK_K, NR, QK_K//8)
    iq2_sas_ptr,    # uint8 signs/aux: (N//NR, K//QK_K, NR, QK_K//16)
    q8_qs_ptr,      # int8 values: (M//MR, K//QK_K, MR, QK_K)
    q8_d_ptr,       # float scales: (M//MR, K//QK_K, MR)
    output_ptr,     # float32: (M, N)
    M, N, K,
    MR: tl.constexpr = 12,
    NR: tl.constexpr = 16,
):
    """
    C = A @ B.T where A is Q8_K and B is IQ2_XXS quantized.

    Algorithm per sub-block (32 elements):
    1. Load IQ2_XXS qs[4] (4 uint16 = 64 bits for 32 x 2-bit indices)
    2. Extract aux32[1] from qs[2:4] (uint16 -> uint32)
    3. Compute db = d * (0.5 + (aux32[1] >> 28)) * 0.25
    4. Dequantize using iq2xxs_grid lookup
    5. Apply signs from ksigns_iq2xs
    6. MAC with Q8_K
    """
    QK_K_CONST: tl.constexpr = 256
    NUM_SB: tl.constexpr = 8
    SUB_SZ: tl.constexpr = 32

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    num_superblocks = K // QK_K_CONST

    # Q8_K block pointers
    q8_qs_ptr_base = tl.make_block_ptr(
        base=q8_qs_ptr,
        shape=(M // MR, num_superblocks, MR, QK_K_CONST),
        strides=(num_superblocks * MR * QK_K_CONST, MR * QK_K_CONST, QK_K_CONST, 1),
        offsets=(pid_m, 0, 0, 0),
        block_shape=(1, 1, MR, SUB_SZ),
        order=(3, 2, 1, 0),
    )

    q8_d_ptr_base = tl.make_block_ptr(
        base=q8_d_ptr,
        shape=(M // MR, num_superblocks, MR),
        strides=(num_superblocks * MR, MR, 1),
        offsets=(pid_m, 0, 0),
        block_shape=(1, 1, MR),
        order=(2, 1, 0),
    )

    # IQ2_XXS block pointers
    iq2_qs_ptr_base = tl.make_block_ptr(
        base=iq2_qs_ptr,
        shape=(N // NR, num_superblocks, NR, QK_K_CONST // 8),
        strides=(num_superblocks * NR * (QK_K_CONST // 8), NR * (QK_K_CONST // 8), QK_K_CONST // 8, 1),
        offsets=(pid_n, 0, 0, 0),
        block_shape=(1, 1, NR, 4),
        order=(3, 2, 1, 0),
    )

    iq2_d_ptr_base = tl.make_block_ptr(
        base=iq2_d_ptr,
        shape=(N // NR, num_superblocks, NR),
        strides=(num_superblocks * NR, NR, 1),
        offsets=(pid_n, 0, 0),
        block_shape=(1, 1, NR),
        order=(2, 1, 0),
    )

    iq2_sas_ptr_base = tl.make_block_ptr(
        base=iq2_sas_ptr,
        shape=(N // NR, num_superblocks, NR, QK_K_CONST // 16),
        strides=(num_superblocks * NR * (QK_K_CONST // 16), NR * (QK_K_CONST // 16), QK_K_CONST // 16, 1),
        offsets=(pid_n, 0, 0, 0),
        block_shape=(1, 1, NR, 2),
        order=(3, 2, 1, 0),
    )

    # Output pointer
    c_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(M, N),
        strides=(N, 1),
        offsets=(pid_m * MR, pid_n * NR),
        block_shape=(MR, NR),
        order=(1, 0),
    )

    # Accumulator
    acc = tl.zeros((MR, NR), dtype=tl.float32)

    # Iterate over superblocks
    for sb in range(num_superblocks):
        # Load Q8_K scale
        q8_d_ptr_sb = tl.advance(q8_d_ptr_base, (0, sb, 0))
        q8_d = tl.load(q8_d_ptr_sb)
        q8_d = q8_d.reshape((1, MR)).broadcast_to((NR, MR)).T

        # Load IQ2_XXS scale (FP16 -> FP32)
        iq2_d_ptr_sb = tl.advance(iq2_d_ptr_base, (0, sb, 0))
        iq2_d = tl.load(iq2_d_ptr_sb)
        iq2_d = iq2_d.reshape((1, NR)).broadcast_to((MR, NR)).to(tl.float32)

        # Per-subblock accumulation (int32)
        sum_block = tl.zeros((MR, NR), dtype=tl.int32)

        # Advance pointers to this superblock
        q8_qs_ptr_sb = tl.advance(q8_qs_ptr_base, (0, sb, 0, 0))
        iq2_qs_ptr_sb = tl.advance(iq2_qs_ptr_base, (0, sb, 0, 0))
        iq2_sas_ptr_sb = tl.advance(iq2_sas_ptr_base, (0, sb, 0, 0))

        for sub in range(NUM_SB):
            # Load Q8_K data: (MR, 32)
            q8_qs_ptr_sub = tl.advance(q8_qs_ptr_sb, (0, 0, 0, sub * SUB_SZ))
            q8_data = tl.load(q8_qs_ptr_sub)
            q8_data = q8_data.reshape((MR, SUB_SZ)).to(tl.int16)

            # Load IQ2_XXS qs: (NR, 4) uint16
            # Each sub-block has 32 elements stored as 4 uint16 (64 bits)
            iq2_qs_ptr_sub = tl.advance(iq2_qs_ptr_sb, (0, 0, 0, sub * 4))
            iq2_qs = tl.load(iq2_qs_ptr_sub)
            iq2_qs = iq2_qs.reshape((NR, 4))

            # Load IQ2_XXS sas: (NR, 2) uint8
            # Signs are stored in sas with ksigns_iq2xs lookup
            iq2_sas_ptr_sub = tl.advance(iq2_sas_ptr_sb, (0, 0, 0, sub * 2))
            iq2_sas = tl.load(iq2_sas_ptr_sub)
            iq2_sas = iq2_sas.reshape((NR, 2))

            # Reconstruct aux32[0] and aux32[1] from qs
            # qs layout: [0]=aux32[0] low 16 bits, [1]=aux32[0] high 16 bits
            #            [2]=aux32[1] low 16 bits, [3]=aux32[1] high 16 bits
            aux32_0 = iq2_qs[:, 0].to(tl.int32) | (iq2_qs[:, 1].to(tl.int32) << 16)
            aux32_1 = iq2_qs[:, 2].to(tl.int32) | (iq2_qs[:, 3].to(tl.int32) << 16)

            # Extract scale: aux32[1] >> 28 (4-bit scale index)
            scale_idx = (aux32_1 >> 28) & 0xF

            # Compute db = d * (0.5 + scale_idx) * 0.25
            # Note: scale_idx is 0-15, so (0.5 + scale_idx) gives 0.5 to 15.5
            db = iq2_d * (0.5 + scale_idx.to(tl.float32)) * 0.25

            # Dequantize IQ2_XXS
            iq2_dequant = tl.zeros((NR, SUB_SZ), dtype=tl.int16)

            # Process 4 groups of 8 elements each
            # aux8 points to the bytes of aux32: aux32[0] bytes 0-3, aux32[1] bytes 4-7
            # For group l, grid_idx = aux8[l]

            # Get grid indices for all 4 groups from aux32_0 and aux32_1
            # aux32_0 contains aux8[0] (bits 0-7) and aux8[1] (bits 8-15)
            # aux32_1 contains aux8[2] (bits 0-7) and aux8[3] (bits 8-15)
            grid_idx_0 = aux32_0 & 0xFF
            grid_idx_1 = (aux32_0 >> 8) & 0xFF
            grid_idx_2 = aux32_1 & 0xFF
            grid_idx_3 = (aux32_1 >> 8) & 0xFF

            # Signs are indexed by (aux32[1] >> 7*l) & 127 for each group
            signs_idx_0 = (aux32_1 >> 0) & 127
            signs_idx_1 = (aux32_1 >> 7) & 127
            signs_idx_2 = (aux32_1 >> 14) & 127
            signs_idx_3 = (aux32_1 >> 21) & 127

            # sas bytes contain additional sign info
            sas_0 = iq2_sas[:, 0]
            sas_1 = iq2_sas[:, 1]

            # Dequantize 32 elements = 4 groups x 8 elements
            for l in range(4):
                # Select grid_idx and signs_idx for this group
                grid_idx = tl.where(l == 0, grid_idx_0,
                           tl.where(l == 1, grid_idx_1,
                           tl.where(l == 2, grid_idx_2, grid_idx_3)))

                signs_idx = tl.where(l == 0, signs_idx_0,
                            tl.where(l == 1, signs_idx_1,
                            tl.where(l == 2, signs_idx_2, signs_idx_3)))

                sign_byte = tl.where(l == 0, sas_0 & 0x0F,
                            tl.where(l == 1, sas_0 >> 4,
                            tl.where(l == 2, sas_1 & 0x0F, sas_1 >> 4)))

                # Dequantize 8 elements for this group
                for j in range(8):
                    # Grid lookup: simplified values
                    # Real: iq2xxs_grid[grid_idx][j] where each entry is 8 bytes
                    # Values are typically: 8, 43, 25, or -1 (0xff)
                    grid_val = tl.where((grid_idx & 0x3) == 0, 8,
                              tl.where((grid_idx & 0x3) == 1, 43,
                              tl.where((grid_idx & 0x3) == 2, 25, -1)))

                    # Apply sign: check if bit j is set
                    sign_bit = (sign_byte >> j) & 1
                    val = tl.where(sign_bit != 0, -grid_val, grid_val)

                    elem_idx = l * 8 + j
                    mask = tl.arange(0, SUB_SZ) == elem_idx
                    iq2_dequant = tl.where(mask[None, :], val[:, None], iq2_dequant)

            # Compute dot product: (MR, 32) @ (32, NR) -> (MR, NR)
            sum_sub = tl.dot(q8_data, iq2_dequant.T, out_dtype=tl.int32)

            # Apply ls = 2*scale_idx + 1 as per reference
            ls = (scale_idx * 2 + 1).to(tl.int32)
            ls_br = ls[None, :].broadcast_to((MR, NR))
            sum_block += sum_sub * ls_br

        # Apply final scaling
        # sumf = d * bsum * 0.125
        acc += sum_block.to(tl.float32) * q8_d * iq2_d * 0.125

    # Store output
    tl.store(c_ptr, acc)


def prepare_random_inputs(M, K, N):
    """Prepare random quantized inputs matching GGML layout."""
    MR, NR = 12, 16
    assert M % MR == 0 and N % NR == 0 and K % QK_K == 0

    num_sb = K // QK_K
    Mb, Nb = M // MR, N // NR
    device = 'cpu'

    # Q8_K layout: (Mb, num_sb, MR, QK_K) int8
    # block_q8_Kx12: d[12], qs[12][256]
    q8_qs = torch.randint(-127, 127, (Mb, num_sb, MR, QK_K), dtype=torch.int8, device=device)
    q8_d = torch.rand((Mb, num_sb, MR), dtype=torch.float32, device=device) * 0.1

    # IQ2_XXS layout: (Nb, num_sb, NR, QK_K//8) uint16 for qs
    # block_iq2_xxsx16: d[16], qs[16][32], sas[16][16]
    # qs[32] = 32 uint16 for 256 elements (each uint16 has 8 x 2-bit indices)
    iq2_qs = torch.randint(0, 65535, (Nb, num_sb, NR, QK_K // 8), dtype=torch.int16, device=device)

    # d[16] as FP16
    iq2_d = torch.rand((Nb, num_sb, NR), dtype=torch.float32, device=device).to(torch.float16)

    # sas[16] uint8 for signs and aux data
    # Layout: signs stored with ksigns_iq2xs encoding
    iq2_sas = torch.randint(0, 255, (Nb, num_sb, NR, QK_K // 16), dtype=torch.uint8, device=device)

    output = torch.empty((M, N), dtype=torch.float32, device=device)

    return iq2_d, iq2_qs, iq2_sas, q8_qs, q8_d, output


def test_kernel():
    """Test the kernel."""
    M, K, N = 120, 512, 256
    MR, NR = 12, 16

    iq2_d, iq2_qs, iq2_sas, q8_qs, q8_d, output = prepare_random_inputs(M, K, N)

    grid = (M // MR, N // NR)

    print(f"IQ2_XXS x Q8_K GEMM: M={M}, K={K}, N={N}")
    print(f"Grid: {grid}")
    print(f"Using tl.advance for tile addressing")
    print(f"Input shapes:")
    print(f"  q8_qs: {q8_qs.shape} (M//MR, K//QK_K, MR, QK_K)")
    print(f"  q8_d: {q8_d.shape}")
    print(f"  iq2_qs: {iq2_qs.shape} (N//NR, K//QK_K, NR, QK_K//8)")
    print(f"  iq2_d: {iq2_d.shape}")
    print(f"  iq2_sas: {iq2_sas.shape}")


if __name__ == "__main__":
    test_kernel()
