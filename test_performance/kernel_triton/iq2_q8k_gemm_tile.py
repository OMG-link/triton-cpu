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
  grid_val = iq2xxs_grid[aux8[l]][j]
  signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127]
  y[j] = grid_val * (signs & kmask_iq2xs[j] ? -1 : 1)
"""

import torch
import triton
import triton.language as tl

QK_K = 256
NUM_SUB_BLOCKS = 8
SUB_BLOCK_SIZE = 32

# ksigns_iq2xs lookup table: maps 7-bit index to 8-bit sign mask
# Each bit in the result represents the sign of one weight (0=positive, 1=negative)
KSIGNS_IQ2XS = [
    0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15,
    144, 17, 18, 147, 20, 149, 150, 23, 24, 153, 154, 27, 156, 29, 30, 159,
    160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170, 43, 172, 45, 46, 175,
    48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63,
    192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207,
    80, 209, 210, 83, 212, 85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95,
    96, 225, 226, 99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255
]


@triton.jit
def iq2_q8k_gemm_kernel(
    iq2_d_ptr,      # int16 scales: (N//NR, K//QK_K, NR) 
    iq2_qs_ptr,     # uint8 indices: (N//NR, K//QK_K, QK_K/ SUB_BLOCK_SIZE, NR, 4) 
    iq2_sign_ptr,   # uint8 signbit indices: (N//NR, K//QK_K, QK_K/ SUB_BLOCK_SIZE, NR, 4)
    iq2_scale_ptr,      # uint8 scale (N//NR, K//QK_K, QK_K/ SUB_BLOCK_SIZE, NR)
    
    q8_qs_ptr,      # int8 values: (M//MR, K//QK_K, MR, QK_K) 
    q8_d_ptr,       # float scales: (M//MR, K//QK_K, MR) 
    
    ksigns_ptr,     # ksigns_iq2xs lookup table: (128,) uint8 
    iq2xxs_grid_ptr, # iq2xxs_grid for iq2 dequant 
    
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
    3. Dequantize using iq2xxs_grid lookup
    4. Apply signs from ksigns_iq2xs
    5. MAC with Q8_K
    """
    QK_K_CONST: tl.constexpr = 256
    SUB_SZ: tl.constexpr = 32
    NUM_SB: tl.constexpr = QK_K_CONST // SUB_SZ

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    num_superblocks = K // QK_K_CONST

    # Q8_K block pointers
    q8_qs_ptr_base = tl.make_block_ptr(
        base=q8_qs_ptr,
        shape=(M // MR, num_superblocks, NUM_SB, SUB_SZ, MR),
        strides=(K * MR, QK_K_CONST * MR, MR * SUB_SZ, MR, 1),
        offsets=(pid_m, 0, 0, 0, 0),
        block_shape=(1, 1, 1, SUB_SZ, MR),
        order=(4, 3, 2, 1, 0),
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
        shape=(N // NR, num_superblocks, NUM_SB, NR, SUB_SZ // 8), 
        strides=(K * NR // 8,  NR * QK_K_CONST // 8 , NR * (SUB_SZ // 8), SUB_SZ // 8, 1), 
        offsets=(pid_n, 0, 0, 0, 0), 
        block_shape=(1, 1, 1, NR, 4), 
        order=(4, 3, 2, 1, 0), 
    ) 

    # IQ2_XXS block pointers
    iq2_sign_ptr_base = tl.make_block_ptr(
        base=iq2_sign_ptr, 
        shape=(N // NR, num_superblocks, NUM_SB, NR, SUB_SZ // 8), 
        strides=(K * NR // 8,  NR * QK_K_CONST // 8 , NR * (SUB_SZ // 8), SUB_SZ // 8, 1), 
        offsets=(pid_n, 0, 0, 0, 0), 
        block_shape=(1, 1, 1, NR, 4), 
        order=(4, 3, 2, 1, 0), 
    ) 

    iq2_scale_ptr_base = tl.make_block_ptr(
        base=iq2_scale_ptr, 
        shape=(N // NR, num_superblocks, NUM_SB, NR), 
        strides=(num_superblocks * NUM_SB * NR, NR * NUM_SB, NR, 1), 
        offsets=(pid_n, 0, 0, 0), 
        block_shape=(1, 1, 1, NR), 
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

        sum_block = tl.zeros((MR, NR), dtype=tl.int32) 

        for sub in range(NUM_SB):
            # Load Q8_K data: (MR, 32)
            q8_qs_ptr_sub = tl.advance(q8_qs_ptr_base, (0, sb, sub, 0, 0))
            q8_data = tl.load(q8_qs_ptr_sub).reshape((SUB_SZ, MR))  # [32, MR]@int8

            # Load IQ2_XXS qs: (NR, 4) uint8
            # Each sub-block has 32 elements stored as 4 uint16 (64 bits)
            iq2_qs_ptr_sub = tl.advance(iq2_qs_ptr_base, (0, sb, sub, 0, 0)) 
            iq2_qs_sub = tl.load(iq2_qs_ptr_sub).reshape((NR, 4)) # [NR, 4]@uint8
            iq2_8ele_lookup_dequant_val = tl.load(iq2xxs_grid_ptr + iq2_qs_sub) #[NR, 4]@uint64
            #[NR, 32]@uint64
            iq2_dequant_broadcast = iq2_8ele_lookup_dequant_val.reshape((NR, 4, 1)).broadcast_to((NR, 4, 8)).reshape(NR, 32)
            # Extract values using shifts instead of masks
            # For each position i (0-31), extract the appropriate byte
            arange_32 = tl.arange(0, 32)
            byte_idx = arange_32 // 8  # 0-3 for the 4 uint64 values
            bit_offset = (arange_32 % 8) * 8  # 0, 8, 16, ... 56

            # Reshape for broadcasting: iq2_dequant_broadcast is (NR, 32) of uint64
            # We need to shift right by bit_offset and mask with 0xFF
            iq2_dequant = ((iq2_dequant_broadcast >> bit_offset[None, :]) & 0xFF).cast(tl.int8)

            # (NR, 4) uint8
            iq2_sign_ptr_sub = tl.advance(iq2_sign_ptr_base,  (0, sb, sub, 0, 0))
            iq2_sign = tl.load(iq2_sign_ptr_sub).reshape((NR, 4)) # [NR, 4]@uint8
            iq2_sign_loopup_dequant_val = tl.load(ksigns_ptr + iq2_sign) # [NR, 4]@uint8
            iq2_sign_dequant_broadcast =  iq2_sign_loopup_dequant_val.reshape((NR, 4, 1)).broadcast_to((NR, 4, 8)).reshape(NR, 32)  # [NR, 32]@uint8

            # Extract sign bits: for each position i (0-31), check the appropriate bit
            sign_bit_idx = arange_32 % 8  # 0-7 for the 8 bits in each sign byte
            iq2_sign_dequant = ((iq2_sign_dequant_broadcast >> sign_bit_idx[None, :]) & 1).cast(tl.int1)

            iq2_dequant_sign = tl.where(iq2_sign_dequant, -iq2_dequant, iq2_dequant) # [NR, 32]@int8
            
            iq2_dequant_sign = tl.trans(iq2_dequant_sign)  # [32, NR]@int8 

            # Compute dot product: (MR, 32) @ (32, NR) -> (MR, NR)
            sum_sub = tl.dot(q8_data.T, iq2_dequant_sign, out_dtype=tl.int32) 

            iq2_scale_ptr_sub = tl.advance(iq2_scale_ptr_base, (0, sb, sub, 0))
            iq2_scale = tl.load(iq2_scale_ptr_sub)  # [NR]
            iq2_scale = iq2_scale.reshape((1, NR)).broadcast_to((MR, NR))
            
            sum_block += sum_sub * iq2_scale


        # Apply final scaling
        # sumf = d * bsum * 0.125
                # Load Q8_K scale
        q8_d_ptr_sb = tl.advance(q8_d_ptr_base, (0, sb, 0))
        q8_d = tl.load(q8_d_ptr_sb).reshape((1, MR))  # [1, MR] float32

        # Load IQ2_XXS scale (FP16 -> FP32)
        iq2_d_ptr_sb = tl.advance(iq2_d_ptr_base, (0, sb, 0))
        iq2_d = tl.load(iq2_d_ptr_sb).reshape((1, NR)).to(tl.float32)  # [1, NR] float32

        q8d_iq2d = tl.dot(q8_d.T, iq2_d, out_dtype=tl.float32)

        acc += sum_block.to(tl.float32) * q8d_iq2d

    acc = acc * 0.125
    # Store output
    tl.store(c_ptr, acc)


def prepare_random_inputs(M, K, N, MR=12, NR=16):
    """
    Prepare random quantized inputs matching the kernel interface.

    Kernel Interface:
    - iq2_d_ptr:     (N//NR, K//QK_K, NR) - FP16 scales
    - iq2_qs_ptr:    (N//NR, K//QK_K, NUM_SB, NR, 4) - uint8 indices (4 uint16 = 64 bits for 32 x 2-bit)
    - iq2_sign_ptr:  (N//NR, K//QK_K, NUM_SB, NR, 4) - uint8 sign indices
    - iq2_scale_ptr: (N//NR, K//QK_K, NUM_SB, NR) - uint8 scales per sub-block

    - q8_qs_ptr:     (M//MR, K//QK_K, MR, QK_K) - int8 values
    - q8_d_ptr:      (M//MR, K//QK_K, MR) - float scales

    - ksigns_ptr:    (128,) - uint8 lookup table
    - iq2xxs_grid_ptr: (256,) or appropriate size - lookup table for dequant
    """
    assert M % MR == 0 and N % NR == 0 and K % QK_K == 0

    num_superblocks = K // QK_K
    NUM_SB = QK_K // SUB_BLOCK_SIZE  # 8 sub-blocks per superblock
    Mb, Nb = M // MR, N // NR
    device = 'cpu'

    # Q8_K layout: (Mb, num_superblocks, MR, QK_K) int8
    q8_qs = torch.randint(-127, 127, (Mb, num_superblocks, MR, QK_K), dtype=torch.int8, device=device)
    q8_d = torch.rand((Mb, num_superblocks, MR), dtype=torch.float32, device=device) * 0.1

    # IQ2_XXS layout:
    # iq2_d: (Nb, num_superblocks, NR) - FP16 scales
    iq2_d = torch.rand((Nb, num_superblocks, NR), dtype=torch.float32, device=device).to(torch.float16)

    # iq2_qs: (Nb, num_superblocks, NUM_SB, NR, 4) - uint8
    # Each sub-block has 32 elements stored as 4 uint64 values (64 bits each, 8 x 2-bit indices)
    # Stored as 4 uint8 values for simplicity in this mock
    iq2_qs = torch.randint(0, 255, (Nb, num_superblocks, NUM_SB, NR, 4), dtype=torch.uint8, device=device)

    # iq2_sign: (Nb, num_superblocks, NUM_SB, NR, 4) - uint8
    # Sign indices for ksigns_iq2xs lookup (4 values per sub-block)
    iq2_sign = torch.randint(0, 127, (Nb, num_superblocks, NUM_SB, NR, 4), dtype=torch.uint8, device=device)

    # iq2_scale: (Nb, num_superblocks, NUM_SB, NR) - uint8
    # Per sub-block scales
    iq2_scale = torch.randint(1, 16, (Nb, num_superblocks, NUM_SB, NR), dtype=torch.uint8, device=device)

    # ksigns_iq2xs lookup table: (128,) uint8
    ksigns = torch.tensor(KSIGNS_IQ2XS, dtype=torch.uint8, device=device)

    # iq2xxs_grid lookup table: (256,) - maps 2-bit values to actual weights
    # In real implementation this would map to {-1, 0, 1} or similar
    iq2xxs_grid = torch.randint(-1, 2, (256,), dtype=torch.int8, device=device)

    output = torch.empty((M, N), dtype=torch.float32, device=device)

    return iq2_d, iq2_qs, iq2_sign, iq2_scale, q8_qs, q8_d, ksigns, iq2xxs_grid, output


def test_kernel():
    """Test the IQ2_XXS x Q8_K GEMM kernel."""
    M, K, N = 120, 512, 256
    MR, NR = 12, 16

    iq2_d, iq2_qs, iq2_sign, iq2_scale, q8_qs, q8_d, ksigns, iq2xxs_grid, output = prepare_random_inputs(M, K, N, MR, NR)

    num_superblocks = K // QK_K
    NUM_SB = QK_K // SUB_BLOCK_SIZE

    grid = (M // MR, N // NR)

    print(f"IQ2_XXS x Q8_K GEMM: M={M}, K={K}, N={N}")
    print(f"Grid: {grid}")
    print(f"MR={MR}, NR={NR}, QK_K={QK_K}, NUM_SB={NUM_SB}")
    print(f"\nInput shapes:")
    print(f"  Q8_K:")
    print(f"    q8_qs: {q8_qs.shape} (M//MR, K//QK_K, MR, QK_K) int8")
    print(f"    q8_d: {q8_d.shape} (M//MR, K//QK_K, MR) float32")
    print(f"  IQ2_XXS:")
    print(f"    iq2_d: {iq2_d.shape} (N//NR, K//QK_K, NR) float16")
    print(f"    iq2_qs: {iq2_qs.shape} (N//NR, K//QK_K, NUM_SB, NR, 4) uint8")
    print(f"    iq2_sign: {iq2_sign.shape} (N//NR, K//QK_K, NUM_SB, NR, 4) uint8")
    print(f"    iq2_scale: {iq2_scale.shape} (N//NR, K//QK_K, NUM_SB, NR) uint8")
    print(f"  Lookup tables:")
    print(f"    ksigns: {ksigns.shape} uint8")
    print(f"    iq2xxs_grid: {iq2xxs_grid.shape} int8")

    # Launch kernel
    print(f"\nLaunching kernel...")
    try:
        iq2_q8k_gemm_kernel[grid](
            iq2_d, iq2_qs, iq2_sign, iq2_scale,
            q8_qs, q8_d,
            ksigns, iq2xxs_grid,
            output,
            M, N, K,
            MR=MR, NR=NR,
        )
        print(f"Kernel execution completed!")
        print(f"Output shape: {output.shape}")
        print(f"Output sample (first 5x5):\n{output[:5, :5]}")
    except Exception as e:
        print(f"Kernel execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_kernel()
