import os
import sys
import time
import torch
import csv

import k1_cache_fix_tool

from kernels.inner import InnerGEMM
from kernels.outer import OuterGEMM
from kernels.q4k_q8k_gemm import Q4K_Q8K_GEMM
from kernels.transpose import TransposeKernel

def correctness_test(kernel, test_shapes):
    print(f"\n===== Correctness Test: {kernel.get_name()} =====")

    for (m, k, n) in test_shapes:
        print(f"Testing correctness for shape (m={m}, k={k}, n={n})...")

        params = kernel.prepare(m, k, n)
        kernel.run(params)
        if not kernel.verify(params):
            print(f"❌ FAILED for shape {m}x{k}x{n}")
            return False

    print(f"✅ Correctness passed for {kernel.get_name()}")
    return True


def performance_test(kernel, test_shapes):
    print(f"\n===== Performance Test: {kernel.get_name()} =====")

    os.makedirs("results", exist_ok=True)
    csv_file = os.path.join("results", f"{kernel.get_name()}.csv")

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        for (m, k, n) in test_shapes:
            try:
                k1_cache_fix_tool.fix_cache_if_corrupted()
            except RuntimeError as e:
                global IGNORE_CACHE_WARNING
                if 'IGNORE_CACHE_WARNING' not in globals():
                    IGNORE_CACHE_WARNING = False
                print(f"Cache may not be in good state: {e}")
                if not IGNORE_CACHE_WARNING:
                    print("Do you want to continue? (y/n): ", end="")
                    choice = input().strip().lower()
                    if choice != 'y':
                        print("Aborting performance test due to cache state.")
                        sys.exit(1)
                    else:
                        IGNORE_CACHE_WARNING = True

            repeats = max(1, int(2**36 / (m * k * n)))
            params = kernel.prepare(m, k, n)
            # Warmup (repeats must be exacly 'repeats' because this is part of hash key of kernel cache)
            kernel.run(params, repeats=repeats)
            # Measure
            elapsed = kernel.run(params, repeats=repeats)
            exp_cycles = kernel.expected_cycles(m, k, n)
            perf_ratio = elapsed * 1.6e9 / exp_cycles

            print(f"(m={m}, k={k}, n={n}) | repeats={repeats} | perf_ratio={perf_ratio:.4f}")
            writer.writerow([f"${m} \\times {k} \\times {n}$", f"{perf_ratio:.4f}"])

    print(f"CSV saved: {csv_file}")


TEST_SETS = {
    "correctness": [
        (48, 512, 32),
    ],
    "transpose_correctness": [
        (32, 32, 1),
    ],
    "normal_1": [
        (32, 256, 32),
        (32, 2048, 32),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ],
    "q4k_q8k_1": [
        (4, 256, 32),
        (512, 1536, 1536),
        (2048, 2048, 2048),
    ],
    "q4k_q8k_2": [
        (12, 256, 32),
        (480, 1536, 1536),
        (1800, 2048, 2048),
    ],
}

KERNELS = [
    {
        "kernel": InnerGEMM(in_dtype="i8", out_dtype="i16", MR=4, NR=4),
        "correctness": ["correctness"],
        "performance": ["normal_1"],
    },
    {
        "kernel": OuterGEMM(in_dtype="i8", out_dtype="i16", MR=4, NR=32),
        "correctness": ["correctness"],
        "performance": ["normal_1"],
    },
    {
        "kernel": OuterGEMM(in_dtype="i8", out_dtype="i16", MR=8, NR=32),
        "correctness": ["correctness"],
        "performance": ["normal_1"],
    },
    {
        "kernel": OuterGEMM(in_dtype="i8", out_dtype="i16", MR=16, NR=16),
        "correctness": ["correctness"],
        "performance": ["normal_1"],
    },
    {
        "kernel": Q4K_Q8K_GEMM(MR=4, NR=32),
        "correctness": ["correctness"],
        "performance": ["q4k_q8k_1"],
    },
    {
        "kernel": Q4K_Q8K_GEMM(MR=12, NR=32),
        "correctness": ["correctness"],
        "performance": ["q4k_q8k_2"],
    },
    {
        "kernel": TransposeKernel(dtype="i8", MR=16, TM=4, TN=32),
        "correctness": ["transpose_correctness"],
        "performance": [],
    },
]


def expand_test_sets(test_set_names):
    shapes = []
    for name in test_set_names:
        shapes.extend(TEST_SETS[name])
    return shapes

def main():
    for entry in KERNELS:
        kernel = entry["kernel"]

        correctness_shapes = expand_test_sets(entry["correctness"])
        performance_shapes = expand_test_sets(entry["performance"])

        print("\n\n------------------------------")
        print("Testing Kernel:", kernel.get_name())
        print("------------------------------")

        if not correctness_test(kernel, correctness_shapes):
            continue
        performance_test(kernel, performance_shapes)


if __name__ == "__main__":
    main()
