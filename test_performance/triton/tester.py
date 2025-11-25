import os
import time
import torch
import csv

from kernels.inner import InnerGEMM
from kernels.outer import OuterGEMM


# ------------------------------
# 正确性测试
# ------------------------------
def correctness_test(kernel, test_shapes, device="cpu"):
    print(f"\n===== Correctness Test: {kernel.get_name()} =====")

    for (m, k, n) in test_shapes:
        print(f"Testing correctness for shape (m={m}, k={k}, n={n})...")

        params = kernel.prepare(m, k, n)
        kernel.run(params)
        ok = kernel.verify(params)
        if not ok:
            print(f"❌ Correctness FAILED for {kernel.get_name()} ({m}x{k}x{n})")
            return False
    print(f"✔ Correctness passed for {kernel.get_name()}")
    return True


# ------------------------------
# 性能测试
# ------------------------------
def performance_test(kernel, test_shapes, device="cpu"):
    print(f"\n===== Performance Test: {kernel.get_name()} =====")

    os.makedirs("results", exist_ok=True)
    csv_file = os.path.join("results", f"{kernel.get_name()}.csv")
    
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        for (m, k, n) in test_shapes:
            repeats = max(1, int(2**36 / (m*k*n)))
            params = kernel.prepare(m, k, n, should_gen_data=True)
            elapsed = kernel.run(params, repeats=repeats)
            exp_cycles = kernel.expected_cycles(m, k, n)
            perf_ratio = elapsed * 1.6e9 / exp_cycles

            print(f"(m={m}, k={k}, n={n}) | repeats={repeats} | perf_ratio={perf_ratio:.4f}")
            writer.writerow([f"${m} \\times {k} \\times {n}$", f"{perf_ratio:.4f}"])

    print(f"CSV saved: {csv_file}")


def main():
    KERNELS = [
        InnerGEMM(in_dtype="i8", out_dtype="i16"),
        OuterGEMM(in_dtype="i8", out_dtype="i16"),
    ]

    correctness_shapes = [
        (48, 256, 32),
    ]

    performance_shapes = [
        (32, 256, 32),
        (32, 2048, 32),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ]

    for kernel in KERNELS:
        passed = correctness_test(kernel, correctness_shapes)
        if not passed:
            continue
        performance_test(kernel, performance_shapes)


if __name__ == "__main__":
    main()
