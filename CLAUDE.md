# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Triton-CPU is an experimental CPU backend for the Triton language and compiler. It extends the upstream Triton repository with CPU-specific code generation capabilities. The CPU backend is maintained out-of-tree in `third_party/cpu/`.

## Build Commands

### Prerequisites
- LLVM/MLIR must be built from source at the revision specified in `cmake/llvm-hash.txt`

### Building Triton

```bash
# Standard build
pip install -e python

# Build with custom LLVM (recommended)
LLVM_BUILD_DIR=/path/to/llvm/build \
LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include \
LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib \
LLVM_SYSPATH=$LLVM_BUILD_DIR \
pip install -e python --no-build-isolation

# Build with clang/lld (faster linking)
TRITON_BUILD_WITH_CLANG_LLD=true pip install -e python --no-build-isolation

# Reduce parallel jobs if running out of memory
MAX_JOBS=4 pip install -e python
```

### Incremental Builds

```bash
# After initial pip install, use ninja for faster rebuilds
make all  # or: ninja -C python/build/<dir>
```

## Running with CPU Backend

Set `TRITON_CPU_BACKEND=1` to use the CPU backend instead of GPU:

```bash
TRITON_CPU_BACKEND=1 python3 tutorials/01-vector-add.py
```

## Testing

```bash
# Run lit tests (MLIR-level tests)
make test-lit

# Run C++ unit tests
make test-cpp

# Run Python unit tests (requires GPU by default)
make test-unit

# Run tests without GPU
make test-nogpu

# Run all tests
make test

# Run interpreter mode (no GPU required)
TRITON_INTERPRET=1 pytest <test_file> --device=cpu
```

## Architecture

### Compiler Pipeline

1. **Frontend** (`python/triton/language/`): Python DSL for writing kernels
2. **Triton IR** (`lib/Dialect/Triton/`): Initial IR representing the kernel
3. **TritonGPU IR** (`lib/Dialect/TritonGPU/`): GPU-specific transformations (shared memory, layouts)
4. **TritonCPU IR** (`third_party/cpu/`): CPU-specific lowering and optimizations
5. **LLVM IR** (`lib/Target/LLVMIR/`): Final code generation

### Key Directories

- `lib/Dialect/`: MLIR dialect definitions and transformations
  - `Triton/`: Core Triton dialect (tensor operations)
  - `TritonGPU/`: GPU-specific dialect (layouts, pipelining)
  - `TritonCPU/`: CPU-specific dialect
- `lib/Conversion/`: Passes to convert between dialects
- `lib/Analysis/`: Compiler analyses
- `python/triton/`: Python package
  - `language/`: Frontend language constructs
  - `compiler/`: Compiler interfaces
  - `backends/`: Backend implementations
- `third_party/cpu/`: CPU backend implementation
  - `lib/TritonToTritonCPU/`: Lowering from Triton to TritonCPU
  - `lib/TritonCPUTransforms/`: CPU-specific optimizations
  - `lib/TritonCPUToLLVM/`: Lowering to LLVM IR
- `test/`: Lit tests organized by dialect
- `python/test/`: Python unit tests

### Backend Implementation

Each backend (NVIDIA, AMD, CPU) provides:
1. A `driver.py` for runtime execution
2. A `compiler.py` for code generation
3. Dialect transformations in `lib/`

The CPU backend at `third_party/cpu/` follows this pattern with additional components for CPU-specific code generation.

## Debug Environment Variables

```bash
# Dump IR at various stages
MLIR_ENABLE_DUMP=1              # Dump IR before every MLIR pass
LLVM_IR_ENABLE_DUMP=1           # Dump IR before every LLVM pass
TRITON_KERNEL_DUMP=1            # Dump IR from each compilation stage
TRITON_DUMP_DIR=./ir-dump       # Directory for dumped IR

# Use interpreter instead of hardware execution
TRITON_INTERPRET=1

# Force recompilation (ignore cache)
TRITON_ALWAYS_COMPILE=1

# Debug specific components
TRITON_LLVM_DEBUG_ONLY="pass-name"  # Limit LLVM debug output
MLIR_ENABLE_TIMING=1                # Time each MLIR pass
```

## Code Style

- C++ code follows LLVM style (`.clang-format` sets `BasedOnStyle: LLVM`)
- Python code follows standard Python conventions
- Pre-commit hooks are configured in `.pre-commit-config.yaml`

## Important Notes

- The CPU backend is work-in-progress
- Changes to core Triton should aim to be upstreamed when possible
- CPU-specific code should live in `third_party/cpu/` or `lib/Dialect/TritonCPU/`