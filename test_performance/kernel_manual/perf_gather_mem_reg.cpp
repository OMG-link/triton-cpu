#include <stdio.h>

// 希望写一个测试程序，对比 riscv 架构下，使用寄存器 gather 操作和 cache 命中的内存级别 gather 性能有多大差距
// 寄存器 gather 指令: vrgather.vv vd, vs2, vs1, vm
// 内存 gather 为 vluxe 指令: vluxe.v vd, (rs1), vs2, vm

#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <cstring>
#include <algorithm>

// 简化的测试配置 - 单个寄存器大小
constexpr size_t VLEN_BITS = 256;
constexpr size_t VLEN_BYTES = VLEN_BITS / 8;  // 32字节 
constexpr size_t TABLE_SIZE_ELEMENTS = VLEN_BYTES;  // 32个int8元素 
constexpr size_t VECTOR_LENGTH = VLEN_BYTES;  // 一次处理32个int8元素 
constexpr size_t ITERATIONS = 1000000;        // 增加迭代次数以获得更稳定结果 

class PerformanceTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    double stop() {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;
        return duration.count();
    }
};

// 生成随机索引
std::vector<uint8_t> generate_random_indices(size_t count, size_t max_index) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, max_index - 1);
    
    std::vector<uint8_t> indices(count);
    for (size_t i = 0; i < count; ++i) {
        indices[i] = static_cast<uint8_t>(dis(gen));
    }
    return indices;
}

// 初始化表数据
std::vector<int8_t> initialize_table() {
    std::vector<int8_t> table(TABLE_SIZE_ELEMENTS);
    
    for (size_t i = 0; i < TABLE_SIZE_ELEMENTS; ++i) {
        table[i] = static_cast<int8_t>((i * 3 + 7) % 256 - 128);  // 有规律但非线性的数据
    }
    
    return table;
}

// 模拟内存Gather - 通过标量操作模拟
double test_memory_gather_scalar(const std::vector<int8_t>& table, 
                                const std::vector<uint8_t>& indices) {
    PerformanceTimer timer;
    std::vector<int8_t> result(VECTOR_LENGTH);
    
    // 预热
    for (size_t i = 0; i < 1000; ++i) {
        for (size_t j = 0; j < VECTOR_LENGTH; ++j) {
            result[j] = table[indices[j]];
        }
    }
    
    timer.start();
    for (size_t iter = 0; iter < ITERATIONS; ++iter) {
        for (size_t j = 0; j < VECTOR_LENGTH; ++j) {
            result[j] = table[indices[j]];
        }
        // 防止编译器过度优化
        asm volatile("" : : "r"(result.data()) : "memory");
    }
    double time = timer.stop();
    
    volatile int8_t dummy = result[0];
    (void)dummy;
    
    return time;
}

// 模拟寄存器Gather - 假设在寄存器中操作
double test_register_gather_scalar(const std::vector<int8_t>& table,
                                  const std::vector<uint8_t>& indices) {
    PerformanceTimer timer;
    std::vector<int8_t> result(VECTOR_LENGTH);
    
    // 模拟"表在寄存器中"的情况 - 使用固定数组模拟寄存器访问
    int8_t register_table[TABLE_SIZE_ELEMENTS];
    memcpy(register_table, table.data(), TABLE_SIZE_ELEMENTS);
    
    // 预热
    for (size_t i = 0; i < 1000; ++i) {
        for (size_t j = 0; j < VECTOR_LENGTH; ++j) {
            result[j] = register_table[indices[j]];
        }
    }
    
    timer.start();
    for (size_t iter = 0; iter < ITERATIONS; ++iter) {
        for (size_t j = 0; j < VECTOR_LENGTH; ++j) {
            result[j] = register_table[indices[j]];
        }
        asm volatile("" : : "r"(result.data()) : "memory");
    }
    double time = timer.stop();
    
    volatile int8_t dummy = result[0];
    (void)dummy;
    
    return time;
}

// 使用RVV内联汇编的实际测试（如果环境支持）
#ifdef __riscv
#define RVV_GATHER_MEM_E8(dst, base, indices) \
    asm volatile("vluxei8.v %0, (%1), %2" : "=v"(dst) : "r"(base), "v"(indices))

#define RVV_GATHER_REG_E8(dst, table, indices) \
    asm volatile("vrgather.vv %0, %1, %2" : "=v"(dst) : "v"(table), "v"(indices))

#define RVV_SET_VL_E8(length) \
    asm volatile("vsetvli zero, %0, e8, m1" : : "r"(length))

double test_memory_gather_rvv(const std::vector<int8_t>& table, 
                             const std::vector<uint8_t>& indices) {
    PerformanceTimer timer;
    std::vector<int8_t> result(VECTOR_LENGTH);
    
    // 预热
    for (size_t i = 0; i < 1000; ++i) {
        RVV_SET_VL_E8(VECTOR_LENGTH);
        RVV_GATHER_MEM_E8(result.data(), table.data(), indices.data());
    }
    
    timer.start();
    for (size_t iter = 0; iter < ITERATIONS; ++iter) {
        RVV_SET_VL_E8(VECTOR_LENGTH);
        RVV_GATHER_MEM_E8(result.data(), table.data(), indices.data());
    }
    double time = timer.stop();
    
    return time;
}

double test_register_gather_rvv(const std::vector<int8_t>& table,
                               const std::vector<uint8_t>& indices) {
    PerformanceTimer timer;
    std::vector<int8_t> result(VECTOR_LENGTH);
    std::vector<int8_t> table_reg = table; // 模拟表在寄存器中
    
    // 预热
    for (size_t i = 0; i < 1000; ++i) {
        RVV_SET_VL_E8(VECTOR_LENGTH);
        RVV_GATHER_REG_E8(result.data(), table_reg.data(), indices.data());
    }
    
    timer.start();
    for (size_t iter = 0; iter < ITERATIONS; ++iter) {
        RVV_SET_VL_E8(VECTOR_LENGTH);
        RVV_GATHER_REG_E8(result.data(), table_reg.data(), indices.data());
    }
    double time = timer.stop();
    
    return time;
}
#endif

void print_system_info() {
    std::cout << "=== Simple RISC-V Vector Gather Performance Test ===\n";
    std::cout << "Data type: int8\n";
    std::cout << "VLEN: " << VLEN_BITS << " bits (" << VLEN_BYTES << " bytes)\n";
    std::cout << "Table size: 1 register (" << TABLE_SIZE_ELEMENTS << " int8 elements)\n";
    std::cout << "Vector length: " << VECTOR_LENGTH << " elements\n";
    std::cout << "Iterations: " << ITERATIONS << "\n\n";
}

int main() {
    print_system_info();
    
    // 准备测试数据
    auto table = initialize_table();
    auto indices = generate_random_indices(VECTOR_LENGTH, TABLE_SIZE_ELEMENTS);
    
    std::cout << "Table elements: ";
    for (size_t i = 0; i < std::min(size_t(10), TABLE_SIZE_ELEMENTS); ++i) 
        std::cout << (int)table[i] << " ";
    std::cout << "...\n";
    
    std::cout << "First 10 indices: ";
    for (size_t i = 0; i < std::min(size_t(10), VECTOR_LENGTH); ++i) 
        std::cout << (int)indices[i] << " ";
    std::cout << "\n\n";
    
    std::cout << "Running scalar simulation tests...\n";
    
    // 运行标量模拟测试
    double time_mem_scalar = test_memory_gather_scalar(table, indices);
    double time_reg_scalar = test_register_gather_scalar(table, indices);
    
    std::cout << "=== Scalar Simulation Results ===\n";
    std::cout << "Memory Gather (scalar sim): " << time_mem_scalar * 1e6 << " μs\n";
    std::cout << "Register Gather (scalar sim): " << time_reg_scalar * 1e6 << " μs\n";
    std::cout << "Speedup Ratio (scalar): " << time_mem_scalar / time_reg_scalar << "x\n\n";
    
#ifdef __riscv
    std::cout << "Running actual RVV tests...\n";
    
    // 运行实际的RVV测试
    double time_mem_rvv = test_memory_gather_rvv(table, indices);
    double time_reg_rvv = test_register_gather_rvv(table, indices);
    
    std::cout << "=== Actual RVV Results ===\n";
    std::cout << "Memory Gather (RVV): " << time_mem_rvv * 1e6 << " μs\n";
    std::cout << "Register Gather (RVV): " << time_reg_rvv * 1e6 << " μs\n";
    std::cout << "Speedup Ratio (RVV): " << time_mem_rvv / time_reg_rvv << "x\n\n";
    
    std::cout << "=== Comparison ===\n";
    std::cout << "Scalar simulation speedup: " << time_mem_scalar / time_reg_scalar << "x\n";
    std::cout << "RVV actual speedup: " << time_mem_rvv / time_reg_rvv << "x\n";
#else
    std::cout << "RISC-V RVV not available - using scalar simulation only\n";
#endif
    
    // 性能分析
    std::cout << "\n=== Performance Analysis ===\n";
    std::cout << "Expected characteristics for single-register table:\n";
    std::cout << "- Both table and indices should fit in L1 cache\n";
    std::cout << "- Memory gather still has cache access latency (~4 cycles)\n";
    std::cout << "- Register gather is pure ALU operation (~1-2 cycles)\n";
    std::cout << "- Typical speedup: 2-5x for this small table size\n";
    std::cout << "- Larger tables would show greater performance difference\n";
    
    std::cout << "\n=== Operations per Second ===\n";
    std::cout << "Memory Gather: " << ITERATIONS / time_mem_scalar << " ops/sec\n";
    std::cout << "Register Gather: " << ITERATIONS / time_reg_scalar << " ops/sec\n";
    
    return 0;
}