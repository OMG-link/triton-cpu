import triton
import triton.language as tl
import torch
import numpy as np

# IQ4K查找表 - 对应ggml-common.h中的iq4k_values
IQ4K_VALUES = torch.tensor([
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    -123, -100, -79, -61, -45, -31, -18,  -6, 5, 17, 29, 42, 57, 73, 93, 117
], dtype=torch.int8)

# 常量定义
QK_K = 256  # GGML标准块大小
K_BLOCK_SIZE = 32  # IQ4_KS子块大小
NUM_SUB_BLOCKS = QK_K // K_BLOCK_SIZE  # 每个QK_K块包含的子块数

@triton.jit
def iq4_ks_q8_k_dot_kernel(
    # 输入指针
    iq4_ks_data_ptr,      # IQ4_KS量化数据指针
    q8_k_data_ptr,        # Q8_K量化数据指针
    iq4k_values_ptr,      # IQ4K查找表指针
    
    # 输出指针
    output_ptr,           # 输出结果指针
    
    # 维度信息
    n_elements,          # 总元素数
    n_blocks,            # 块数量 (n_elements / QK_K)
    
    # 内存步长
    iq4_ks_stride,       # IQ4_KS数据步长
    q8_k_stride,         # Q8_K数据步长
    output_stride,       # 输出步长
    
    # Triton程序配置
    BLOCK_SIZE: tl.constexpr,
):
    """
    IQ4_KS与Q8_K点积计算的Triton内核
    
    对应C++函数: vec_dot_iq4_ks_q8_k
    """
    
    # 获取程序ID
    pid = tl.program_id(axis=0)
    
    # 计算当前块的数据偏移
    block_offset = pid * QK_K
    
    # 检查边界
    if block_offset >= n_elements:
        return
    
    # 加载 IQ4_KS 全局缩放因子
    # IQ4_KS 数据布局: [全局缩放因子] [块1缩放因子] [块1量化数据] ...
    global_scale_offset = pid * iq4_ks_stride
    global_scale = tl.load(iq4_ks_data_ptr + global_scale_offset)
    
    # 加载Q8_K全局缩放因子
    q8_k_scale_offset = pid * q8_k_stride
    q8_k_scale = tl.load(q8_k_data_ptr + q8_k_scale_offset)
    
    # 组合缩放因子
    combined_scale = global_scale * q8_k_scale
    
    # 初始化累加器
    sumf = 0.0
    
    # 处理每个子块 (32个元素)
    for sub_block in range(NUM_SUB_BLOCKS):
        # 计算子块偏移
        sub_block_offset = block_offset + sub_block * K_BLOCK_SIZE
        
        # 加载IQ4_KS子块缩放因子
        # scales[ib] & 254 提取7位缩放值
        scale_offset = global_scale_offset + 1 + sub_block  # +1跳过全局缩放因子
        scale_byte = tl.load(iq4_ks_data_ptr + scale_offset)
        scale_value = (scale_byte & 254) - 127  # 转换为-127到127范围
        
        # 计算局部缩放因子
        local_scale = combined_scale * scale_value
        
        # 确定使用哪组查找表值
        # (scales[ib] & 1) << 4 决定使用前16个还是后16个值
        table_offset = (scale_byte & 1) << 4
        
        # 加载量化数据
        qx_offset = global_scale_offset + 1 + NUM_SUB_BLOCKS + sub_block * (K_BLOCK_SIZE // 2)
        qy_offset = q8_k_scale_offset + 1 + sub_block * K_BLOCK_SIZE
        
        # 初始化子块累加器
        suml = 0.0
        
        # 处理子块内的16个元素对
        for j in range(K_BLOCK_SIZE // 2):
            # 加载IQ4_KS量化数据 (4位打包)
            qx_byte = tl.load(iq4_ks_data_ptr + qx_offset + j)
            
            # 提取低4位和高4位
            low_4bits = qx_byte & 0xf
            high_4bits = (qx_byte >> 4) & 0xf
            
            # 加载Q8_K数据
            qy_low = tl.load(q8_k_data_ptr + qy_offset + j)
            qy_high = tl.load(q8_k_data_ptr + qy_offset + j + K_BLOCK_SIZE // 2)
            
            # 查表获取解量化值
            val_low = tl.load(iq4k_values_ptr + table_offset + low_4bits)
            val_high = tl.load(iq4k_values_ptr + table_offset + high_4bits)
            
            # 累加点积
            suml += qy_low * val_low + qy_high * val_high
        
        # 应用局部缩放并累加到总结果
        sumf += local_scale * suml
    
    # 存储结果
    tl.store(output_ptr + pid * output_stride, sumf)


def iq4_ks_q8_k_dot_triton(
    iq4_ks_data: torch.Tensor,
    q8_k_data: torch.Tensor,
    iq4k_values: torch.Tensor = None
) -> torch.Tensor:
    """
    IQ4_KS与Q8_K点积计算的Triton包装函数
    
    Args:
        iq4_ks_data: IQ4_KS量化数据 [n_blocks, block_size]
        q8_k_data: Q8_K量化数据 [n_blocks, block_size] 
        iq4k_values: IQ4K查找表，如果为None则使用默认值
        
    Returns:
        点积结果 [n_blocks]
    """
    if iq4k_values is None:
        iq4k_values = IQ4K_VALUES
    
    # 确保数据在GPU上
    device = iq4_ks_data.device
    iq4_ks_data = iq4_ks_data.to(device)
    q8_k_data = q8_k_data.to(device)
    iq4k_values = iq4k_values.to(device)
    
    # 获取维度信息
    n_blocks = iq4_ks_data.shape[0]
    n_elements = n_blocks * QK_K
    
    # 创建输出张量
    output = torch.zeros(n_blocks, device=device, dtype=torch.float32)
    
    # 计算内存步长
    iq4_ks_stride = iq4_ks_data.stride(0)
    q8_k_stride = q8_k_data.stride(0)
    output_stride = output.stride(0)
    
    # 启动Triton内核
    grid = (n_blocks,)
    
    iq4_ks_q8_k_dot_kernel[grid](
        iq4_ks_data_ptr=iq4_ks_data,
        q8_k_data_ptr=q8_k_data,
        iq4k_values_ptr=iq4k_values,
        output_ptr=output,
        n_elements=n_elements,
        n_blocks=n_blocks,
        iq4_ks_stride=iq4_ks_stride,
        q8_k_stride=q8_k_stride,
        output_stride=output_stride,
        BLOCK_SIZE=QK_K,
    )
    
    return output


# 辅助函数：创建测试数据
def create_test_data(n_blocks: int = 4, device: str = "cuda"):
    """
    创建测试用的IQ4_KS和Q8_K数据
    """
    # IQ4_KS数据布局: [全局缩放因子] [块缩放因子] [量化数据]
    iq4_ks_block_size = 1 + NUM_SUB_BLOCKS + (QK_K // 2)  # 全局缩放 + 子块缩放 + 量化数据
    iq4_ks_data = torch.randn(n_blocks, iq4_ks_block_size, device=device, dtype=torch.float32)
    
    # Q8_K数据布局: [全局缩放因子] [量化数据]
    q8_k_block_size = 1 + QK_K  # 全局缩放 + 量化数据
    q8_k_data = torch.randn(n_blocks, q8_k_block_size, device=device, dtype=torch.float32)
    
    # 设置合理的缩放因子范围
    iq4_ks_data[:, 0] = torch.rand(n_blocks, device=device) * 2.0 - 1.0  # 全局缩放 [-1, 1]
    q8_k_data[:, 0] = torch.rand(n_blocks, device=device) * 2.0 - 1.0     # 全局缩放 [-1, 1]
    
    # 设置子块缩放因子 [-127, 127]
    for i in range(1, 1 + NUM_SUB_BLOCKS):
        iq4_ks_data[:, i] = torch.randint(-127, 128, (n_blocks,), device=device).float()
    
    # 设置量化数据 (4位索引 0-15)
    quantized_data_start = 1 + NUM_SUB_BLOCKS
    iq4_ks_data[:, quantized_data_start:] = torch.randint(0, 16, 
        (n_blocks, QK_K // 2), device=device).float()
    
    # Q8_K量化数据 (8位有符号整数)
    q8_k_data[:, 1:] = torch.randint(-128, 128, 
        (n_blocks, QK_K), device=device).float()
    
    return iq4_ks_data, q8_k_data


# 测试函数
def test_iq4_ks_q8_k_dot():
    """
    测试IQ4_KS与Q8_K点积计算
    """
    print("测试IQ4_KS与Q8_K点积计算...")
    
    # 创建测试数据
    n_blocks = 8
    iq4_ks_data, q8_k_data = create_test_data(n_blocks)
    
    # 执行Triton计算
    result = iq4_ks_q8_k_dot_triton(iq4_ks_data, q8_k_data)
    
    print(f"输入块数: {n_blocks}")
    print(f"输出形状: {result.shape}")
    print(f"结果范围: [{result.min().item():.6f}, {result.max().item():.6f}]")
    print(f"结果均值: {result.mean().item():.6f}")
    print(f"结果标准差: {result.std().item():.6f}")
    
    return result


if __name__ == "__main__":
    # 检查CUDA可用性
    if torch.cuda.is_available():
        print("CUDA可用，运行Triton测试...")
        test_iq4_ks_q8_k_dot()
    else:
        print("CUDA不可用，跳过Triton测试")
        print("程序结构:")
        print("- iq4_ks_q8_k_dot_kernel: Triton内核函数")
        print("- iq4_ks_q8_k_dot_triton: 包装函数")
        print("- create_test_data: 测试数据生成")
        print("- test_iq4_ks_q8_k_dot: 测试函数")
