import triton
import triton.language as tl
import torch
import numpy as np

# typedef struct {
#     uint8_t  scales[QK_K/32];
#     uint8_t  qs[QK_K/2];
# } block_iq4_ks;

# typedef struct {
#     float   d;              // delta
#     float   sum;            // sum of quants in the entire block
#     int8_t  qs[QK_K];       // quants
#     int16_t bsums[QK_K/16]; // sum of quants in groups of 16
# } block_q8_K;

# IQ4K查找表 - 对应 ggml-common.h 中的 iq4k_values
IQ4K_VALUES = torch.tensor([
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    -123, -100, -79, -61, -45, -31, -18,  -6, 5, 17, 29, 42, 57, 73, 93, 117
], dtype=torch.int8)

# 常量定义
QK_K = 256  # GGML标准块大小
K_BLOCK_SIZE = 32  # IQ4_KS子块大小
NUM_SUB_BLOCKS = QK_K // K_BLOCK_SIZE  # 每个QK_K块包含的子块数

# 输入数据改成 struct of array 结构
@triton.jit
def iq4_ks_q8_k_dot_kernel(
	iq4_qs_data_ptr, # int4_packed uint8
	iq4_scales_data_ptr, # uint8
	q8_k_qs_ptr, # int8
	q8_k_d_ptr, # float
	q4_dequant_data_ptr, # int8
	output_ptr, # float 输出浮点矩阵
	value_table_ptr
):
	# 每个 pid 负责一个 super block
	# 还有一个global scale，这里先忽略

	QK_K = 256
	K_BLOCK_SIZE = 32
	NUM_SUB_BLOCKS = QK_K // K_BLOCK_SIZE

	pid = tl.program_id(axis=0)

	global_iq4_scales_offset = pid * QK_K//32

	q8_k_d = tl.load(q8_k_d_ptr + pid)
	suml = 0.0

	for subblock in range(0, NUM_SUB_BLOCKS): # 256/32 = 8
		iq4_scale_offset = global_iq4_scales_offset + subblock
		iq_4_scales_bytes = tl.load(iq4_scales_data_ptr + iq4_scale_offset)
		iq4_scales_value = (iq_4_scales_bytes & 254) - 127 # 转换为-127到127范围
		
		# (scales[ib] & 1) << 4 决定使用表前16个还是后16个值
		table_offset = (iq_4_scales_bytes & 1) << 4
		table = tl.load(value_table_ptr + table_offset + tl.arange(0, 16))
		
		block_offset = pid * QK_K//2 + subblock * K_BLOCK_SIZE // 2	
		iq4_qs_data = tl.load(iq4_qs_data_ptr + block_offset + tl.arange(0, 16))
		
		iq4_low_4bits = iq4_qs_data & 0xf 
		iq4_high_4bits = iq4_qs_data >> 4 

		# 查表使用 tl.gather 
		iq4_low_4bits_dequant_values = tl.gather(table, iq4_low_4bits, axis=0)
		iq4_high_4bits_dequant_values = tl.gather(table, iq4_high_4bits, axis=0)

		# 内积计算
		q8_k_qs_data_low = tl.load(q8_k_qs_ptr + pid * QK_K + subblock * K_BLOCK_SIZE + tl.arange(0, 16))
		q8_k_qs_data_high = tl.load(q8_k_qs_ptr + pid * QK_K + subblock * K_BLOCK_SIZE + 16 + tl.arange(0, 16))
		suml += q8_k_d * iq4_scales_value * tl.sum(q8_k_qs_data_low * iq4_low_4bits_dequant_values, axis=0)
		suml += q8_k_d * iq4_scales_value * tl.sum(q8_k_qs_data_high * iq4_high_4bits_dequant_values, axis=0)
	tl.store(output_ptr, suml)


iq4_qs_data = torch.full((QK_K//2, ), 0x10, dtype=torch.uint8) # 16个 int4_packed
iq4_scales_data = torch.randint(0, 0x11, (QK_K//32,), dtype=torch.uint8)
q8_k_qs = torch.randint(-128, 127, (QK_K,), dtype=torch.int8)
q8_k_d = torch.rand(1, dtype=torch.float32)
q4_dequant_data = torch.zeros(QK_K, dtype=torch.int8)
output = torch.zeros(1, dtype=torch.float32)

# 调用kernel 处理一行一列，每个 kernel 处理 256 长度数据
# 这里就一个线程，所以输出 就是 1 个常数
iq4_ks_q8_k_dot_kernel[(1,)](
	iq4_qs_data_ptr=iq4_qs_data,
	iq4_scales_data_ptr=iq4_scales_data,
	q8_k_qs_ptr=q8_k_qs,
	q8_k_d_ptr=q8_k_d,
	q4_dequant_data_ptr=q4_dequant_data,
	output_ptr=output,
	value_table_ptr=IQ4K_VALUES,
)

print(output)
print(q4_dequant_data)


