# 默认测试（N=512, K=1024, 5轮测试）
TRITON_ALWAYS_COMPILE=1  TRITON_CPU_BACKEND=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=./ir-dump python3 q40_q80_gemv.py

# 自定义参数
TRITON_ALWAYS_COMPILE=1  TRITON_CPU_BACKEND=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=./ir-dump python3 q40_q80_gemv.py --n 2048 --k 2048 --rounds 10

# 简单功能测试
TRITON_ALWAYS_COMPILE=1  TRITON_CPU_BACKEND=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=./ir-dump python3 q40_q80_gemv.py --simple-test

# 完整参数示例 909 组数据
TRITON_ALWAYS_COMPILE=1  TRITON_CPU_BACKEND=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=./ir-dump python3 q40_q80_gemv.py --n 2048 --k 2048 --rounds 5 --warmup 3 --target-gb 2.0


# 9 组 数据
TRITON_ALWAYS_COMPILE=1  TRITON_CPU_BACKEND=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=./ir-dump python3 q40_q80_gemv.py --n 20480 --k 20480 --rounds 5 --warmup 3 --target-gb 2.0

scp -r ./kernel_triton shenrh@192.168.123.44:/home/shenrh/triton-cpu/test_performance/kernel_triton 
scp -r  shenrh@192.168.123.44:/home/shenrh/triton-cpu/test_performance/kernel_triton/ir-dump/HIABAAYPNQMZZJ46UHSXX6GCTAEMZV5W3RKOYUCK376UYMUBR2GA ./ir-dump/
scp ./q40_q80_gemm.py shenrh@192.168.123.44:/home/shenrh/triton-cpu/test_performance/kernel_triton