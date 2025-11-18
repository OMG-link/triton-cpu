clang++ -DSPACEMIT_X60 -march=rv64gcv_zvfh -O3  q4k_q8_main_driver.cpp q4k_q8_kernel.cpp -o q4k_q8_kernel

clang++ -DSPACEMIT_X60 -march=rv64gcv_zvfh -O3 -g q4k_q8_main_driver.cpp q4k_q8_kernel.cpp -o q4k_q8_kernel

clang++ -DSPACEMIT_X60 -march=rv64gcv_zvfh -O3 -fopenmp q40_q80_gemv_main_driver.cpp q40_q80_gemv_kernel.cpp -o q40_q80_gemv_kernel

# false - 不绑定，由操作系统调度
export OMP_PROC_BIND=false

# true - 绑定，使用默认策略
export OMP_PROC_BIND=true

# master - 主线程绑定，其他线程靠近主线程
export OMP_PROC_BIND=master

# close - 线程绑定在相邻的CPU核心上
export OMP_PROC_BIND=close

# spread - 线程均匀分布在CPU核心上
export OMP_PROC_BIND=spread


# 指定具体的CPU核心
export OMP_PLACES="{0,1,2,3,4,5,6,7}"

# 使用线程、核心或socket
export OMP_PLACES=threads    # 每个硬件线程
export OMP_PLACES=cores      # 每个物理核心
export OMP_PLACES=sockets    # 每个CPU插槽

# 指定核心范围
export OMP_PLACES="{0:8}"    # 核心0到7
export OMP_PLACES="{0:4:2}"  # 核心0,2,4,6


# 绑定线程到物理核心，避免超线程
export OMP_NUM_THREADS=16
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# 或者明确指定核心
export OMP_PLACES="{0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30}"



OMP_PROC_BIND=true ./q40_q80_gemv_kernel 20480 20480 10

OMP_PLACES="{0,1,2,3,4,6,7}" OMP_PROC_BIND=true ./q40_q80_gemv_kernel 20480 20480 10