scp ./i8_i8_gemm_driver.py i8_i8_gemm.py shenrh@192.168.123.44:/home/shenrh/triton-cpu/test_performance/kernel_triton

TRITON_ALWAYS_COMPILE=1  TRITON_CPU_BACKEND=1  python i8_i8_gemm_driver.py --sweep --vlen 256 --freq 1.6 --num-threads=1 --rep=100 --warmup=10 --n-kernel-repeat=30 --csv-out i8_i8_gemm_results.csv --plot-out i8_i8_gemm_results.png 

TRITON_ALWAYS_COMPILE=1  TRITON_CPU_BACKEND=1  python i8_i8_gemm_driver.py --m 192 --n 2048 --k 2048 --vlen 256 --freq 1.6 --num-threads=1 --rep=100 --warmup=10 --n-kernel-repeat=30 --csv-out i8_i8_gemm_results.csv --plot-out i8_i8_gemm_results.png 

scp shenrh@192.168.123.44:/home/shenrh/triton-cpu/test_performance/kernel_triton/i8_i8_gemm_results.csv ./
scp shenrh@192.168.123.44:/home/shenrh/triton-cpu/test_performance/kernel_triton/i8_i8_gemm_results.png ././k