clang++ -DSPACEMIT_X60 -march=rv64gcv_zvfh -O3  q4k_q8_main_driver.cpp q4k_q8_kernel.cpp -o q4k_q8_kernel

clang++ -DSPACEMIT_X60 -march=rv64gcv_zvfh -O3 -g q4k_q8_main_driver.cpp q4k_q8_kernel.cpp -o q4k_q8_kernel
