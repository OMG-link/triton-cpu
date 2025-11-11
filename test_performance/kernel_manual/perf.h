#ifndef PERF_H
#define PERF_H

#include <linux/perf_event.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

static inline void flush_l3_cache() {
#ifdef _OPENMP
#pragma omp parallel
    {
#endif
        static const int seed = time(0) % 256;
        const size_t L3_CACHE_SIZE_KB = 8 * 1024;
        const size_t BUFFER_SIZE = L3_CACHE_SIZE_KB * 1024 * 2;
        static char *buffer = static_cast<char *>(malloc(BUFFER_SIZE));
        volatile char sink = 0;
        for (size_t i = 0; i < BUFFER_SIZE; i += 64) {
            buffer[i] = static_cast<char>((seed * i) % 128);
        }
        for (size_t i = 0; i < BUFFER_SIZE; i += 64) {
            sink ^= buffer[i];
        }
#ifdef _OPENMP
    }
#endif
}

static inline int get_cpu_id() { return sched_getcpu(); }

static inline int perf_open_event(uint32_t type, uint64_t config) {
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.type = type;
    pe.size = sizeof(pe);
    pe.config = config;
    pe.disabled = 1;
    pe.inherit = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;

    int fd = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
    if (fd < 0) {
        perror("perf_event_open");
        exit(1);
    }
    return fd;
}

static inline void perf_close_event(int fd) {
    if (close(fd) < 0) {
        perror("close");
        exit(1);
    }
}

static inline void perf_reset(int fd) {
    if (ioctl(fd, PERF_EVENT_IOC_RESET, 0) < 0 || ioctl(fd, PERF_EVENT_IOC_ENABLE, 0) < 0) {
        perror("perf reset/enable");
        exit(1);
    }
}

static inline void perf_disable(int fd) {
    if (ioctl(fd, PERF_EVENT_IOC_DISABLE, 0) < 0) {
        perror("perf disable");
        exit(1);
    }
}

static inline uint64_t perf_read(int fd) {
    uint64_t value;
    ssize_t ret = read(fd, &value, sizeof(value));
    if (ret != sizeof(value)) {
        perror("perf read");
        exit(1);
    }
    return value;
}

static inline int perf_event_cycles(void) { return perf_open_event(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES); }

static inline int perf_event_instructions(void) { return perf_open_event(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS); }

static inline int perf_event_task_clock(void) { return perf_open_event(PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK); }

static inline int perf_event_page_faults(void) { return perf_open_event(PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS); }

static inline int perf_event_dtlb_access(void) {
    return perf_open_event(PERF_TYPE_HW_CACHE, PERF_COUNT_HW_CACHE_DTLB | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16));
}

static inline int perf_event_dtlb_miss(void) { return perf_open_event(PERF_TYPE_HW_CACHE, PERF_COUNT_HW_CACHE_DTLB | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)); }

static inline int perf_event_l1d_access(void) { return perf_open_event(PERF_TYPE_HW_CACHE, PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16)); }

static inline int perf_event_l1d_miss(void) { return perf_open_event(PERF_TYPE_HW_CACHE, PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)); }

static inline int perf_event_llc_access(void) { return perf_open_event(PERF_TYPE_HW_CACHE, PERF_COUNT_HW_CACHE_LL | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16)); }

static inline int perf_event_llc_miss(void) { return perf_open_event(PERF_TYPE_HW_CACHE, PERF_COUNT_HW_CACHE_LL | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)); }

#ifdef __cplusplus
}
#endif

#endif /* PERF_H */