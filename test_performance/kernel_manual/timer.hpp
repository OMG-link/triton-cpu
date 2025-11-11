#include <chrono>

#ifdef SPACEMIT_X60
class Timer {
  public:
    void start() { start_time = std::chrono::high_resolution_clock::now(); }

    int64_t stop() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        return static_cast<int64_t>(duration.count() * 1.6);
    }

  private:
    std::chrono::high_resolution_clock::time_point start_time;
};
#else
#ifdef XUANTIE_C910
static inline int64_t rdcycle() {
    int64_t cycle;
    asm volatile("rdcycle %0" : "=r"(cycle));
    return cycle;
}

class Timer {
  public:
    void start() { start_cycle = rdcycle(); }

    int64_t stop() {
        auto end_cycle = rdcycle();
        return end_cycle - start_cycle;
    }

  private:
    int64_t start_cycle;
};
#else
#error "Unknown CPU. Cannot determine the timer to use."
#endif
#endif