#include <atomic>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <thread>

[[gnu::noinline]]
static void trip_asan_heap_oob()
{
    volatile int* p = new int[4];
    p[64] = 0;
    std::fprintf(stderr, "ASan canary did not fire\n");
}

[[gnu::noinline]]
static void trip_ubsan_signed_overflow()
{
    volatile int x = INT_MAX;
    volatile int y = 1;
    volatile int z = x + y;
    (void)z;
    std::fprintf(stderr, "UBSan canary did not fire\n");
}

[[gnu::noinline]]
static void trip_tsan_data_race()
{
    static int shared = 0;
    auto writer = [] {
        for (int i = 0; i < 1'000'000; ++i) shared = i;
    };
    std::thread t1(writer);
    std::thread t2(writer);
    t1.join();
    t2.join();
    std::fprintf(stderr, "TSan canary did not fire\n");
}

[[gnu::noinline]]
static void trip_tysan_type_punning()
{
    alignas(double) unsigned char storage[sizeof(double)] = {};
    double* dp = new (storage) double(3.14);
    int* ip = reinterpret_cast<int*>(dp);
    volatile int v = *ip;
    (void)v;
    std::fprintf(stderr, "TySan canary did not fire\n");
}

[[gnu::noinline, clang::optnone]]
static float nsan_loop_sum(float seed)
{
    float s = seed;
    for (int i = 0; i < 100; ++i) {
        s += 1.0e-7f;
        s -= 1.0e-7f;
        s += 1.0e8f;
        s -= 1.0e8f;
    }
    return s;
}

[[gnu::noinline]]
static void trip_nsan_catastrophic_cancellation()
{
    volatile float seed = 1.0f;
    float r = nsan_loop_sum(seed);
    std::fprintf(stderr, "result=%g\n", static_cast<double>(r));
    std::fprintf(stderr, "NSan canary did not fire\n");
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s {asan|ubsan|tsan|tysan|nsan}\n", argv[0]);
        return 2;
    }
    std::string which = argv[1];
    if      (which == "asan")  trip_asan_heap_oob();
    else if (which == "ubsan") trip_ubsan_signed_overflow();
    else if (which == "tsan")  trip_tsan_data_race();
    else if (which == "tysan") trip_tysan_type_punning();
    else if (which == "nsan")  trip_nsan_catastrophic_cancellation();
    else {
        std::fprintf(stderr, "unknown canary: %s\n", argv[1]);
        return 2;
    }
    return 0;
}
