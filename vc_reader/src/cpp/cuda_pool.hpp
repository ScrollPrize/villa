#pragma once
// Stream-ordered allocation from each device's default CUDA memory pool.
#include <cuda_runtime.h>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
namespace vcr::device {
// Whether `device` supports cudaMallocAsync. The first call for a device also makes its
// default pool keep freed memory: the release threshold defaults to 0, so every
// synchronize handed the pool's free memory back to the driver and later allocations
// mapped it again (periodic 0.14-0.27 s cudaMallocAsync stalls in a training data
// pipeline). PyTorch's caching allocator likewise retains freed memory.
inline bool pooled_allocation(int device) {
#if CUDART_VERSION >= 11020
    static std::mutex mutex;
    static std::unordered_map<int, bool> pooled;
    const auto check = [](cudaError_t status) {
        if (status != cudaSuccess) throw std::runtime_error(std::string("CUDA memory pool: ") + cudaGetErrorString(status));
    };
    std::lock_guard lock(mutex);
    if (auto it = pooled.find(device); it != pooled.end()) return it->second;
    int supported = 0;
    check(cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported, device));
    if (supported) {
        cudaMemPool_t pool;
        check(cudaDeviceGetDefaultMemPool(&pool, device));
        std::uint64_t threshold = UINT64_MAX;
        check(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold));
    }
    return pooled[device] = supported != 0;
#else
    (void)device;
    return false;
#endif
}
struct PoolStats { std::uint64_t reserved, reserved_high, used, used_high; };
inline PoolStats pool_stats(int device) {
    PoolStats stats{};
#if CUDART_VERSION >= 11020
    const auto check = [](cudaError_t status) {
        if (status != cudaSuccess) throw std::runtime_error(std::string("CUDA memory pool: ") + cudaGetErrorString(status));
    };
    cudaMemPool_t pool;
    check(cudaDeviceGetDefaultMemPool(&pool, device));
    check(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &stats.reserved));
    check(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemHigh, &stats.reserved_high));
    check(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &stats.used));
    check(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &stats.used_high));
#else
    (void)device;
#endif
    return stats;
}
}  // namespace vcr::device
