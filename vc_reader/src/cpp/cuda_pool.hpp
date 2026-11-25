#pragma once
// Release policy of the device's default memory pool, which every stream-ordered
// vc_reader allocation uses. CUDA's own default (threshold 0) returns freed memory
// to the driver at each synchronization, so every call pays for fresh allocations.
// By default vc_reader keeps freed memory for reuse, as CuPy and PyTorch caching
// allocators do; set_pool_release_threshold caps what is kept.
//
// The setting lives in the pool itself, shared by every module in the process. A
// pool CUDA has not configured reads 0, which the first allocation replaces with
// the default; a requested 0 is stored as 1 byte, which releases the same memory.
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace vcr::device {
inline void pool_check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("CUDA memory pool: ") + cudaGetErrorString(e));
}
inline cudaMemPool_t default_pool(int device) {
    cudaMemPool_t pool{}; pool_check(cudaDeviceGetDefaultMemPool(&pool, device)); return pool;
}
// Called before a pooled allocation on device: applies the default once per pool.
inline void prepare_pool(int device) {
    thread_local int prepared = -1;  // the last device seen on this thread
    if (prepared == device) return;
    auto pool = default_pool(device);
    std::uint64_t threshold = 0;
    pool_check(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold));
    if (threshold == 0) {
        threshold = std::numeric_limits<std::uint64_t>::max();
        pool_check(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold));
    }
    prepared = device;
}
// Keep at most `bytes` of freed memory in device's pool across synchronizations
// (UINT64_MAX keeps everything, 0 releases it all).
inline void set_pool_release_threshold(int device, std::uint64_t bytes) {
    if (bytes == 0) bytes = 1;
    pool_check(cudaMemPoolSetAttribute(default_pool(device), cudaMemPoolAttrReleaseThreshold, &bytes));
}
inline std::uint64_t pool_release_threshold(int device) {
    std::uint64_t bytes = 0;
    pool_check(cudaMemPoolGetAttribute(default_pool(device), cudaMemPoolAttrReleaseThreshold, &bytes));
    return bytes == 1 ? 0 : bytes;
}
}
