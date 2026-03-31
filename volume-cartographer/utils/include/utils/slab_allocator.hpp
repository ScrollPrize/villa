#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdlib>

namespace utils {

// Lock-free slab allocator for fixed-size buffers.
//
// Maintains a stack of pre-allocated buffers. allocate() pops from
// the stack (falling back to aligned_alloc), deallocate() pushes
// back (falling back to free if the pool is full).
//
// Two variants:
//   SlabAllocator  — thread-safe (atomic top), for cross-thread alloc/free
//   LocalSlabAllocator — single-thread (plain int top), for thread_local use
//
// Common sizes in the rendering pipeline:
//   4 KB  = 16^3 blocks
//   1 MB  = 512*512*4 tile ARGB32 buffers
//   2 MB  = 128^3 decoded chunks (handled by HugePageAllocator)

// Thread-safe variant: lock-free stack with atomic CAS.
// Use when allocate() and deallocate() may run on different threads
// (e.g., QImage buffers allocated on workers, freed on main thread).
template<size_t SlabSize, size_t PoolCount = 64>
class SlabAllocator {
    std::array<void*, PoolCount> pool_;
    std::atomic<int> top_{0};

public:
    SlabAllocator() = default;

    ~SlabAllocator()
    {
        int t = top_.load(std::memory_order_relaxed);
        for (int i = 0; i < t; ++i)
            std::free(pool_[i]);
    }

    SlabAllocator(const SlabAllocator&) = delete;
    SlabAllocator& operator=(const SlabAllocator&) = delete;

    void* allocate()
    {
        int t = top_.load(std::memory_order_relaxed);
        while (t > 0) {
            if (top_.compare_exchange_weak(t, t - 1,
                    std::memory_order_acquire, std::memory_order_relaxed))
                return pool_[t - 1];
        }
        return std::aligned_alloc(64, SlabSize);
    }

    void deallocate(void* p)
    {
        if (!p) return;
        int t = top_.load(std::memory_order_relaxed);
        while (static_cast<size_t>(t) < PoolCount) {
            pool_[t] = p;
            if (top_.compare_exchange_weak(t, t + 1,
                    std::memory_order_release, std::memory_order_relaxed))
                return;
        }
        std::free(p);
    }

    static constexpr size_t slab_size() { return SlabSize; }
    static constexpr size_t pool_count() { return PoolCount; }
};

// Single-thread variant: no atomics, use with thread_local.
// Zero overhead for same-thread alloc/free patterns.
template<size_t SlabSize, size_t PoolCount = 64>
class LocalSlabAllocator {
    std::array<void*, PoolCount> pool_;
    int top_{0};

public:
    LocalSlabAllocator() = default;

    ~LocalSlabAllocator()
    {
        for (int i = 0; i < top_; ++i)
            std::free(pool_[i]);
    }

    LocalSlabAllocator(const LocalSlabAllocator&) = delete;
    LocalSlabAllocator& operator=(const LocalSlabAllocator&) = delete;

    void* allocate()
    {
        if (top_ > 0)
            return pool_[--top_];
        return std::aligned_alloc(64, SlabSize);
    }

    void deallocate(void* p)
    {
        if (!p) return;
        if (static_cast<size_t>(top_) < PoolCount) {
            pool_[top_++] = p;
            return;
        }
        std::free(p);
    }

    static constexpr size_t slab_size() { return SlabSize; }
    static constexpr size_t pool_count() { return PoolCount; }
};

// 1 MB slab for 512x512 ARGB32 tile buffers (thread-safe: allocated on
// worker threads, freed on main thread when QImage is destroyed)
using TileBufferSlab = SlabAllocator<512 * 512 * 4, 32>;

// 4 KB slab for 16^3 sampler blocks (thread-local: same-thread lifecycle)
using BlockSlab = LocalSlabAllocator<4096, 64>;

} // namespace utils
