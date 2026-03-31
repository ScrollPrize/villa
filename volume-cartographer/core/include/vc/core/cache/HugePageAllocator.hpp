#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#else
#include <cstdlib>
#endif

namespace vc::cache {

// Three-tier huge-page-aligned memory allocator for the chunk cache.
//
//   16^3  =   4 KB blocks (OS page)      — sampler cache slot
//   128^3 =   2 MB chunks (huge page)    — H265 decode unit
//   1024^3 =  1 GB shards (1GB huge page) — disk I/O unit
//
// Pre-allocates a pool of 2MB-aligned buffers using mmap with MAP_HUGETLB
// on Linux, falling back to regular mmap with manual alignment otherwise.
class HugePageAllocator {
public:
    static constexpr size_t k4KB  = 4096;
    static constexpr size_t k2MB  = 2 * 1024 * 1024;
    static constexpr size_t k1GB  = 1024ULL * 1024 * 1024;

    static constexpr size_t kDefaultPoolSize = 512;  // 512 * 2MB = 1GB

    static HugePageAllocator& instance()
    {
        static HugePageAllocator inst;
        return inst;
    }

    // Allocate a 2MB-aligned buffer from the pool.
    uint8_t* allocate2MB()
    {
        std::lock_guard<std::mutex> lk(mutex_);
        if (!freeList_.empty()) {
            auto* p = freeList_.back();
            freeList_.pop_back();
            return p;
        }
        // Pool exhausted — allocate a new buffer on demand
        return allocSingle2MB();
    }

    // Return a 2MB buffer to the pool.
    void free2MB(uint8_t* ptr)
    {
        if (!ptr) return;
        std::lock_guard<std::mutex> lk(mutex_);
        freeList_.push_back(ptr);
    }

    // Memory-map 1GB from a file descriptor (shard-level).
    // Uses MAP_HUGETLB | MAP_HUGE_1GB on Linux if available.
    static uint8_t* mmap1GB([[maybe_unused]] int fd, [[maybe_unused]] off_t offset)
    {
#ifdef __linux__
        // Try 1GB huge page first
        void* p = ::mmap(nullptr, k1GB, PROT_READ, MAP_PRIVATE | MAP_HUGETLB | MAP_HUGE_1GB,
                         fd, offset);
        if (p != MAP_FAILED) return static_cast<uint8_t*>(p);

        // Fallback: regular mmap
        p = ::mmap(nullptr, k1GB, PROT_READ, MAP_PRIVATE, fd, offset);
        if (p != MAP_FAILED) return static_cast<uint8_t*>(p);

        return nullptr;
#else
        return nullptr;
#endif
    }

    // Unmap a 1GB shard mapping.
    static void munmap1GB(uint8_t* ptr)
    {
        if (!ptr) return;
#ifdef __linux__
        ::munmap(ptr, k1GB);
#endif
    }

    // Pool statistics
    size_t poolTotal() const { return poolTotal_; }
    size_t poolFree() const
    {
        std::lock_guard<std::mutex> lk(mutex_);
        return freeList_.size();
    }

private:
    HugePageAllocator() { initPool(kDefaultPoolSize); }

    ~HugePageAllocator()
    {
#ifdef __linux__
        if (hugePoolBase_) {
            ::munmap(hugePoolBase_, hugePoolBytes_);
        }
#endif
        // Free overflow allocations (not part of the bulk pool)
        for (auto* p : allAllocations_) freeSingle2MB(p);
    }

    void initPool(size_t count)
    {
#ifdef __linux__
        // Try to allocate the entire pool as one big mmap with 2MB huge pages
        size_t totalBytes = count * k2MB;
        void* p = ::mmap(nullptr, totalBytes,
                         PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB,
                         -1, 0);
        if (p != MAP_FAILED) {
            hugePoolBase_ = static_cast<uint8_t*>(p);
            hugePoolBytes_ = totalBytes;
            hugePageMode_ = true;
            freeList_.reserve(count);
            for (size_t i = 0; i < count; i++) {
                freeList_.push_back(hugePoolBase_ + i * k2MB);
            }
            poolTotal_ = count;
            return;
        }
#endif
        // Fallback: individual allocations with manual alignment
        hugePageMode_ = false;
        freeList_.reserve(count);
        for (size_t i = 0; i < count; i++) {
            auto* buf = allocSingle2MB();
            if (buf) freeList_.push_back(buf);
        }
        poolTotal_ = freeList_.size();
    }

    uint8_t* allocSingle2MB()
    {
#ifdef __linux__
        // Try huge page first
        void* p = ::mmap(nullptr, k2MB,
                         PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB,
                         -1, 0);
        if (p != MAP_FAILED) {
            allAllocations_.push_back(static_cast<uint8_t*>(p));
            poolTotal_++;
            return static_cast<uint8_t*>(p);
        }

        // Fallback: regular mmap, over-allocate for alignment
        size_t allocSize = k2MB + k2MB;  // extra for alignment
        p = ::mmap(nullptr, allocSize,
                   PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS,
                   -1, 0);
        if (p == MAP_FAILED) return nullptr;

        auto* base = static_cast<uint8_t*>(p);
        auto* aligned = reinterpret_cast<uint8_t*>(
            (reinterpret_cast<uintptr_t>(base) + k2MB - 1) & ~(k2MB - 1));

        // Trim the excess before and after
        size_t headWaste = static_cast<size_t>(aligned - base);
        if (headWaste > 0) ::munmap(base, headWaste);
        size_t tailWaste = allocSize - headWaste - k2MB;
        if (tailWaste > 0) ::munmap(aligned + k2MB, tailWaste);

        allAllocations_.push_back(aligned);
        poolTotal_++;
        return aligned;
#else
        // Non-Linux: use aligned_alloc
        void* p = std::aligned_alloc(k2MB, k2MB);
        if (!p) return nullptr;
        allAllocations_.push_back(static_cast<uint8_t*>(p));
        poolTotal_++;
        return static_cast<uint8_t*>(p);
#endif
    }

    void freeSingle2MB([[maybe_unused]] uint8_t* ptr)
    {
        if (!ptr) return;
        if (hugePageMode_) return;  // freed as bulk in destructor
#ifdef __linux__
        ::munmap(ptr, k2MB);
#else
        std::free(ptr);
#endif
    }

    mutable std::mutex mutex_;
    std::vector<uint8_t*> freeList_;
    std::vector<uint8_t*> allAllocations_;  // for cleanup of overflow allocs
    uint8_t* hugePoolBase_ = nullptr;
    size_t hugePoolBytes_ = 0;
    size_t poolTotal_ = 0;
    bool hugePageMode_ = false;
};

// RAII wrapper for a 2MB buffer from the huge page pool.
// Use as the storage backing for ChunkData::bytes replacement.
struct HugePageBuffer {
    uint8_t* ptr = nullptr;
    size_t size = 0;

    HugePageBuffer() = default;

    explicit HugePageBuffer(size_t sz)
        : ptr(HugePageAllocator::instance().allocate2MB()),
          size(sz) {}

    ~HugePageBuffer()
    {
        if (ptr) HugePageAllocator::instance().free2MB(ptr);
    }

    HugePageBuffer(HugePageBuffer&& o) noexcept : ptr(o.ptr), size(o.size)
    {
        o.ptr = nullptr;
        o.size = 0;
    }

    HugePageBuffer& operator=(HugePageBuffer&& o) noexcept
    {
        if (this != &o) {
            if (ptr) HugePageAllocator::instance().free2MB(ptr);
            ptr = o.ptr;
            size = o.size;
            o.ptr = nullptr;
            o.size = 0;
        }
        return *this;
    }

    HugePageBuffer(const HugePageBuffer&) = delete;
    HugePageBuffer& operator=(const HugePageBuffer&) = delete;

    uint8_t* data() noexcept { return ptr; }
    const uint8_t* data() const noexcept { return ptr; }
    bool empty() const noexcept { return size == 0; }

    void resize(size_t newSize)
    {
        if (!ptr) ptr = HugePageAllocator::instance().allocate2MB();
        size = newSize;
    }
};

}  // namespace vc::cache
