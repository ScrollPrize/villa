#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <utils/hash.hpp>

namespace vc::cache {

// Fixed block geometry: 16 x 16 x 16 uint8 voxels = 4096 bytes per block.
// All allocation and eviction happens at this granularity for every level.
constexpr int kBlockSize = 16;
constexpr size_t kBlockBytes = size_t(kBlockSize) * kBlockSize * kBlockSize;

struct BlockKey {
    int level = 0;
    int bz = 0;
    int by = 0;
    int bx = 0;

    constexpr bool operator==(const BlockKey& o) const noexcept = default;
};

struct BlockKeyHash {
    size_t operator()(const BlockKey& k) const noexcept
    {
        return utils::hash_combine_values(k.level, k.bz, k.by, k.bx);
    }
};

struct alignas(64) Block {
    uint8_t data[kBlockBytes];
    std::atomic<uint8_t> used{1};  // clock-sweep NRU flag
};

using BlockPtr = std::shared_ptr<Block>;

// Block-granularity cache for voxel data.
//
// Two regions:
//   - Resident: always-in-memory blocks (e.g. coarsest level). Never evicted.
//   - Evictable: bounded-capacity clock-sweep NRU arena.
//
// get() / put() yield std::shared_ptr<Block>. A block dropped from the
// eviction arena while a sampler still holds a shared_ptr stays alive in
// memory until the last reference is released.
class BlockCache {
public:
    struct Config {
        size_t evictableBytes = 10ULL << 30;   // 10 GiB default
    };

    explicit BlockCache(Config cfg);
    ~BlockCache();

    BlockCache(const BlockCache&) = delete;
    BlockCache& operator=(const BlockCache&) = delete;

    // Lookup. Returns null if not cached. On evictable-region hit, marks the
    // block as recently used.
    [[nodiscard]] BlockPtr get(const BlockKey& key) noexcept;

    // Insert into the evictable region, copying kBlockBytes from `src`.
    // Evicts NRU entries if the arena is full; evicted blocks remain alive
    // while outstanding shared_ptrs reference them.
    void put(const BlockKey& key, const uint8_t* src);

    // Insert into the resident region (grows on demand, never evicted).
    void putResident(const BlockKey& key, const uint8_t* src);

    [[nodiscard]] size_t capacity() const noexcept { return nSlots_; }
    [[nodiscard]] size_t residentSize() const noexcept;
    [[nodiscard]] size_t evictableSize() const noexcept;

    void clearEvictable();

private:
    [[nodiscard]] size_t reclaimSlotLocked();

    Config config_;
    size_t nSlots_ = 0;

    mutable std::mutex mutex_;
    std::unordered_map<BlockKey, BlockPtr, BlockKeyHash> evictableMap_;
    std::unordered_map<BlockKey, BlockPtr, BlockKeyHash> residentMap_;

    // Parallel arrays sized to nSlots_. slot i is occupied iff occupied_[i].
    std::vector<BlockPtr> slotBlock_;
    std::vector<BlockKey> slotKey_;
    std::vector<uint8_t> occupied_;
    size_t occupiedCount_ = 0;
    size_t clockHand_ = 0;
};

}  // namespace vc::cache
