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

// Single-tier block cache. Clock-sweep NRU eviction over a fixed arena.
// get() / put() yield std::shared_ptr<Block>; a block dropped from the
// arena while a sampler still holds a pointer stays alive in memory until
// the last reference is released.
class BlockCache {
public:
    struct Config {
        size_t bytes = 10ULL << 30;   // 10 GiB default
    };

    explicit BlockCache(Config cfg);
    ~BlockCache();

    BlockCache(const BlockCache&) = delete;
    BlockCache& operator=(const BlockCache&) = delete;

    // Lookup. Returns null if not cached. On hit, marks the block recently used.
    [[nodiscard]] BlockPtr get(const BlockKey& key) noexcept;

    // Insert, copying kBlockBytes from `src`. Evicts NRU entries if full.
    void put(const BlockKey& key, const uint8_t* src);

    [[nodiscard]] size_t capacity() const noexcept { return nSlots_; }
    [[nodiscard]] size_t size() const noexcept;

    void clear();

private:
    [[nodiscard]] size_t reclaimSlotLocked();

    Config config_;
    size_t nSlots_ = 0;

    mutable std::mutex mutex_;
    std::unordered_map<BlockKey, BlockPtr, BlockKeyHash> map_;

    std::vector<BlockPtr> slotBlock_;
    std::vector<BlockKey> slotKey_;
    std::vector<uint8_t> occupied_;
    size_t occupiedCount_ = 0;
    size_t clockHand_ = 0;
};

}  // namespace vc::cache
