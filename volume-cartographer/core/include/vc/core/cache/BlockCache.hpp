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
    int bz = 0;  // block index along z (16-voxel units at this level)
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

// Block-granularity cache for voxel data. Two regions:
//   - Resident: always-in-memory blocks (e.g. coarsest level). Never evicted.
//   - Evictable: arena of fixed-size block slots; NRU clock-sweep eviction
//     when full. Hashmap `BlockKey → Block*` for lookup.
//
// The evictable arena is sized at construction. Lookups return raw pointers
// whose lifetime extends until the block is reclaimed by a sweep — the caller
// must not retain pointers across cache-mutating operations.
class BlockCache {
public:
    struct Config {
        size_t evictableBytes = 10ULL << 30;   // 10 GiB default
    };

    explicit BlockCache(Config cfg);
    ~BlockCache();

    BlockCache(const BlockCache&) = delete;
    BlockCache& operator=(const BlockCache&) = delete;

    // Lookup a block. Returns nullptr if not present. On hit, sets the NRU
    // `used` flag so the block survives the next eviction sweep.
    [[nodiscard]] const Block* get(const BlockKey& key) noexcept;

    // Insert a block, copying kBlockBytes from `src`. If the key already
    // exists the existing entry is overwritten. For evictable insertions,
    // evicts NRU entries if the arena is full.
    void put(const BlockKey& key, const uint8_t* src);

    // Insert into the always-resident region (grows on demand, not evicted).
    void putResident(const BlockKey& key, const uint8_t* src);

    [[nodiscard]] size_t evictableSlots() const noexcept { return nSlots_; }
    [[nodiscard]] size_t residentSlots() const noexcept;
    [[nodiscard]] size_t evictableUsed() const noexcept;

    void clearEvictable();

private:
    // Clock-sweep eviction. Advances the hand until a slot with used==0 is
    // found, clearing used bits as it passes. Returns the reclaimed slot.
    [[nodiscard]] Block* reclaimOneLocked();

    Config config_;

    mutable std::mutex mutex_;
    std::unordered_map<BlockKey, Block*, BlockKeyHash> evictableMap_;
    std::unordered_map<BlockKey, Block*, BlockKeyHash> residentMap_;

    // Evictable arena: contiguous slots, each mapped into evictableMap_ when
    // populated. Reverse index `slotKeys_[i]` tells us which key currently
    // owns slot `i` (or a sentinel if empty).
    std::unique_ptr<Block[]> slots_;
    size_t nSlots_ = 0;
    std::vector<BlockKey> slotKeys_;
    std::vector<uint8_t> slotOccupied_;  // 0 = empty, 1 = holds a block
    size_t occupiedCount_ = 0;
    size_t clockHand_ = 0;

    // Resident arena grows on demand; pointers are stable.
    std::vector<std::unique_ptr<Block>> residentArena_;
};

}  // namespace vc::cache
