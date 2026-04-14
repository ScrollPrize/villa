#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <utils/hash.hpp>

#include "ChunkKey.hpp"

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

// One block is exactly 4096 voxel bytes — nothing else. Put it in its own
// type so callers see `block->data` rather than raw buffers, but keep the
// struct size == page size so the arena has no per-slot waste.
struct Block {
    uint8_t data[kBlockBytes];
};
static_assert(sizeof(Block) == 4096, "Block must be exactly one 4 KiB page");

// Non-owning pointer into the cache's mmap-backed arena. Eviction is handled
// by overwriting in place; samplers holding a BlockPtr for a slot that has
// since been reused will read the NEW contents, not the old ones (no UAF,
// but 1 frame of stale voxel data is possible).
using BlockPtr = Block*;

// Single-tier block cache with a contiguous mmap-backed arena. Clock-sweep
// NRU eviction over the slot array. On eviction idle we madvise(MADV_DONTNEED)
// so the OS can reclaim physical pages while the virtual mapping persists.
class BlockCache {
public:
    struct Config {
        size_t bytes = 10ULL << 30;   // 10 GiB default

        // Per-level residency floor, in slots. A level's blocks are protected
        // from eviction while that level's occupancy is at or below its
        // floor. Caller must keep the sum of floors well below the total
        // slot count (e.g. <= capacity/2) so the clock sweep can always
        // make progress. Zero means "no protection" for that level.
        std::array<size_t, kMaxLevels> levelFloor{};
    };

    explicit BlockCache(Config cfg);
    ~BlockCache();

    BlockCache(const BlockCache&) = delete;
    BlockCache& operator=(const BlockCache&) = delete;

    // Lookup. Returns null if not cached. On hit, marks the block recently used.
    [[nodiscard]] BlockPtr get(const BlockKey& key) noexcept;

    // Insert, copying kBlockBytes from `src`. Evicts NRU entries if full.
    void put(const BlockKey& key, const uint8_t* src) noexcept;

    [[nodiscard]] size_t capacity() const noexcept { return nSlots_; }
    [[nodiscard]] size_t size() const noexcept;

    void clear();

private:
    [[nodiscard]] size_t reclaimSlotLocked();

    Config config_;
    size_t nSlots_ = 0;

    // Contiguous mmap'd arena of Block objects. Virtual region is sized at
    // startup; physical pages commit only on first touch of each slot.
    Block* arena_ = nullptr;
    size_t arenaBytes_ = 0;

    mutable std::mutex mutex_;
    std::unordered_map<BlockKey, size_t, BlockKeyHash> map_;

    std::vector<BlockKey> slotKey_;
    // Parallel bitmasks (1 bit per slot): "occupied" (has valid key) and
    // "used" (clock-sweep NRU flag). Packs 2.5M slots into 310 KB each
    // vs. ~2.5 MB for a byte-per-slot vector.
    std::vector<uint64_t> occupiedBits_;
    std::vector<uint64_t> usedBits_;
    size_t occupiedCount_ = 0;
    size_t clockHand_ = 0;

    // Occupancy and floor per pyramid level. Blocks at a level with
    // occupancy <= floor are protected from the clock sweep.
    std::array<size_t, kMaxLevels> levelOccupied_{};
    std::array<size_t, kMaxLevels> levelFloor_{};

    static constexpr size_t bitWord(size_t i) noexcept { return i >> 6; }
    static constexpr uint64_t bitMask(size_t i) noexcept { return uint64_t(1) << (i & 63u); }
    bool isOccupied(size_t i) const noexcept { return (occupiedBits_[bitWord(i)] >> (i & 63u)) & 1u; }
    void setOccupied(size_t i, bool v) noexcept {
        if (v) occupiedBits_[bitWord(i)] |= bitMask(i);
        else   occupiedBits_[bitWord(i)] &= ~bitMask(i);
    }
    bool isUsed(size_t i) const noexcept { return (usedBits_[bitWord(i)] >> (i & 63u)) & 1u; }
    void setUsed(size_t i, bool v) noexcept {
        if (v) usedBits_[bitWord(i)] |= bitMask(i);
        else   usedBits_[bitWord(i)] &= ~bitMask(i);
    }
};

}  // namespace vc::cache
