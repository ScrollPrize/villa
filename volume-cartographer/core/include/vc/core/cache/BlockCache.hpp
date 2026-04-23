#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>
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
        // Pack into two 64-bit words and mix with a single prime multiply.
        // Avoids the chained-XOR bias of the older boost-style combine,
        // which clustered spatially-adjacent keys onto nearby buckets.
        const uint64_t hi = (uint64_t(uint32_t(k.level)) << 40)
                          ^ (uint64_t(uint32_t(k.bz)) << 20)
                          ^  uint64_t(uint32_t(k.by));
        const uint64_t lo = (uint64_t(uint32_t(k.by)) << 32)
                          |  uint64_t(uint32_t(k.bx));
        return size_t(hi ^ (lo * 0x9E3779B97F4A7C15ULL));
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

    // Peek: like get() but does not touch the recently-used bit. For
    // "is this resident?" queries that shouldn't affect eviction order.
    [[nodiscard]] bool contains(const BlockKey& key) const noexcept;

    // Batch peek: one lock acquisition for N lookups. Used by the
    // fetchInteractive triage path, which checks ~hundreds-of-thousands
    // of keys per second during viewport changes.
    void containsBatch(const std::vector<BlockKey>& keys,
                       std::vector<uint8_t>& out) const;

    // Insert, copying kBlockBytes from `src`. Evicts NRU entries if full.
    // `gen` is the cache generation the caller was issued; if it no longer
    // matches (volume was switched, cache cleared), the put is a no-op.
    void put(const BlockKey& key, const uint8_t* src, uint64_t gen) noexcept;

    // Scoped batch-put: take the unique_lock once, call put() many times,
    // release on destruction. Eliminates 512 lock/unlock pairs per 128³
    // chunk insert in the sampler hot path.
    class BatchPut {
    public:
        explicit BatchPut(BlockCache& cache, uint64_t gen) noexcept
            : cache_(cache), gen_(gen), lock_(cache.mutex_) {}
        BatchPut(const BatchPut&) = delete;
        BatchPut& operator=(const BatchPut&) = delete;
        void put(const BlockKey& key, const uint8_t* src) noexcept;
    private:
        BlockCache& cache_;
        uint64_t gen_;
        std::unique_lock<std::shared_mutex> lock_;
    };

    [[nodiscard]] size_t capacity() const noexcept { return nSlots_; }
    [[nodiscard]] size_t size() const noexcept;

    // Returns the current generation. Callers capture this at construction
    // and pass it to put()/BatchPut so stale inserts are rejected.
    [[nodiscard]] uint64_t generation() const noexcept;

    // Clear all entries and bump the generation counter. Subsequent puts
    // with an older generation are silently rejected.
    void clear();

private:
    // Body of put()/BatchPut::put — assumes unique_lock on mutex_ is held.
    // Rejects the insert if gen != generation_.
    void putLocked(const BlockKey& key, const uint8_t* src, uint64_t gen) noexcept;
    [[nodiscard]] size_t reclaimSlotLocked();

    Config config_;
    size_t nSlots_ = 0;
    uint64_t generation_ = 0;  // bumped on clear(); guarded by mutex_

    // Contiguous mmap'd arena of Block objects. Virtual region is sized at
    // startup; a background thread pre-faults pages in 1 GB increments via
    // madvise(MADV_POPULATE_WRITE) so first-touch page faults don't stall
    // the render thread as the cache fills.
    Block* arena_ = nullptr;
    size_t arenaBytes_ = 0;
    std::jthread prefaultThread_;

    // shared_mutex: get()/contains()/size() take a shared lock so the render
    // thread and the 4 worker pools can read concurrently; put()/clear()
    // take the exclusive lock. usedBits_ is atomic so mutations from get()
    // (setUsed on hit) and the clock sweep are lock-free.
    mutable std::shared_mutex mutex_;
    std::unordered_map<BlockKey, size_t, BlockKeyHash> map_;

    std::vector<BlockKey> slotKey_;
    // Parallel bitmasks (1 bit per slot): "occupied" (has valid key) and
    // "used" (clock-sweep NRU flag). Packs 2.5M slots into 310 KB each
    // vs. ~2.5 MB for a byte-per-slot vector. occupiedBits_ is guarded by
    // mutex_; usedBits_ is atomic per word so get() can set it lock-free.
    std::vector<uint64_t> occupiedBits_;
    std::unique_ptr<std::atomic<uint64_t>[]> usedBits_;
    size_t usedBitsWords_ = 0;
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
    bool isUsed(size_t i) const noexcept {
        return (usedBits_[bitWord(i)].load(std::memory_order_relaxed) >> (i & 63u)) & 1u;
    }
    void setUsed(size_t i, bool v) noexcept {
        const uint64_t m = bitMask(i);
        if (v) usedBits_[bitWord(i)].fetch_or(m, std::memory_order_relaxed);
        else   usedBits_[bitWord(i)].fetch_and(~m, std::memory_order_relaxed);
    }
};

}  // namespace vc::cache
