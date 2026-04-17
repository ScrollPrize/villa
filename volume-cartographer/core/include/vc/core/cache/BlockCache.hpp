#pragma once

#include <array>
#include <atomic>
#include <bit>
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
    void put(const BlockKey& key, const uint8_t* src) noexcept;

    // Scoped batch-put: take the unique_lock once, call put() many times,
    // release on destruction. Eliminates 512 lock/unlock pairs per 128³
    // chunk insert in the sampler hot path.
    class BatchPut {
    public:
        explicit BatchPut(BlockCache& cache) noexcept
            : cache_(cache), lock_(cache.arenaMutex_) {}
        BatchPut(const BatchPut&) = delete;
        BatchPut& operator=(const BatchPut&) = delete;
        void put(const BlockKey& key, const uint8_t* src) noexcept;

        // Reserve an arena slot for `key` without copying. Caller writes
        // exactly kBlockBytes into the returned 16-byte-aligned buffer.
        // Skips the src→tmp→arena double copy that put() performs — use
        // this when the producer can assemble the block directly at its
        // final destination.
        [[nodiscard]] uint8_t* acquire(const BlockKey& key) noexcept;
    private:
        BlockCache& cache_;
        std::unique_lock<std::shared_mutex> lock_;
    };

    [[nodiscard]] size_t capacity() const noexcept { return nSlots_; }
    [[nodiscard]] size_t size() const noexcept;

    // Sum of per-shard hit counters. Relaxed read — stats are diagnostic,
    // not load-bearing, so tearing across shards is fine.
    [[nodiscard]] uint64_t blockHits() const noexcept;

    // Monotonic counter bumped on every slot reclaim (clock-sweep eviction)
    // and on clear(). Callers that cache "nothing has changed since last
    // check" decisions (e.g. fetchInteractive's dedup) read this to detect
    // evictions without needing the cache mutex.
    [[nodiscard]] uint64_t evictionVersion() const noexcept {
        return evictionVersion_.load(std::memory_order_relaxed);
    }

    void clear();

private:
    // Body of put()/BatchPut::put — assumes unique_lock on arenaMutex_ is held.
    void putLocked(const BlockKey& key, const uint8_t* src) noexcept;
    // Reserve a slot (overwrite if key exists, else allocate/reclaim). Returns
    // SIZE_MAX iff nSlots_==0. Updates shard map/slotKey_/occupancy.
    [[nodiscard]] size_t acquireSlotLocked(const BlockKey& key) noexcept;
    [[nodiscard]] size_t reclaimSlotLocked();

    Config config_;
    size_t nSlots_ = 0;

    // Contiguous mmap'd arena of Block objects. Virtual region is sized at
    // startup; a background thread pre-faults pages in 1 GB increments via
    // madvise(MADV_POPULATE_WRITE) so first-touch page faults don't stall
    // the render thread as the cache fills.
    Block* arena_ = nullptr;
    size_t arenaBytes_ = 0;
    std::jthread prefaultThread_;

    // Sharded reader lock: the map is split into N shards by hash, each with
    // its own shared_mutex. get()/contains() lock only the relevant shard,
    // so 12 concurrent render threads rarely collide on the same cache line.
    // Previously a single shared_mutex served every reader — even with
    // multiple readers allowed in parallel, the CAS-based reader counter on
    // that lock's cache line saturated with atomic traffic under hot render
    // (~54% of CPU was in pthread_rwlock_rd{lock,unlock} atomics).
    //
    // Writers (put/acquire/reclaim/clear) hold `arenaMutex_` exclusively for
    // the arena bookkeeping (slotKey_, occupiedBits_, clockHand_, levelOccupied_,
    // occupiedCount_) and take the relevant shard's unique_lock briefly to
    // insert/erase its map entry. The nested order is always arenaMutex_
    // before shard — readers never take arenaMutex_, so no deadlock path.
    static constexpr size_t kShards = 32;  // power of 2
    static_assert(std::has_single_bit(kShards), "kShards must be a power of 2");
    // alignas(64): each shard owns one cacheline's worth of frequently-written
    // state (mutex + hit counter) so shards don't false-share. Previously a
    // single global statBlockHits_.fetch_add on every get() hit cost ~12% of
    // total CPU under 12-thread render — the atomic traffic ping-ponged one
    // cacheline across all cores. Per-shard counters eliminate that.
    struct alignas(64) MapShard {
        mutable std::shared_mutex mutex;
        std::unordered_map<BlockKey, size_t, BlockKeyHash> map;
        std::atomic<uint64_t> hits{0};
    };
    std::array<MapShard, kShards> shards_;
    static size_t shardIndex(const BlockKey& k) noexcept {
        return BlockKeyHash{}(k) & (kShards - 1);
    }

    // arenaMutex_: protects arena bookkeeping (slotKey_, occupiedBits_,
    // clockHand_, occupiedCount_, levelOccupied_). Readers do NOT take this
    // lock — they only touch shards_[i].mutex. Write paths take arenaMutex_
    // exclusively, then take the relevant shard lock for map updates.
    mutable std::shared_mutex arenaMutex_;

    std::vector<BlockKey> slotKey_;
    // Parallel bitmasks (1 bit per slot): "occupied" (has valid key) and
    // "used" (clock-sweep NRU flag). Packs 2.5M slots into 310 KB each
    // vs. ~2.5 MB for a byte-per-slot vector. occupiedBits_ is guarded by
    // arenaMutex_; usedBits_ is atomic per word so get() can set it lock-free.
    std::vector<uint64_t> occupiedBits_;
    std::unique_ptr<std::atomic<uint64_t>[]> usedBits_;
    size_t usedBitsWords_ = 0;
    size_t occupiedCount_ = 0;
    size_t clockHand_ = 0;

    // Occupancy and floor per pyramid level. Blocks at a level with
    // occupancy <= floor are protected from the clock sweep.
    std::array<size_t, kMaxLevels> levelOccupied_{};
    std::array<size_t, kMaxLevels> levelFloor_{};

    // Bumped every time a slot is reclaimed (eviction) or the whole cache
    // is cleared. Readers use this to invalidate "last-seen" caches
    // (fetchInteractive dedup) without needing any cache lock. Relaxed atomic —
    // we only need monotonicity, not ordering against the slot writes.
    std::atomic<uint64_t> evictionVersion_{0};

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
