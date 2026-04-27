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
    //
    // FastRwLock: single 64-bit atomic (reader count in low 32, writer bit
    // in high bit). Each read_lock/unlock is ONE LSE atomic op (LDADDAL /
    // LDADDL) — ~30-50 cycles uncontended vs. glibc pthread_rwlock which
    // does several atomic ops per acquire for futex-wakeup bookkeeping.
    // Under the 12-thread render workload the uncontended cost savings
    // dominated: pthread_rwlock was ~44% of CPU on heavy composite
    // frames; this cuts that to just the raw atomic RMW cost.
    struct alignas(64) FastRwLock {
        static constexpr uint64_t kWriterBit = 1ull << 32;
        static constexpr uint64_t kReaderMask = 0xFFFFFFFFull;
        mutable std::atomic<uint64_t> state{0};

        void lock_shared() const noexcept {
            uint64_t prev = state.fetch_add(1, std::memory_order_acquire);
            if (!(prev & kWriterBit)) return;  // uncontended fast path
            // Writer holds the lock — undo our reader and wait.
            state.fetch_sub(1, std::memory_order_relaxed);
            for (;;) {
                while (state.load(std::memory_order_relaxed) & kWriterBit) {
#if defined(__aarch64__)
                    asm volatile("yield" ::: "memory");
#endif
                }
                prev = state.fetch_add(1, std::memory_order_acquire);
                if (!(prev & kWriterBit)) return;
                state.fetch_sub(1, std::memory_order_relaxed);
            }
        }
        void unlock_shared() const noexcept {
            state.fetch_sub(1, std::memory_order_release);
        }
        void lock() noexcept {
            // Set writer bit. Another writer may race; spin until we're
            // the sole writer. Then wait for existing readers to drain.
            for (;;) {
                uint64_t prev = state.fetch_or(kWriterBit, std::memory_order_acquire);
                if (!(prev & kWriterBit)) break;
                while (state.load(std::memory_order_relaxed) & kWriterBit) {
#if defined(__aarch64__)
                    asm volatile("yield" ::: "memory");
#endif
                }
            }
            while ((state.load(std::memory_order_acquire) & kReaderMask) != 0) {
#if defined(__aarch64__)
                asm volatile("yield" ::: "memory");
#endif
            }
        }
        void unlock() noexcept {
            state.fetch_and(~kWriterBit, std::memory_order_release);
        }
    };

    // Per-shard lock-free open-addressing hash table. Writers are still
    // serialized by arenaMutex_ (single writer globally), so the insert /
    // erase paths just atomic-store entries with release semantics.
    // Readers probe lock-free with acquire-loads and verify the returned
    // slot via slotKeyPacked_ — that verify catches the ~1/2^32 hash-tag
    // collisions as well as racing writer modifications.
    //
    // Replaces the prior FastRwLock + std::unordered_map: reads take zero
    // locks now, just a couple of atomic loads per probe step. On heavy
    // composite workloads with ~25% L2-miss rate, every miss previously
    // went through the shard rwlock (~40-70 cycles per pair uncontended,
    // much worse under the 12-thread coherence storm). This path has no
    // rwlock tax at all.
    //
    // Entry layout (64 bits):
    //   [63:62] state: 00 empty, 10 occupied, 01 tombstone
    //   [61:32] arena slot index (30 bits — supports 1B slots, we use ≤3M)
    //   [31: 0] 32-bit hash of the BlockKey
    // Empty is all-zero; ctor memset suffices.
    static constexpr uint64_t kEntryEmpty = 0ull;
    static constexpr uint64_t kEntryOccupied = 0x8000000000000000ull;
    static constexpr uint64_t kEntryTombstone = 0x4000000000000000ull;
    static constexpr uint64_t kEntryStateMask = 0xC000000000000000ull;
    static constexpr uint64_t kEntrySlotMask = 0x3FFFFFFF00000000ull;
    static constexpr int kEntrySlotShift = 32;
    static constexpr uint64_t kEntryHashMask = 0x00000000FFFFFFFFull;
    static uint32_t shardMapHash(const BlockKey& k) noexcept {
        // Different mixer from BlockKeyHash + l2Index so shard-map slots
        // don't cluster with either of those.
        uint64_t h = packBlockKey(k) * 0xD6E8FEB86659FD93ull;
        h ^= h >> 32;
        uint32_t r = uint32_t(h);
        return r == 0 ? 1u : r;  // reserve 0 for "empty-signifier"; shift anything that lands there
    }
    static uint64_t makeOccupiedEntry(uint32_t slot, uint32_t hash) noexcept {
        return kEntryOccupied | (uint64_t(slot) << kEntrySlotShift) | uint64_t(hash);
    }

    // 2^18 = 256K slots per shard × 8 B = 2 MB per shard × 32 shards = 64 MB.
    // With max arena ~2.5M slots / 32 shards ≈ 80K entries per shard, load
    // factor peaks at ~0.3 → short probe chains, fast lookups.
    static constexpr size_t kShardMapBits = 18;
    static constexpr size_t kShardMapSize = size_t(1) << kShardMapBits;
    static constexpr size_t kShardMapMask = kShardMapSize - 1;

    struct alignas(64) MapShard {
        std::unique_ptr<std::atomic<uint64_t>[]> table;
        std::atomic<uint64_t> hits{0};
    };
    std::array<MapShard, kShards> shards_;
    static size_t shardIndex(const BlockKey& k) noexcept {
        return BlockKeyHash{}(k) & (kShards - 1);
    }

    // arenaMutex_: protects arena bookkeeping (slotKeyPacked_, occupiedBits_,
    // clockHand_, occupiedCount_, levelOccupied_). Readers do NOT take this
    // lock — they only touch shards_[i].mutex (slow path) or l2_ (fast path).
    // Write paths take arenaMutex_ exclusively, then take the relevant shard
    // lock for map updates.
    mutable std::shared_mutex arenaMutex_;

    // Pack BlockKey into 64 bits: [level:4][bz:20][by:20][bx:20]. Covers
    // any realistic volume (2^20 blocks × 16 voxels = 16.7M voxels per axis)
    // and all level ids we use. The all-ones pattern (UINT64_MAX) maps to
    // kEmptyKey {-1,-1,-1,-1}, so it doubles as the "empty slot" sentinel.
    static uint64_t packBlockKey(const BlockKey& k) noexcept {
        return (uint64_t(uint32_t(k.level) & 0xFu)     << 60)
             | (uint64_t(uint32_t(k.bz)    & 0xFFFFFu) << 40)
             | (uint64_t(uint32_t(k.by)    & 0xFFFFFu) << 20)
             |  uint64_t(uint32_t(k.bx)    & 0xFFFFFu);
    }
    static BlockKey unpackBlockKey(uint64_t p) noexcept {
        auto sx20 = [](uint32_t v) -> int {
            return int(v & 0x80000u ? (v | 0xFFF00000u) : v);
        };
        auto sx4 = [](uint32_t v) -> int {
            return int(v & 0x8u ? (v | 0xFFFFFFF0u) : v);
        };
        BlockKey k;
        k.level = sx4(uint32_t((p >> 60) & 0xFu));
        k.bz    = sx20(uint32_t((p >> 40) & 0xFFFFFu));
        k.by    = sx20(uint32_t((p >> 20) & 0xFFFFFu));
        k.bx    = sx20(uint32_t( p        & 0xFFFFFu));
        return k;
    }
    // 32-bit key tag for the L2 direct-mapped verify check. Uses a different
    // multiplier from the L2 index hash so colliding L2 slots don't also
    // collide tags.
    static uint32_t keyTag32(uint64_t packed) noexcept {
        uint64_t h = packed * 0x9E3779B97F4A7C15ULL;
        h ^= h >> 32;
        return uint32_t(h);
    }
    static size_t l2Index(uint64_t packed, size_t bits) noexcept {
        uint64_t h = packed * 0xBF58476D1CE4E5B9ULL;
        return size_t(h >> (64 - bits));
    }

    // Lock-free direct-mapped L2 cache sitting in front of the shard maps.
    // Each entry packs [keyTag32:slot32] into one 64-bit atomic. On hit,
    // readers verify via slotKeyPacked_[slot] — if the arena slot no longer
    // holds this key (eviction race or hash collision), we fall through to
    // the slow path. No rwlock on the hot path — get() used to spend ~24% of
    // CPU on pthread_rwlock_rdlock/unlock atomics (LDADD4_acq + CAS4_rel on
    // aarch64); this L2 replaces that with a single relaxed atomic load +
    // one verify load to the same cacheline.
    // 8-way set-associative L2: 2^17 = 128K sets × 8 entries = 1M entries
    // × 8 bytes = 8 MB (fits in L3). Each set occupies exactly one 64-byte
    // cacheline, so a lookup loads one line and compares 8 tags — vs. the
    // prior direct-mapped 1M × 1-way which, on heavy composite workloads
    // with ~500K unique blocks, had ~25% collision rate. With 8 ways,
    // collision rate drops by ~500× (Poisson: P(set load > 8) ≈ 5e-4 at
    // mean load 4). Slow-path rwlock traffic (was ~40% of CPU in
    // pthread_rwlock on heavy Max-composite frames) should fall by the
    // same factor.
    static constexpr size_t kL2Bits = 17;
    static constexpr size_t kL2Ways = 8;
    static constexpr size_t kL2Sets = size_t(1) << kL2Bits;
    static constexpr size_t kL2Size = kL2Sets * kL2Ways;
    static constexpr uint64_t kL2Empty = 0;  // keyTag32 returns 0 only for packed==0
    std::unique_ptr<std::atomic<uint64_t>[]> l2_;
    // Round-robin eviction counter per set (non-atomic; races are benign
    // — worst case we overwrite slightly less-ideal slots). 128K bytes
    // fits in L2D.
    std::unique_ptr<uint8_t[]> l2RrCounters_;

    // Per-slot packed BlockKey, readable lock-free by the L2 verify path.
    // Written under arenaMutex_ only. UINT64_MAX means "empty slot".
    std::unique_ptr<std::atomic<uint64_t>[]> slotKeyPacked_;
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
        auto& word = usedBits_[bitWord(i)];
        const uint64_t m = bitMask(i);
        if (v) {
            // Short-circuit if already set. A relaxed load is an ordinary
            // LDR with no coherence round-trip; fetch_or is LDSET which
            // takes ~20-50 cycles uncontended and more with N-thread
            // contention on the same 64-slot word. Under 12-thread render,
            // the SAME word is hit by many L2 hits per frame because
            // adjacent slot indices share a bit-word — eliminating the
            // redundant LDSET traffic was visible as ~30% of CPU in
            // post-L2-hit code (the LDSET was the actual hot atomic, with
            // sample skid making the mov/ldr that followed look worse).
            if ((word.load(std::memory_order_relaxed) & m) == 0)
                word.fetch_or(m, std::memory_order_relaxed);
        } else {
            word.fetch_and(~m, std::memory_order_relaxed);
        }
    }
};

}  // namespace vc::cache
