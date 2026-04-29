#include "vc/core/cache/BlockCache.hpp"

#include <bit>
#include <cstring>
#include <new>
#include <stdexcept>

#include <sys/mman.h>

namespace vc::cache {

namespace {
constexpr BlockKey kEmptyKey{-1, -1, -1, -1};
}  // namespace

BlockCache::BlockCache(Config cfg)
    : config_(cfg)
{
    nSlots_ = config_.bytes / sizeof(Block);
    arenaBytes_ = nSlots_ * sizeof(Block);

    levelFloor_ = config_.levelFloor;
    size_t totalFloor = 0;
    for (auto f : levelFloor_) totalFloor += f;
    // Keep at least half the arena unprotected so the clock sweep always has
    // candidates. Oversubscribed floors are clamped proportionally.
    if (nSlots_ && totalFloor > nSlots_ / 2) {
        double scale = double(nSlots_ / 2) / double(totalFloor);
        for (auto& f : levelFloor_) f = size_t(double(f) * scale);
    }

    // Anonymous private mmap: virtual region committed, physical pages arrive
    // on first touch. Kernel manages paging; we can madvise(MADV_DONTNEED) on
    // individual slots to release physical pages back while keeping the
    // virtual mapping stable.
    void* p = ::mmap(nullptr, arenaBytes_,
                     PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
                     -1, 0);
    if (p == MAP_FAILED) {
        throw std::bad_alloc();
    }
    arena_ = static_cast<Block*>(p);

    // Block lookups are spatially scattered — tell kernel not to read-ahead.
    ::madvise(arena_, arenaBytes_, MADV_RANDOM);

    // Promote arena to 2 MiB pages where the kernel can: each block is exactly
    // 4 KiB = 1 standard page, so one huge page covers 512 blocks. TLB reach
    // for a 10 GiB arena jumps from ~2.5M pages to ~5k — render threads that
    // pick blocks across the arena rarely miss the TLB. MADV_HUGEPAGE is
    // advisory, silently ignored when THP is disabled in the kernel.
#ifdef MADV_HUGEPAGE
    ::madvise(arena_, arenaBytes_, MADV_HUGEPAGE);
#endif

    // Pre-fault arena pages in 1 GB increments on a background thread.
    // First-touch page faults are expensive (kernel context switch per 4 KB
    // page); when the cache fills during rendering, thousands of faults stall
    // the decode/insert path. MADV_POPULATE_WRITE (Linux 5.14+) commits the
    // physical pages without touching data, so they're ready when put()
    // memcpys into them.
    if (arenaBytes_ > 0) {
        prefaultThread_ = std::jthread([ptr = reinterpret_cast<uint8_t*>(arena_),
                                        total = arenaBytes_](std::stop_token stop) {
            constexpr size_t kChunk = size_t(1) << 30;  // 1 GB
            for (size_t off = 0; off < total && !stop.stop_requested(); off += kChunk) {
                size_t len = std::min(kChunk, total - off);
                ::madvise(ptr + off, len, MADV_POPULATE_WRITE);
            }
        });
    }

    slotKeyPacked_ = std::unique_ptr<std::atomic<uint64_t>[]>(
        new std::atomic<uint64_t>[nSlots_]);
    for (size_t i = 0; i < nSlots_; ++i)
        slotKeyPacked_[i].store(UINT64_MAX, std::memory_order_relaxed);

    // 8-way set-associative L2; see BlockCache.hpp for sizing rationale.
    l2_ = std::unique_ptr<std::atomic<uint64_t>[]>(
        new std::atomic<uint64_t>[kL2Size]);
    for (size_t i = 0; i < kL2Size; ++i)
        l2_[i].store(kL2Empty, std::memory_order_relaxed);
    l2RrCounters_ = std::unique_ptr<uint8_t[]>(new uint8_t[kL2Sets]());

    const size_t words = (nSlots_ + 63u) / 64u;
    occupiedBits_.assign(words, 0);
    usedBitsWords_ = words;
    usedBits_ = std::unique_ptr<std::atomic<uint64_t>[]>(
        new std::atomic<uint64_t>[words]());
    // Per-shard lock-free hash table. Allocate zero-initialized.
    for (auto& s : shards_) {
        s.table = std::unique_ptr<std::atomic<uint64_t>[]>(
            new std::atomic<uint64_t>[kShardMapSize]());
    }
}

BlockCache::~BlockCache()
{
    // Stop the background prefault thread BEFORE unmapping the arena it's
    // madvise()-ing. jthread dtor requests stop + joins, so after this
    // line no madvise calls are in flight.
    if (prefaultThread_.joinable())
        prefaultThread_ = {};  // request_stop + join via dtor of old thread
    if (arena_) {
        ::munmap(arena_, arenaBytes_);
        arena_ = nullptr;
    }
}

BlockPtr BlockCache::get(const BlockKey& key) noexcept
{
    // 8-way set-associative L2 fast path: probe 8 entries in one cacheline
    // for a matching tag, verify via slotKeyPacked_. Lock-free throughout.
    // The verify check still catches keyTag32 collisions (1/2^32) and
    // eviction/reassignment races.
    const uint64_t packed = packBlockKey(key);
    const uint32_t tag = keyTag32(packed);
    {
        const size_t setIdx = l2Index(packed, kL2Bits);
        std::atomic<uint64_t>* set = &l2_[setIdx * kL2Ways];
        for (size_t way = 0; way < kL2Ways; ++way) {
            const uint64_t entry = set[way].load(std::memory_order_acquire);
            if (uint32_t(entry >> 32) != tag) continue;
            const uint32_t slot = uint32_t(entry);
            if (slot < nSlots_ &&
                slotKeyPacked_[slot].load(std::memory_order_acquire) == packed) {
                setUsed(slot, true);
                return &arena_[slot];
            }
        }
    }

    // Slow path: probe the shard's lock-free hash table. Each entry is an
    // atomic<uint64_t>; an acquire-load tells us empty / occupied / tombstone
    // + a 32-bit hash + the arena slot. Empty == end of probe chain.
    // On hash match, verify via slotKeyPacked_ — a match means the arena
    // slot still holds this key (catches both 1/2^32 hash collisions and
    // races against writers that moved blocks around).
    const size_t sh = shardIndex(key);
    MapShard& shard = shards_[sh];
    const uint32_t mh = shardMapHash(key);
    size_t idx = mh & kShardMapMask;
    for (size_t probe = 0; probe < kShardMapSize; ++probe) {
        const uint64_t e = shard.table[(idx + probe) & kShardMapMask]
                                .load(std::memory_order_acquire);
        if (e == kEntryEmpty) break;  // not in table
        if ((e & kEntryStateMask) == kEntryOccupied
            && uint32_t(e & kEntryHashMask) == mh) {
            const uint32_t slot = uint32_t((e & kEntrySlotMask) >> kEntrySlotShift);
            if (slot < nSlots_
                && slotKeyPacked_[slot].load(std::memory_order_acquire) == packed) {
                setUsed(slot, true);
                shard.hits.fetch_add(1, std::memory_order_relaxed);
                {
                    const size_t setIdx = l2Index(packed, kL2Bits);
                    std::atomic<uint64_t>* set = &l2_[setIdx * kL2Ways];
                    size_t wayToUse = kL2Ways;
                    for (size_t way = 0; way < kL2Ways; ++way) {
                        if (set[way].load(std::memory_order_relaxed) == kL2Empty) {
                            wayToUse = way; break;
                        }
                    }
                    if (wayToUse == kL2Ways) {
                        const uint8_t c = l2RrCounters_[setIdx];
                        l2RrCounters_[setIdx] = uint8_t((c + 1u) & (kL2Ways - 1u));
                        wayToUse = c & (kL2Ways - 1u);
                    }
                    set[wayToUse].store((uint64_t(tag) << 32) | uint32_t(slot),
                                        std::memory_order_release);
                }
                return &arena_[slot];
            }
            // Hash matched but slot's packed-key doesn't — keep probing
            // for a deeper colliding entry (rare).
        }
        // tombstone or hash-mismatch: keep probing.
    }
    return nullptr;
}

uint64_t BlockCache::blockHits() const noexcept
{
    uint64_t total = 0;
    for (const auto& s : shards_)
        total += s.hits.load(std::memory_order_relaxed);
    return total;
}

bool BlockCache::contains(const BlockKey& key) const noexcept
{
    const uint64_t packed = packBlockKey(key);
    const size_t sh = shardIndex(key);
    const MapShard& shard = shards_[sh];
    const uint32_t mh = shardMapHash(key);
    size_t idx = mh & kShardMapMask;
    for (size_t probe = 0; probe < kShardMapSize; ++probe) {
        const uint64_t e = shard.table[(idx + probe) & kShardMapMask]
                                .load(std::memory_order_acquire);
        if (e == kEntryEmpty) return false;
        if ((e & kEntryStateMask) == kEntryOccupied
            && uint32_t(e & kEntryHashMask) == mh) {
            const uint32_t slot = uint32_t((e & kEntrySlotMask) >> kEntrySlotShift);
            if (slot < nSlots_
                && slotKeyPacked_[slot].load(std::memory_order_acquire) == packed) {
                return true;
            }
        }
    }
    return false;
}

void BlockCache::containsBatch(const std::vector<BlockKey>& keys,
                               std::vector<uint8_t>& out) const
{
    out.assign(keys.size(), 0);
    if (keys.empty()) return;
    // Lock-free probe path — no grouping needed since reads don't take
    // any lock now. We just delegate to contains() per key.
    for (size_t i = 0; i < keys.size(); ++i) {
        if (contains(keys[i])) out[i] = 1;
    }
}

void BlockCache::put(const BlockKey& key, const uint8_t* src, uint64_t gen) noexcept
{
    std::unique_lock lock(arenaMutex_);
    putLocked(key, src, gen);
}

void BlockCache::BatchPut::put(const BlockKey& key, const uint8_t* src) noexcept
{
    cache_.putLocked(key, src, gen_);
}

uint8_t* BlockCache::BatchPut::acquire(const BlockKey& key) noexcept
{
    size_t slot = cache_.acquireSlotLocked(key, gen_);
    if (slot == SIZE_MAX) return nullptr;
    return cache_.arena_[slot].data;
}

size_t BlockCache::acquireSlotLocked(const BlockKey& key, uint64_t gen) noexcept
{
    if (gen != generation_) return SIZE_MAX;
    // Degenerate config (cache size rounded below one block) — bail
    // instead of dividing by zero in the slot-assignment arithmetic.
    if (nSlots_ == 0) return SIZE_MAX;

    const size_t sh = shardIndex(key);
    MapShard& shard = shards_[sh];
    const uint64_t packedKey = packBlockKey(key);
    const uint32_t mh = shardMapHash(key);
    // Probe the shard's lock-free table for an existing entry. Writers are
    // serialized via arenaMutex_ so no writer-writer contention.
    {
        size_t idx = mh & kShardMapMask;
        for (size_t probe = 0; probe < kShardMapSize; ++probe) {
            const uint64_t e = shard.table[(idx + probe) & kShardMapMask]
                                    .load(std::memory_order_acquire);
            if (e == kEntryEmpty) break;
            if ((e & kEntryStateMask) == kEntryOccupied
                && uint32_t(e & kEntryHashMask) == mh) {
                const uint32_t existing = uint32_t((e & kEntrySlotMask) >> kEntrySlotShift);
                if (existing < nSlots_
                    && slotKeyPacked_[existing].load(std::memory_order_acquire) == packedKey) {
                    setUsed(existing, true);
                    return existing;
                }
            }
        }
    }

    size_t slot;
    if (occupiedCount_ < nSlots_) {
        // Find first unoccupied slot starting at clockHand_. Scan 64 slots
        // at a time via the occupancy bitmap: pick the first 64-bit word
        // whose complement has a set bit, then ctz the position. O(nSlots_/64)
        // worst case, O(1) when sparse — vs. O(nSlots_) scalar while-loop.
        const size_t words = occupiedBits_.size();
        const size_t startWord = clockHand_ / 64u;
        const size_t startBit = clockHand_ % 64u;
        auto findFree = [&](size_t w) -> std::pair<bool, size_t> {
            uint64_t mask = ~occupiedBits_[w];
            if (w == startWord && startBit > 0) {
                mask &= ~((uint64_t(1) << startBit) - 1);
            }
            if (mask) {
                size_t bit = static_cast<size_t>(std::countr_zero(mask));
                size_t s = w * 64u + bit;
                if (s < nSlots_) return {true, s};
            }
            return {false, 0};
        };
        bool found = false;
        for (size_t w = startWord; w < words; ++w) {
            auto [ok, s] = findFree(w);
            if (ok) { slot = s; found = true; break; }
        }
        if (!found) {
            // Wrap.
            for (size_t w = 0; w <= startWord; ++w) {
                uint64_t mask = ~occupiedBits_[w];
                if (w == startWord) {
                    mask &= (startBit == 0) ? 0 : ((uint64_t(1) << startBit) - 1);
                }
                if (mask) {
                    size_t bit = static_cast<size_t>(std::countr_zero(mask));
                    size_t s = w * 64u + bit;
                    if (s < nSlots_) { slot = s; found = true; break; }
                }
            }
        }
        if (!found) { slot = reclaimSlotLocked(); }
        else {
            setOccupied(slot, true);
            occupiedCount_++;
            clockHand_ = (slot + 1) % nSlots_;
        }
    } else {
        slot = reclaimSlotLocked();
    }

    setUsed(slot, true);
    const uint64_t packed = packedKey;
    slotKeyPacked_[slot].store(packed, std::memory_order_release);
    // Insert into the lock-free shard table: probe linearly, replace first
    // empty-or-tombstone slot with an occupied entry. Readers are lock-
    // free; the release-store publishes our entry to concurrent probes.
    {
        size_t idx = mh & kShardMapMask;
        const uint64_t newEntry = makeOccupiedEntry(uint32_t(slot), mh);
        for (size_t probe = 0; probe < kShardMapSize; ++probe) {
            const size_t pos = (idx + probe) & kShardMapMask;
            const uint64_t e = shard.table[pos].load(std::memory_order_relaxed);
            if (e == kEntryEmpty || (e & kEntryStateMask) == kEntryTombstone) {
                shard.table[pos].store(newEntry, std::memory_order_release);
                break;
            }
        }
    }
    if (key.level >= 0 && key.level < kMaxLevels)
        levelOccupied_[key.level]++;
    // Populate L2 for this key so the very first get() after insert goes
    // lock-free. Same empty-preferred / round-robin-fallback placement as
    // the slow-path populate in get().
    const uint32_t tag = keyTag32(packed);
    {
        const size_t setIdx = l2Index(packed, kL2Bits);
        std::atomic<uint64_t>* set = &l2_[setIdx * kL2Ways];
        size_t wayToUse = kL2Ways;
        for (size_t way = 0; way < kL2Ways; ++way) {
            if (set[way].load(std::memory_order_relaxed) == kL2Empty) {
                wayToUse = way; break;
            }
        }
        if (wayToUse == kL2Ways) {
            const uint8_t c = l2RrCounters_[setIdx];
            l2RrCounters_[setIdx] = uint8_t((c + 1u) & (kL2Ways - 1u));
            wayToUse = c & (kL2Ways - 1u);
        }
        set[wayToUse].store((uint64_t(tag) << 32) | uint32_t(slot),
                            std::memory_order_release);
    }
    return slot;
}

void BlockCache::putLocked(const BlockKey& key, const uint8_t* src, uint64_t gen) noexcept
{
    const size_t slot = acquireSlotLocked(key, gen);
    if (slot == SIZE_MAX) return;
    std::memcpy(arena_[slot].data, src, kBlockBytes);
}

size_t BlockCache::reclaimSlotLocked()
{
    // Clock sweep with per-level floor protection. A slot is "protected"
    // while its level's occupancy is at or below that level's floor — the
    // sweep walks past such slots without clearing their used bit, so they
    // can't win eviction. Config guarantees sum(floors) <= nSlots_/2, so a
    // non-protected victim always exists when the arena is full.
    for (;;) {
        size_t i = clockHand_;
        clockHand_ = (clockHand_ + 1) % nSlots_;
        if (!isOccupied(i)) continue;
        const uint64_t pk = slotKeyPacked_[i].load(std::memory_order_relaxed);
        const BlockKey k = unpackBlockKey(pk);
        const bool protectedSlot =
            (k.level >= 0 && k.level < kMaxLevels)
            && (levelOccupied_[k.level] <= levelFloor_[k.level]);
        if (protectedSlot) continue;
        if (!isUsed(i)) {
            if (k.level >= 0 && k.level < kMaxLevels
                && levelOccupied_[k.level] > 0)
                levelOccupied_[k.level]--;
            // Erase from the shard's lock-free table: probe for the
            // matching (hash, slot) entry and mark it tombstone. Linear
            // probing requires tombstones to preserve later entries'
            // reachability.
            {
                const size_t sh = shardIndex(k);
                auto& shard = shards_[sh];
                const uint32_t mh = shardMapHash(k);
                size_t idx = mh & kShardMapMask;
                for (size_t probe = 0; probe < kShardMapSize; ++probe) {
                    const size_t pos = (idx + probe) & kShardMapMask;
                    const uint64_t e = shard.table[pos].load(std::memory_order_relaxed);
                    if (e == kEntryEmpty) break;
                    if ((e & kEntryStateMask) == kEntryOccupied
                        && uint32_t(e & kEntryHashMask) == mh
                        && uint32_t((e & kEntrySlotMask) >> kEntrySlotShift) == uint32_t(i)) {
                        shard.table[pos].store(kEntryTombstone, std::memory_order_release);
                        break;
                    }
                }
            }
            // Invalidate slot FIRST so any in-flight L2 reader verifies
            // against an empty packed-key. If a stale L2 entry still points
            // here, the verify load will mismatch → reader falls through.
            slotKeyPacked_[i].store(UINT64_MAX, std::memory_order_release);
            // Clear the L2 entry in whichever way holds this (tag, slot)
            // pair — best effort, only touch entries that still match
            // both the tag and the slot index so we don't invalidate an
            // unrelated block.
            const uint32_t oldTag = keyTag32(pk);
            {
                const size_t setIdx = l2Index(pk, kL2Bits);
                std::atomic<uint64_t>* set = &l2_[setIdx * kL2Ways];
                for (size_t way = 0; way < kL2Ways; ++way) {
                    const uint64_t cur = set[way].load(std::memory_order_relaxed);
                    if (uint32_t(cur >> 32) == oldTag &&
                        uint32_t(cur) == uint32_t(i)) {
                        set[way].store(kL2Empty, std::memory_order_release);
                        break;
                    }
                }
            }
            evictionVersion_.fetch_add(1, std::memory_order_relaxed);
            return i;
        }
        setUsed(i, false);
    }
}

size_t BlockCache::size() const noexcept
{
    std::shared_lock lock(arenaMutex_);
    return occupiedCount_;
}

uint64_t BlockCache::generation() const noexcept
{
    return generation_.load(std::memory_order_relaxed);
}

void BlockCache::clear()
{
    std::unique_lock lock(arenaMutex_);
    generation_.fetch_add(1, std::memory_order_relaxed);
    for (auto& s : shards_) {
        for (size_t i = 0; i < kShardMapSize; ++i)
            s.table[i].store(kEntryEmpty, std::memory_order_relaxed);
        s.hits.store(0, std::memory_order_relaxed);
    }
    for (size_t i = 0; i < nSlots_; ++i)
        slotKeyPacked_[i].store(UINT64_MAX, std::memory_order_relaxed);
    for (size_t i = 0; i < kL2Size; ++i)
        l2_[i].store(kL2Empty, std::memory_order_relaxed);
    std::fill(occupiedBits_.begin(), occupiedBits_.end(), 0);
    for (size_t i = 0; i < usedBitsWords_; ++i)
        usedBits_[i].store(0, std::memory_order_relaxed);
    occupiedCount_ = 0;
    clockHand_ = 0;
    levelOccupied_.fill(0);
    evictionVersion_.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace vc::cache
