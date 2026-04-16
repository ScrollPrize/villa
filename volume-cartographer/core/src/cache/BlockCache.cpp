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

    slotKey_.assign(nSlots_, kEmptyKey);
    const size_t words = (nSlots_ + 63u) / 64u;
    occupiedBits_.assign(words, 0);
    usedBitsWords_ = words;
    usedBits_ = std::unique_ptr<std::atomic<uint64_t>[]>(
        new std::atomic<uint64_t>[words]());
    // Default max_load_factor of 1.0 means libstdc++ sizes the bucket array
    // at 2x insertion count. At 2.5M slots that's ~40 MB of extra buckets.
    // Push the load factor to 1.0 before reserving so the reserve matches
    // actual need (buckets ≈ nSlots_ instead of ≈ 2*nSlots_).
    map_.max_load_factor(1.0f);
    map_.reserve(nSlots_);
}

BlockCache::~BlockCache()
{
    if (arena_) {
        ::munmap(arena_, arenaBytes_);
        arena_ = nullptr;
    }
}

BlockPtr BlockCache::get(const BlockKey& key) noexcept
{
    std::shared_lock lock(mutex_);
    if (auto it = map_.find(key); it != map_.end()) {
        setUsed(it->second, true);  // lock-free atomic fetch_or
        return &arena_[it->second];
    }
    return nullptr;
}

bool BlockCache::contains(const BlockKey& key) const noexcept
{
    std::shared_lock lock(mutex_);
    return map_.find(key) != map_.end();
}

void BlockCache::containsBatch(const std::vector<BlockKey>& keys,
                               std::vector<uint8_t>& out) const
{
    out.assign(keys.size(), 0);
    if (keys.empty()) return;
    std::shared_lock lock(mutex_);
    for (size_t i = 0; i < keys.size(); ++i) {
        if (map_.find(keys[i]) != map_.end()) out[i] = 1;
    }
}

void BlockCache::put(const BlockKey& key, const uint8_t* src) noexcept
{
    std::unique_lock lock(mutex_);
    putLocked(key, src);
}

void BlockCache::BatchPut::put(const BlockKey& key, const uint8_t* src) noexcept
{
    cache_.putLocked(key, src);
}

void BlockCache::putLocked(const BlockKey& key, const uint8_t* src) noexcept
{
    // Degenerate config (cache size rounded below one block) — bail
    // instead of dividing by zero in the slot-assignment arithmetic.
    if (nSlots_ == 0) return;

    if (auto it = map_.find(key); it != map_.end()) {
        Block* b = &arena_[it->second];
        std::memcpy(b->data, src, kBlockBytes);
        setUsed(it->second, true);
        return;
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

    Block* b = &arena_[slot];
    std::memcpy(b->data, src, kBlockBytes);
    setUsed(slot, true);
    slotKey_[slot] = key;
    map_[key] = slot;
    if (key.level >= 0 && key.level < kMaxLevels)
        levelOccupied_[key.level]++;
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
        const BlockKey& k = slotKey_[i];
        const bool protectedSlot =
            (k.level >= 0 && k.level < kMaxLevels)
            && (levelOccupied_[k.level] <= levelFloor_[k.level]);
        if (protectedSlot) continue;
        if (!isUsed(i)) {
            if (k.level >= 0 && k.level < kMaxLevels
                && levelOccupied_[k.level] > 0)
                levelOccupied_[k.level]--;
            map_.erase(k);
            slotKey_[i] = kEmptyKey;
            return i;
        }
        setUsed(i, false);
    }
}

size_t BlockCache::size() const noexcept
{
    std::shared_lock lock(mutex_);
    return occupiedCount_;
}

void BlockCache::clear()
{
    std::unique_lock lock(mutex_);
    map_.clear();
    std::fill(slotKey_.begin(), slotKey_.end(), kEmptyKey);
    std::fill(occupiedBits_.begin(), occupiedBits_.end(), 0);
    for (size_t i = 0; i < usedBitsWords_; ++i)
        usedBits_[i].store(0, std::memory_order_relaxed);
    occupiedCount_ = 0;
    clockHand_ = 0;
    levelOccupied_.fill(0);
    // Tell kernel we don't need any of these pages for now.
    if (arena_ && arenaBytes_) {
        ::madvise(arena_, arenaBytes_, MADV_DONTNEED);
    }
}

}  // namespace vc::cache
