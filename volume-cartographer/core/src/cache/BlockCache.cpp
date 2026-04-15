#include "vc/core/cache/BlockCache.hpp"

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
    usedBits_.assign(words, 0);
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
    std::lock_guard lock(mutex_);
    if (auto it = map_.find(key); it != map_.end()) {
        setUsed(it->second, true);
        return &arena_[it->second];
    }
    return nullptr;
}

void BlockCache::put(const BlockKey& key, const uint8_t* src) noexcept
{
    std::lock_guard lock(mutex_);

    if (auto it = map_.find(key); it != map_.end()) {
        Block* b = &arena_[it->second];
        std::memcpy(b->data, src, kBlockBytes);
        setUsed(it->second, true);
        return;
    }

    size_t slot;
    if (occupiedCount_ < nSlots_) {
        slot = clockHand_;
        while (isOccupied(slot)) slot = (slot + 1) % nSlots_;
        setOccupied(slot, true);
        occupiedCount_++;
        clockHand_ = (slot + 1) % nSlots_;
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
    std::lock_guard lock(mutex_);
    return occupiedCount_;
}

void BlockCache::clear()
{
    std::lock_guard lock(mutex_);
    map_.clear();
    std::fill(slotKey_.begin(), slotKey_.end(), kEmptyKey);
    std::fill(occupiedBits_.begin(), occupiedBits_.end(), 0);
    std::fill(usedBits_.begin(), usedBits_.end(), 0);
    occupiedCount_ = 0;
    clockHand_ = 0;
    levelOccupied_.fill(0);
    // Tell kernel we don't need any of these pages for now.
    if (arena_ && arenaBytes_) {
        ::madvise(arena_, arenaBytes_, MADV_DONTNEED);
    }
}

}  // namespace vc::cache
