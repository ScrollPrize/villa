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
}

size_t BlockCache::reclaimSlotLocked()
{
    for (;;) {
        size_t i = clockHand_;
        clockHand_ = (clockHand_ + 1) % nSlots_;
        if (!isOccupied(i)) continue;
        if (!isUsed(i)) {
            map_.erase(slotKey_[i]);
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
    // Tell kernel we don't need any of these pages for now.
    if (arena_ && arenaBytes_) {
        ::madvise(arena_, arenaBytes_, MADV_DONTNEED);
    }
}

}  // namespace vc::cache
