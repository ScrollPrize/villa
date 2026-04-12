#include "vc/core/cache/BlockCache.hpp"

#include <cstring>

namespace vc::cache {

namespace {
constexpr BlockKey kEmptyKey{-1, -1, -1, -1};
}

BlockCache::BlockCache(Config cfg)
    : config_(cfg)
{
    nSlots_ = config_.evictableBytes / kBlockBytes;
    slotBlock_.assign(nSlots_, nullptr);
    slotKey_.assign(nSlots_, kEmptyKey);
    occupied_.assign(nSlots_, 0);
    evictableMap_.reserve(nSlots_);
}

BlockCache::~BlockCache() = default;

BlockPtr BlockCache::get(const BlockKey& key) noexcept
{
    std::lock_guard lock(mutex_);
    if (auto it = residentMap_.find(key); it != residentMap_.end()) {
        return it->second;
    }
    if (auto it = evictableMap_.find(key); it != evictableMap_.end()) {
        it->second->used.store(1, std::memory_order_relaxed);
        return it->second;
    }
    return nullptr;
}

void BlockCache::put(const BlockKey& key, const uint8_t* src)
{
    std::lock_guard lock(mutex_);

    if (auto it = residentMap_.find(key); it != residentMap_.end()) {
        std::memcpy(it->second->data, src, kBlockBytes);
        return;
    }

    if (auto it = evictableMap_.find(key); it != evictableMap_.end()) {
        std::memcpy(it->second->data, src, kBlockBytes);
        it->second->used.store(1, std::memory_order_relaxed);
        return;
    }

    size_t slot;
    if (occupiedCount_ < nSlots_) {
        slot = clockHand_;
        while (occupied_[slot]) slot = (slot + 1) % nSlots_;
        occupied_[slot] = 1;
        occupiedCount_++;
        clockHand_ = (slot + 1) % nSlots_;
    } else {
        slot = reclaimSlotLocked();
    }

    auto block = std::make_shared<Block>();
    std::memcpy(block->data, src, kBlockBytes);
    block->used.store(1, std::memory_order_relaxed);
    slotBlock_[slot] = block;
    slotKey_[slot] = key;
    evictableMap_[key] = std::move(block);
}

void BlockCache::putResident(const BlockKey& key, const uint8_t* src)
{
    std::lock_guard lock(mutex_);
    if (auto it = residentMap_.find(key); it != residentMap_.end()) {
        std::memcpy(it->second->data, src, kBlockBytes);
        return;
    }
    auto block = std::make_shared<Block>();
    std::memcpy(block->data, src, kBlockBytes);
    block->used.store(1, std::memory_order_relaxed);
    residentMap_[key] = std::move(block);
}

size_t BlockCache::reclaimSlotLocked()
{
    for (;;) {
        size_t i = clockHand_;
        clockHand_ = (clockHand_ + 1) % nSlots_;
        if (!occupied_[i]) continue;
        auto& block = slotBlock_[i];
        if (block->used.load(std::memory_order_relaxed) == 0) {
            evictableMap_.erase(slotKey_[i]);
            slotBlock_[i].reset();     // drops our ref; external refs keep alive
            slotKey_[i] = kEmptyKey;
            return i;
        }
        block->used.store(0, std::memory_order_relaxed);
    }
}

size_t BlockCache::residentSize() const noexcept
{
    std::lock_guard lock(mutex_);
    return residentMap_.size();
}

size_t BlockCache::evictableSize() const noexcept
{
    std::lock_guard lock(mutex_);
    return occupiedCount_;
}

void BlockCache::clearEvictable()
{
    std::lock_guard lock(mutex_);
    evictableMap_.clear();
    for (auto& p : slotBlock_) p.reset();
    std::fill(slotKey_.begin(), slotKey_.end(), kEmptyKey);
    std::fill(occupied_.begin(), occupied_.end(), 0);
    occupiedCount_ = 0;
    clockHand_ = 0;
}

}  // namespace vc::cache
