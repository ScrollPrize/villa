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
    slots_ = std::make_unique<Block[]>(nSlots_);
    slotKeys_.assign(nSlots_, kEmptyKey);
    slotOccupied_.assign(nSlots_, 0);
    evictableMap_.reserve(nSlots_);
}

BlockCache::~BlockCache() = default;

const Block* BlockCache::get(const BlockKey& key) noexcept
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

    Block* slot = nullptr;
    if (occupiedCount_ < nSlots_) {
        for (size_t i = clockHand_; ; i = (i + 1) % nSlots_) {
            if (!slotOccupied_[i]) {
                slot = &slots_[i];
                slotOccupied_[i] = 1;
                slotKeys_[i] = key;
                occupiedCount_++;
                clockHand_ = (i + 1) % nSlots_;
                break;
            }
        }
    } else {
        slot = reclaimOneLocked();
    }

    evictableMap_[key] = slot;
    std::memcpy(slot->data, src, kBlockBytes);
    slot->used.store(1, std::memory_order_relaxed);
}

void BlockCache::putResident(const BlockKey& key, const uint8_t* src)
{
    std::lock_guard lock(mutex_);
    if (auto it = residentMap_.find(key); it != residentMap_.end()) {
        std::memcpy(it->second->data, src, kBlockBytes);
        return;
    }
    auto block = std::make_unique<Block>();
    Block* ptr = block.get();
    std::memcpy(ptr->data, src, kBlockBytes);
    ptr->used.store(1, std::memory_order_relaxed);
    residentArena_.push_back(std::move(block));
    residentMap_[key] = ptr;
}

Block* BlockCache::reclaimOneLocked()
{
    // Clock sweep: advance until we find a slot with used==0; clear used
    // bits along the way. Occupied slots always exist here (caller ensures).
    for (;;) {
        size_t i = clockHand_;
        clockHand_ = (clockHand_ + 1) % nSlots_;
        if (!slotOccupied_[i]) continue;
        Block& b = slots_[i];
        if (b.used.load(std::memory_order_relaxed) == 0) {
            evictableMap_.erase(slotKeys_[i]);
            slotKeys_[i] = kEmptyKey;
            return &b;  // slot stays occupied; caller overwrites
        }
        b.used.store(0, std::memory_order_relaxed);
    }
}

size_t BlockCache::residentSlots() const noexcept
{
    std::lock_guard lock(mutex_);
    return residentArena_.size();
}

size_t BlockCache::evictableUsed() const noexcept
{
    std::lock_guard lock(mutex_);
    return occupiedCount_;
}

void BlockCache::clearEvictable()
{
    std::lock_guard lock(mutex_);
    evictableMap_.clear();
    std::fill(slotOccupied_.begin(), slotOccupied_.end(), 0);
    std::fill(slotKeys_.begin(), slotKeys_.end(), kEmptyKey);
    occupiedCount_ = 0;
    clockHand_ = 0;
}

}  // namespace vc::cache
