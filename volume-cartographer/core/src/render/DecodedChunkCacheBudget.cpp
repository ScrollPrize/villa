#include "vc/core/render/DecodedChunkCacheBudget.hpp"

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

namespace vc::render {

DecodedChunkCacheBudget::DecodedChunkCacheBudget(std::size_t maximumBytes)
    : maximumBytes_(maximumBytes)
{
}

DecodedChunkCacheBudget::~DecodedChunkCacheBudget() = default;

DecodedChunkCacheBudget::Stats DecodedChunkCacheBudget::stats() const
{
    Stats result;
    result.decodedBytes = decodedBytes_.load(std::memory_order_acquire);
    result.maximumBytes = maximumBytes_.load(std::memory_order_acquire);
    result.recycledBytes = recycledBytes_.load(std::memory_order_acquire);
    result.reusedBuffers = reusedBuffers_.load(std::memory_order_relaxed);
    result.bufferRequests = bufferRequests_.load(std::memory_order_relaxed);
    result.recycleRejected = recycleRejected_.load(std::memory_order_relaxed);
    result.trimmedBuffers = trimmedBuffers_.load(std::memory_order_relaxed);
    result.trimmedBytes = trimmedBytes_.load(std::memory_order_relaxed);
    result.reuseEnforcementBusy = reuseEnforcementBusy_.load(std::memory_order_relaxed);
    {
        std::lock_guard lock(participantsMutex_);
        result.cacheCount = participants_.size();
    }
    return result;
}

std::size_t DecodedChunkCacheBudget::maximumBytes() const noexcept
{
    return maximumBytes_.load(std::memory_order_acquire);
}

void DecodedChunkCacheBudget::setMaximumBytes(std::size_t maximumBytes)
{
    {
        std::lock_guard lock(recycledMutex_);
        maximumBytes_.store(maximumBytes, std::memory_order_release);
    }
    enforce();
}

DecodedChunkCacheBudget::Buffer DecodedChunkCacheBudget::takeBuffer(std::size_t size)
{
    bufferRequests_.fetch_add(1, std::memory_order_relaxed);
    const auto pop = [&]() -> Buffer {
        if (recycledBytes_.load(std::memory_order_relaxed) == 0) return {};
        std::lock_guard lock(recycledMutex_);
        auto it = recycled_.find(size);
        if (it == recycled_.end() || it->second.empty()) return {};
        auto bytes = std::move(it->second.back());
        it->second.pop_back();
        if (it->second.empty()) recycled_.erase(it);
        recycledBytes_.fetch_sub(bytes.capacity(), std::memory_order_relaxed);
        reusedBuffers_.fetch_add(1, std::memory_order_relaxed);
        return bytes;
    };
    auto bytes = pop();
    if (!bytes.empty()) return bytes;
    const auto maximum = maximumBytes_.load(std::memory_order_acquire);
    if (!size || size > maximum ||
        decodedBytes_.load(std::memory_order_acquire) <= maximum-size) return {};
    // Make room using the same global LRU/reader protections as publication.
    // This moves a necessary eviction before allocating its replacement, so
    // streaming through a full cache can reuse the retired allocation. No
    // source/pool locks are held here; don't queue behind another enforcer.
    enforce(size, true);
    return pop();
}

std::shared_ptr<const DecodedChunkCacheBudget::Buffer>
DecodedChunkCacheBudget::shareBuffer(Buffer bytes)
{
    return std::shared_ptr<const Buffer>(new Buffer(std::move(bytes)),
        [weak = weak_from_this()](Buffer* buffer) noexcept {
            std::unique_ptr<Buffer> owner(buffer);
            if (auto budget = weak.lock())
                budget->recycleBuffer(std::move(*buffer));
        });
}

void DecodedChunkCacheBudget::recycleBuffer(Buffer bytes) noexcept
{
    if (bytes.empty()) return;
    // No enforcement/callbacks here: this deleter can run under a source lock.
    // Allocation failure only disables reuse; it must not escape a destructor.
    try {
        std::lock_guard lock(recycledMutex_);
        const auto maximum = maximumBytes_.load(std::memory_order_acquire);
        const auto decoded = decodedBytes_.load(std::memory_order_acquire);
        const auto available = maximum - std::min(maximum, decoded);
        const auto idle = recycledBytes_.load(std::memory_order_relaxed);
        const auto capacity = bytes.capacity();
        if (bytes.empty() || capacity > available || idle > available - capacity) {
            recycleRejected_.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        auto it = recycled_.try_emplace(bytes.size()).first;
        try {
            it->second.push_back(std::move(bytes));
        } catch (...) {
            if (it->second.empty()) recycled_.erase(it);
            throw;
        }
        recycledBytes_.fetch_add(capacity, std::memory_order_relaxed);
    } catch (...) {}
    // Rejected storage is released after the pool mutex has been dropped.
}

void DecodedChunkCacheBudget::trimRecycled()
{
    while (recycledBytes_.load(std::memory_order_relaxed) != 0) {
        Buffer discarded;
        {
            std::lock_guard lock(recycledMutex_);
            const auto maximum = maximumBytes_.load(std::memory_order_acquire);
            const auto decoded = decodedBytes_.load(std::memory_order_acquire);
            if (recycledBytes_.load(std::memory_order_relaxed) <=
                maximum - std::min(maximum, decoded)) return;
            // Retain the rest of the pool: exceeding the ceiling by one
            // buffer must not discard all reusable allocations. Detach one
            // without allocating bookkeeping storage, free outside the lock.
            auto it = recycled_.begin();
            while (it != recycled_.end() && it->second.empty())
                it = recycled_.erase(it);
            if (it == recycled_.end()) return;
            discarded = std::move(it->second.back());
            it->second.pop_back();
            if (it->second.empty()) recycled_.erase(it);
            const auto capacity = discarded.capacity();
            recycledBytes_.fetch_sub(capacity, std::memory_order_relaxed);
            trimmedBuffers_.fetch_add(1, std::memory_order_relaxed);
            trimmedBytes_.fetch_add(capacity, std::memory_order_relaxed);
        }
    }
}

std::uint64_t DecodedChunkCacheBudget::registerCache(Participant participant)
{
    std::lock_guard lock(participantsMutex_);
    const std::uint64_t id = nextParticipantId_++;
    participants_.emplace(id, std::move(participant));
    return id;
}

void DecodedChunkCacheBudget::unregisterCache(std::uint64_t id)
{
    std::lock_guard lock(participantsMutex_);
    participants_.erase(id);
}

std::uint64_t DecodedChunkCacheBudget::nextTouch() noexcept
{
    return nextTouch_.fetch_add(1, std::memory_order_relaxed);
}

void DecodedChunkCacheBudget::addBytes(std::size_t bytes) noexcept
{
    decodedBytes_.fetch_add(bytes, std::memory_order_acq_rel);
}

void DecodedChunkCacheBudget::removeBytes(std::size_t bytes) noexcept
{
    auto current = decodedBytes_.load(std::memory_order_acquire);
    while (true) {
        const std::size_t next = current > bytes ? current - bytes : 0;
        if (decodedBytes_.compare_exchange_weak(
                current, next, std::memory_order_acq_rel, std::memory_order_acquire)) {
            return;
        }
    }
}

void DecodedChunkCacheBudget::enforce(std::size_t reserveBytes, bool tryOnly)
{
    trimRecycled();
    const auto target = [&] {
        const auto maximum=maximumBytes_.load(std::memory_order_acquire);
        return maximum-std::min(maximum,reserveBytes);
    };
    if (decodedBytes_.load(std::memory_order_acquire) <= target()) return;
    std::unique_lock enforcementLock(enforcementMutex_,std::defer_lock);
    if (tryOnly) { if (!enforcementLock.try_lock()) {
        reuseEnforcementBusy_.fetch_add(1, std::memory_order_relaxed);
        return;
    } }
    else enforcementLock.lock();
    while (decodedBytes_.load(std::memory_order_acquire) > target()) {
        std::vector<std::pair<std::uint64_t, Participant>> snapshot;
        {
            std::lock_guard lock(participantsMutex_);
            snapshot.reserve(participants_.size());
            for (const auto& [id, participant] : participants_)
                snapshot.emplace_back(id, participant);
        }

        // Victim selection prefers participants that volunteered to be
        // evicted first (bulk background consumers such as lasagna channel
        // sampling): while any of them still holds decoded data, the
        // interactive viewers' tiles are left alone. Within either class the
        // global LRU decides.
        std::uint64_t victimId = 0;
        std::uint64_t oldestTouch = std::numeric_limits<std::uint64_t>::max();
        bool victimEvictFirst = false;
        for (const auto& [id, participant] : snapshot) {
            if (!participant.oldestDecodedTouch)
                continue;
            const auto touch = participant.oldestDecodedTouch();
            if (!touch)
                continue;
            const bool preferOverCurrent =
                participant.evictFirst != victimEvictFirst
                    ? participant.evictFirst
                    : *touch < oldestTouch;
            if (victimId == 0 || preferOverCurrent) {
                oldestTouch = *touch;
                victimId = id;
                victimEvictFirst = participant.evictFirst;
            }
        }
        if (victimId == 0)
            break;

        auto victim = std::find_if(
            snapshot.begin(), snapshot.end(),
            [victimId](const auto& item) { return item.first == victimId; });
        if (victim == snapshot.end() || !victim->second.evictOldestDecoded ||
            victim->second.evictOldestDecoded() == 0) {
            // The candidate changed between inspection and eviction. Retry
            // against a fresh snapshot; concurrent invalidation may already
            // have brought the aggregate below its ceiling.
            continue;
        }
    }
}

} // namespace vc::render
