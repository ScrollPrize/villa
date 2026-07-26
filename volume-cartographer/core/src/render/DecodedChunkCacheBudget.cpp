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
    result.maximumBytes = maximumBytes_;
    {
        std::lock_guard lock(participantsMutex_);
        result.cacheCount = participants_.size();
    }
    return result;
}

std::size_t DecodedChunkCacheBudget::maximumBytes() const noexcept
{
    return maximumBytes_;
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

void DecodedChunkCacheBudget::enforce()
{
    if (decodedBytes_.load(std::memory_order_acquire) <= maximumBytes_)
        return;

    std::lock_guard enforcementLock(enforcementMutex_);
    while (decodedBytes_.load(std::memory_order_acquire) > maximumBytes_) {
        std::vector<std::pair<std::uint64_t, Participant>> snapshot;
        {
            std::lock_guard lock(participantsMutex_);
            snapshot.reserve(participants_.size());
            for (const auto& [id, participant] : participants_)
                snapshot.emplace_back(id, participant);
        }

        std::uint64_t victimId = 0;
        std::uint64_t oldestTouch = std::numeric_limits<std::uint64_t>::max();
        for (const auto& [id, participant] : snapshot) {
            if (!participant.oldestDecodedTouch)
                continue;
            const auto touch = participant.oldestDecodedTouch();
            if (touch && *touch < oldestTouch) {
                oldestTouch = *touch;
                victimId = id;
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
