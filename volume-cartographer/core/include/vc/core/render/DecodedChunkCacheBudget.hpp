#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>

namespace vc::render {

class ChunkCache;

// Coordinates decoded chunk storage across multiple ChunkCache instances.
// The byte ceiling belongs to the budget, not to any individual cache.
class DecodedChunkCacheBudget final {
public:
    struct Stats {
        std::size_t decodedBytes = 0;
        std::size_t maximumBytes = 0;
        std::size_t cacheCount = 0;
    };

    explicit DecodedChunkCacheBudget(std::size_t maximumBytes);
    ~DecodedChunkCacheBudget();

    DecodedChunkCacheBudget(const DecodedChunkCacheBudget&) = delete;
    DecodedChunkCacheBudget& operator=(const DecodedChunkCacheBudget&) = delete;

    [[nodiscard]] Stats stats() const;
    [[nodiscard]] std::size_t maximumBytes() const noexcept;

private:
    friend class ChunkCache;

    struct Participant {
        std::function<std::optional<std::uint64_t>()> oldestDecodedTouch;
        std::function<std::size_t()> evictOldestDecoded;
    };

    std::uint64_t registerCache(Participant participant);
    void unregisterCache(std::uint64_t id);
    std::uint64_t nextTouch() noexcept;
    void addBytes(std::size_t bytes) noexcept;
    void removeBytes(std::size_t bytes) noexcept;
    void enforce();

    const std::size_t maximumBytes_;
    std::atomic_size_t decodedBytes_{0};
    std::atomic_uint64_t nextTouch_{1};
    mutable std::mutex participantsMutex_;
    std::unordered_map<std::uint64_t, Participant> participants_;
    std::uint64_t nextParticipantId_ = 1;
    std::mutex enforcementMutex_;
};

} // namespace vc::render
