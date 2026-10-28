#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

namespace vc::render {

class ChunkCache;

// Coordinates decoded chunk storage across multiple ChunkCache instances.
// The byte ceiling belongs to the budget, not to any individual cache.
class DecodedChunkCacheBudget final : public std::enable_shared_from_this<DecodedChunkCacheBudget> {
public:
    struct Stats {
        std::size_t decodedBytes = 0;
        std::size_t maximumBytes = 0;
        std::size_t cacheCount = 0;
        // Released storage retained for reuse, charged against maximumBytes.
        std::size_t recycledBytes = 0;
        std::uint64_t reusedBuffers = 0;
        std::uint64_t bufferRequests = 0;
        std::uint64_t recycleRejected = 0;
        std::uint64_t trimmedBuffers = 0;
        std::uint64_t trimmedBytes = 0;
        std::uint64_t reuseEnforcementBusy = 0;
        std::size_t pendingRequests = 0;
        std::size_t reserveBytes = 0;
        std::uint64_t backgroundPasses = 0;
        std::uint64_t backgroundEvictedBytes = 0;
        std::uint64_t hardLimitPasses = 0;
    };

    explicit DecodedChunkCacheBudget(std::size_t maximumBytes);
    ~DecodedChunkCacheBudget();

    DecodedChunkCacheBudget(const DecodedChunkCacheBudget&) = delete;
    DecodedChunkCacheBudget& operator=(const DecodedChunkCacheBudget&) = delete;

    [[nodiscard]] Stats stats() const;
    [[nodiscard]] std::size_t maximumBytes() const noexcept;
    // Changes the global decoded-byte ceiling in place. Existing participants
    // and in-flight work remain registered; reductions evict decoded LRU data.
    void setMaximumBytes(std::size_t maximumBytes);

private:
    friend class ChunkCache;

    struct Participant {
        std::function<std::optional<std::uint64_t>(bool)> oldestDecodedTouch;
        std::function<std::size_t(bool)> evictOldestDecoded;
        // Preferred eviction victim (see
        // ChunkCacheOptions::decodedEvictionPreferSelf): while any flagged
        // participant still holds decoded data, enforcement evicts among the
        // flagged ones only.
        bool evictFirst = false;
    };

    std::uint64_t registerCache(Participant participant);
    void unregisterCache(std::uint64_t id);
    std::uint64_t nextTouch() noexcept;
    void addBytes(std::size_t bytes) noexcept;
    void removeBytes(std::size_t bytes) noexcept;
    void enforce();
    void enforceTarget(std::size_t reserveBytes, bool background);
    void requestCleanup();
    std::size_t reserveBytes() const noexcept;
    void beginRequest() noexcept;
    void endRequests(std::size_t count = 1) noexcept;
    using Buffer = std::vector<std::byte>;
    Buffer takeBuffer(std::size_t size);
    std::shared_ptr<const Buffer> shareBuffer(Buffer bytes);
    void recycleBuffer(Buffer bytes) noexcept;
    void trimRecycled();

    std::atomic_size_t maximumBytes_;
    std::atomic_size_t decodedBytes_{0};
    std::atomic_uint64_t nextTouch_{1};
    mutable std::mutex participantsMutex_;
    std::unordered_map<std::uint64_t, Participant> participants_;
    std::uint64_t nextParticipantId_ = 1;
    std::mutex enforcementMutex_;
    // Never acquire cache/enforcement locks while holding this mutex. Final
    // reader releases may recycle while a source lock is held.
    mutable std::mutex recycledMutex_;
    std::unordered_map<std::size_t, std::vector<Buffer>> recycled_;
    std::atomic_size_t recycledBytes_{0};
    std::atomic_uint64_t reusedBuffers_{0};
    std::atomic_uint64_t bufferRequests_{0};
    std::atomic_uint64_t recycleRejected_{0};
    std::atomic_uint64_t trimmedBuffers_{0};
    std::atomic_uint64_t trimmedBytes_{0};
    std::atomic_uint64_t reuseEnforcementBusy_{0};
    std::atomic_size_t pendingRequests_{0};
    std::atomic_size_t meanChunkBytes_{0};
    std::atomic_bool cleanupQueued_{false};
    std::atomic_uint64_t backgroundPasses_{0};
    std::atomic_uint64_t backgroundEvictedBytes_{0};
    std::atomic_uint64_t hardLimitPasses_{0};
};

} // namespace vc::render
