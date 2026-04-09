#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <unordered_map>
#include <vector>

#include "ChunkKey.hpp"

namespace vc::cache {

enum class Priority : uint8_t {
    Interactive = 0,  // viewport chunks the user is looking at NOW
    Prefetch = 1,     // background bulk download
};

// Shard-level IO pool with priority scheduling.
//
// Work items are shards (or individual chunks for non-sharded datasets).
// Callers submit ChunkKeys; the pool maps them to ShardKeys via a
// caller-provided ShardMapper and deduplicates at the shard level.
//
// Interactive items are always drained before prefetch items.
// A shard is never downloaded twice (tracked via Done state).
// In-flight shards cannot be re-queued.
// Priority promotion: a prefetch shard can be promoted to interactive.
class IOPool {
public:
    // FetchFunc receives a ShardKey and returns all chunks from that shard.
    // For sharded datasets: downloads whole shard, returns all inner chunks.
    // For non-sharded: downloads one chunk, returns single-element vector.
    using FetchResult = std::vector<std::pair<ChunkKey, std::vector<uint8_t>>>;
    using FetchFunc = std::function<FetchResult(const ShardKey&)>;

    using CompletionCallback = std::function<void(FetchResult&&)>;

    // Maps a ChunkKey to its containing ShardKey.
    // For sharded: computes shard grid indices.
    // For non-sharded: identity mapping (ShardKey = chunk coords).
    using ShardMapper = std::function<ShardKey(const ChunkKey&)>;

    explicit IOPool(int numThreads = 4);
    ~IOPool();

    void start();

    IOPool(const IOPool&) = delete;
    IOPool& operator=(const IOPool&) = delete;

    void setShardMapper(ShardMapper fn);
    void setFetchFunc(FetchFunc fn);
    void setCompletionCallback(CompletionCallback cb);

    // Submit chunk keys at a given priority.
    // Maps to shards, deduplicates, promotes priority if already queued lower.
    void submit(const std::vector<ChunkKey>& keys, Priority pri);

    // Update interactive viewport. New shards get promoted to interactive,
    // old interactive shards not in the new set get demoted to prefetch.
    void updateInteractive(const std::vector<ChunkKey>& keys);

    void cancelPending();

    [[nodiscard]] size_t interactiveCount() const noexcept;
    [[nodiscard]] size_t prefetchCount() const noexcept;
    [[nodiscard]] size_t pendingCount() const noexcept;

    void stop();

private:
    ShardKey popNext();  // blocks until a shard is available or shutdown

    ShardMapper shardMapper_;
    FetchFunc fetchFunc_;
    CompletionCallback onComplete_;

    mutable std::mutex mutex_;
    std::condition_variable cv_;

    enum class ShardState : uint8_t { Queued, InFlight, Done };
    struct ShardEntry {
        Priority priority;
        ShardState state;
    };

    std::unordered_map<ShardKey, ShardEntry, ShardKeyHash> shards_;
    std::deque<ShardKey> interactiveQ_;  // high priority
    std::deque<ShardKey> prefetchQ_;     // low priority

    bool shutdown_ = false;

    int numThreads_;
    std::vector<std::jthread> workers_;
};

}  // namespace vc::cache
