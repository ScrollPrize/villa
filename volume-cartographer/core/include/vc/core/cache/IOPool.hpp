#pragma once

#include <array>
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

// Shard-level IO pool for interactive chunk fetches.
//
// Work items are shards (or individual chunks for non-sharded datasets).
// Callers submit ChunkKeys; the pool maps them to ShardKeys via a
// caller-provided ShardMapper and deduplicates at the shard level.
class IOPool {
public:
    using FetchResult = std::vector<std::pair<ChunkKey, std::vector<uint8_t>>>;
    using FetchFunc = std::function<FetchResult(const ShardKey&)>;

    using CompletionCallback = std::function<void(FetchResult&&)>;

    using ShardMapper = std::function<ShardKey(const ChunkKey&)>;

    explicit IOPool(int numThreads = 4);
    ~IOPool();

    void start();

    IOPool(const IOPool&) = delete;
    IOPool& operator=(const IOPool&) = delete;

    void setShardMapper(ShardMapper fn);
    void setFetchFunc(FetchFunc fn);
    void setCompletionCallback(CompletionCallback cb);

    void submit(const std::vector<ChunkKey>& keys);

    // Update interactive viewport. New shards get queued; old queued shards
    // not in the new set get dropped.
    void updateInteractive(const std::vector<ChunkKey>& keys);

    void cancelPending();

    [[nodiscard]] size_t pendingCount() const noexcept;

    void stop();

private:
    ShardKey popNext();

    ShardMapper shardMapper_;
    FetchFunc fetchFunc_;
    CompletionCallback onComplete_;

    mutable std::mutex mutex_;
    std::condition_variable cv_;

    enum class ShardState : uint8_t { Queued, InFlight, Done };

    std::unordered_map<ShardKey, ShardState, ShardKeyHash> shards_;
    // One queue per pyramid level. popNext() drains the coarsest (highest
    // level index) non-empty queue first, so low-res chunks stream in ahead
    // of high-res for the same viewport.
    std::array<std::deque<ShardKey>, kMaxLevels> queues_;
    size_t queueTotal_ = 0;

    bool shutdown_ = false;

    int numThreads_;
    std::vector<std::jthread> workers_;
};

}  // namespace vc::cache
