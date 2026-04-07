#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <unordered_set>
#include <vector>

#include "ChunkKey.hpp"

namespace vc::cache {

// Two-queue IO pool: interactive + prefetch.
// Workers always drain the interactive queue first.
// Both queues dedup against a shared set — a chunk is never downloaded twice.
// Before fetching from S3, the fetch function checks local disk cache.
class IOPool {
public:
    using CompletionCallback =
        std::function<void(const ChunkKey&, std::vector<uint8_t>&&)>;
    using FetchFunc = std::function<std::vector<uint8_t>(const ChunkKey&)>;

    explicit IOPool(int numThreads = 4);
    ~IOPool();

    void start();

    IOPool(const IOPool&) = delete;
    IOPool& operator=(const IOPool&) = delete;

    void setFetchFunc(FetchFunc fn);
    void setCompletionCallback(CompletionCallback cb);

    // Interactive queue: for chunks the user is looking at RIGHT NOW.
    // These download before anything in the prefetch queue.
    void submitInteractive(const ChunkKey& key);
    void submitInteractive(const std::vector<ChunkKey>& keys);

    // Prefetch queue: background bulk download of entire levels.
    // Only processed when the interactive queue is empty.
    void submitPrefetch(const ChunkKey& key);
    void submitPrefetch(const std::vector<ChunkKey>& keys);

    void cancelPending();

    [[nodiscard]] size_t interactiveCount() const noexcept;
    [[nodiscard]] size_t prefetchCount() const noexcept;
    [[nodiscard]] size_t pendingCount() const noexcept;

    void stop();

private:
    ChunkKey popNext();  // blocks until a key is available

    FetchFunc fetchFunc_;
    CompletionCallback onComplete_;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<ChunkKey> interactive_;   // high priority
    std::deque<ChunkKey> prefetch_;      // low priority
    std::unordered_set<ChunkKey, ChunkKeyHash> seen_;  // dedup: never download twice
    bool shutdown_ = false;

    int numThreads_;
    std::vector<std::jthread> workers_;
};

}  // namespace vc::cache
