#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <thread>
#include <vector>

#include "ChunkKey.hpp"
#include <utils/priority_queue.hpp>

namespace vc::cache {

// Background I/O thread pool for async chunk fetching.
// Tasks are prioritized: coarser pyramid level = higher priority.
// Duplicate submissions for the same key are deduplicated.
//
// Internally uses utils::DeduplicatingPriorityQueue for thread-safe
// dedup + priority ordering, and std::jthread workers running consume_loop.
class IOPool {
public:
    // Callback signature: called from a worker thread when a chunk is fetched.
    // (key, compressed bytes) — empty bytes means fetch failed.
    using CompletionCallback =
        std::function<void(const ChunkKey&, std::vector<uint8_t>&&)>;

    // FetchFunc: the actual data fetching function (wraps ChunkSource::fetch).
    // Called from worker threads. Must be thread-safe.
    using FetchFunc = std::function<std::vector<uint8_t>(const ChunkKey&)>;

    explicit IOPool(int numThreads = 4, size_t maxQueueSize = 1000);
    ~IOPool();

    // Start worker threads. Must be called after setFetchFunc/setCompletionCallback.
    void start();

    IOPool(const IOPool&) = delete;
    IOPool& operator=(const IOPool&) = delete;

    // Set the fetch function (typically wraps ChunkSource + DiskStore lookup).
    void setFetchFunc(FetchFunc fn);

    // Set the completion callback (called from worker thread).
    void setCompletionCallback(CompletionCallback cb);

    // Set the current epoch. Tasks from the current epoch get higher priority.
    void setCurrentEpoch(uint64_t epoch);

    // Submit a chunk for background fetching.
    // Deduplicates: if the key is already queued or in-flight, this is a no-op.
    void submit(const ChunkKey& key);

    // Submit multiple keys at once (batch, reduces lock contention).
    // Uses backpressure: cancels pending tasks if queue would overflow.
    void submit(const std::vector<ChunkKey>& keys);

    // Submit keys without cancel-on-overflow (for background prefetch).
    // Silently drops keys that don't fit in the queue.
    void submitBackground(const std::vector<ChunkKey>& keys);

    // Cancel all pending (not in-flight) tasks.
    void cancelPending();

    // Number of pending + in-flight tasks.
    [[nodiscard]] size_t pendingCount() const;

    // Gracefully stop all workers. Blocks until all threads exit.
    void stop();

private:
    // Internal task type — stored in the priority queue.
    struct Task {
        ChunkKey key;
        uint64_t seq = 0;
        uint64_t epoch = 0;

        // Priority ordering: coarser level first, then newer epoch, then FIFO.
        bool operator>(const Task& o) const noexcept
        {
            if (key.level != o.key.level) return key.level < o.key.level;
            if (epoch != o.epoch) return epoch < o.epoch;
            return seq > o.seq;
        }
    };

    // Hash/Equal that operate only on the ChunkKey for dedup.
    struct TaskHash {
        size_t operator()(const Task& t) const noexcept {
            return ChunkKeyHash()(t.key);
        }
    };
    struct TaskEqual {
        bool operator()(const Task& a, const Task& b) const noexcept {
            return a.key == b.key;
        }
    };

    using Queue = utils::DeduplicatingPriorityQueue<Task, TaskHash, TaskEqual, std::greater<Task>>;

    FetchFunc fetchFunc_;
    CompletionCallback onComplete_;

    Queue queue_;
    size_t maxQueueSize_;
    std::atomic<uint64_t> nextSeq_{0};
    std::atomic<uint64_t> currentEpoch_{0};

    int numThreads_;
    std::vector<std::jthread> workers_;
};

}  // namespace vc::cache
