#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <string>
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

    // Short label used to name worker threads via pthread_setname_np so
    // perf / top / htop can tell pools apart. Linux TASK_COMM_LEN is 16
    // (incl. NUL); actual thread name will be "<label>N" where N is the
    // worker index, so keep label ≤ 13 chars.
    void setThreadLabel(std::string label) { threadLabel_ = std::move(label); }

    void start();

    IOPool(const IOPool&) = delete;
    IOPool& operator=(const IOPool&) = delete;

    void setShardMapper(ShardMapper fn);
    void setFetchFunc(FetchFunc fn);
    void setCompletionCallback(CompletionCallback cb);

    void submit(const std::vector<ChunkKey>& keys);

    // Update interactive viewport. New shards get queued; old queued shards
    // not in the new set get dropped. targetLevel is the pyramid level the
    // viewer is currently displaying at — popNext gives it the highest
    // weight so the user reaches that resolution fastest, while levels
    // adjacent to it still make steady progress.
    void updateInteractive(const std::vector<ChunkKey>& keys, int targetLevel = 0);

    void cancelPending();

    [[nodiscard]] size_t pendingCount() const noexcept;
    [[nodiscard]] uint64_t stateVersion() const noexcept {
        return stateVersion_.load(std::memory_order_relaxed);
    }

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
    // Per-level pops served so far. popNext picks the level with the
    // smallest served/weight ratio among non-empty queues, yielding
    // weighted round-robin instead of strict priority.
    std::array<uint64_t, kMaxLevels> served_{};
    // Zoom-target level; popNext weights levels by distance from this.
    int targetLevel_ = 0;

    bool shutdown_ = false;

    // Count of workers currently blocked on cv_.wait inside popNext. Lets
    // enqueue/updateInteractive skip futex_wake syscalls when every worker
    // is already running — they'll pick up new items via their own popNext
    // loop. Without this, every panning-viewport frame wakes all workers
    // even though they're still processing the previous batch, generating
    // ~Nthreads useless context switches per frame.
    int idleCount_ = 0;

    // Monotonic state counter for callers that deduplicate submissions.
    // Bumped when workers make progress, inter-stage submits add work, or
    // pending work is dropped. Viewport queue reprioritization intentionally
    // does not bump it, so identical idle frames can keep deduplicating while
    // workers drain the existing queue.
    std::atomic<uint64_t> stateVersion_{0};

    int numThreads_;
    std::string threadLabel_;
    std::vector<std::jthread> workers_;
};

}  // namespace vc::cache
