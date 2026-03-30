#include "vc/core/cache/IOPool.hpp"
#include "vc/core/cache/CacheDebugLog.hpp"

#include <algorithm>
#include <exception>

namespace vc::cache {

IOPool::IOPool(int numThreads, size_t maxQueueSize)
    : maxQueueSize_(maxQueueSize)
    , numThreads_(numThreads)
{
}

void IOPool::start()
{
    workers_.reserve(numThreads_);
    for (int i = 0; i < numThreads_; i++) {
        workers_.emplace_back([this](std::stop_token stop) {
            // consume_loop pattern: pop tasks until shutdown
            for (;;) {
                Task task;
                try {
                    task = queue_.pop();
                } catch (const std::runtime_error&) {
                    return; // shutdown signalled
                }

                if (stop.stop_requested()) return;

                // Fetch the chunk data
                std::vector<uint8_t> data;
                if (fetchFunc_) {
                    try {
                        data = fetchFunc_(task.key);
                    } catch (const std::exception& e) {
                        if (auto* log = cacheDebugLog())
                            std::fprintf(log, "[IOPOOL] fetch exception for lvl=%d pos=(%d,%d,%d): %s\n",
                                         task.key.level, task.key.iz, task.key.iy, task.key.ix, e.what());
                        queue_.complete(task);
                        continue;
                    } catch (...) {
                        if (auto* log = cacheDebugLog())
                            std::fprintf(log, "[IOPOOL] unknown fetch exception for lvl=%d pos=(%d,%d,%d)\n",
                                         task.key.level, task.key.iz, task.key.iy, task.key.ix);
                        queue_.complete(task);
                        continue;
                    }
                }

                queue_.complete(task);

                // Notify completion
                if (onComplete_) {
                    onComplete_(task.key, std::move(data));
                }
            }
        });
    }
}

IOPool::~IOPool() { stop(); }

void IOPool::setFetchFunc(FetchFunc fn)
{
    fetchFunc_ = std::move(fn);
}

void IOPool::setCompletionCallback(CompletionCallback cb)
{
    onComplete_ = std::move(cb);
}

void IOPool::setCurrentEpoch(uint64_t epoch)
{
    currentEpoch_.store(epoch, std::memory_order_relaxed);
}

void IOPool::submit(const ChunkKey& key)
{
    // Backpressure: if queue is at capacity, flush all pending (queued but
    // not in-flight) tasks to make room.  New requests are more relevant
    // than old ones since the user has likely navigated elsewhere.
    if (queue_.queued_count() >= maxQueueSize_) {
        queue_.cancel_pending();
    }

    queue_.submit(Task{
        key,
        nextSeq_.fetch_add(1, std::memory_order_relaxed),
        currentEpoch_.load(std::memory_order_relaxed)
    });
}

void IOPool::submit(const std::vector<ChunkKey>& keys)
{
    // Backpressure: if adding this batch would exceed capacity, flush all
    // pending tasks first.  In-flight downloads continue uninterrupted.
    size_t currentSize = queue_.queued_count();
    if (currentSize + keys.size() > maxQueueSize_) {
        queue_.cancel_pending();
    }

    uint64_t epoch = currentEpoch_.load(std::memory_order_relaxed);
    std::vector<Task> tasks;
    tasks.reserve(keys.size());
    for (const auto& k : keys) {
        tasks.push_back(Task{
            k,
            nextSeq_.fetch_add(1, std::memory_order_relaxed),
            epoch
        });
    }
    queue_.submit_batch(tasks.begin(), tasks.end());
}

void IOPool::submitBackground(const std::vector<ChunkKey>& keys)
{
    // Background submit: no cancel_pending. If queue is full, just skip.
    if (queue_.queued_count() >= maxQueueSize_) return;

    uint64_t epoch = currentEpoch_.load(std::memory_order_relaxed);
    std::vector<Task> tasks;
    tasks.reserve(keys.size());
    for (const auto& k : keys) {
        tasks.push_back(Task{
            k,
            nextSeq_.fetch_add(1, std::memory_order_relaxed),
            epoch
        });
    }
    queue_.submit_batch(tasks.begin(), tasks.end());
}

void IOPool::cancelPending()
{
    queue_.cancel_pending();
}

size_t IOPool::pendingCount() const
{
    return queue_.queued_count() + queue_.in_flight_count();
}

void IOPool::stop()
{
    queue_.shutdown();
    for (auto& w : workers_) {
        w.request_stop();
    }
    // Workers will unblock from queue_.pop() due to shutdown
    workers_.clear(); // jthread destructors join
}

}  // namespace vc::cache
