#include "vc/core/cache/IOPool.hpp"
#include "vc/core/cache/CacheDebugLog.hpp"

#include <algorithm>
#include <exception>

namespace vc::cache {

IOPool::IOPool(int numThreads)
    : numThreads_(numThreads)
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
                        continue;
                    } catch (...) {
                        if (auto* log = cacheDebugLog())
                            std::fprintf(log, "[IOPOOL] unknown fetch exception for lvl=%d pos=(%d,%d,%d)\n",
                                         task.key.level, task.key.iz, task.key.iy, task.key.ix);
                        continue;
                    }
                }

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

void IOPool::submit(const ChunkKey& key)
{
    queue_.submit(Task{key, nextSeq_.fetch_add(1, std::memory_order_relaxed)});
}

void IOPool::submit(const std::vector<ChunkKey>& keys)
{
    std::vector<Task> tasks;
    tasks.reserve(keys.size());
    for (const auto& k : keys) {
        tasks.push_back(Task{k, nextSeq_.fetch_add(1, std::memory_order_relaxed)});
    }
    queue_.submit_batch(tasks.begin(), tasks.end());
}

void IOPool::cancelPending()
{
    queue_.cancel_pending();
}

size_t IOPool::pendingCount() const noexcept
{
    return queue_.queued_count();
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
