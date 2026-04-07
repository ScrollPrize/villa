#include "vc/core/cache/IOPool.hpp"
#include "vc/core/cache/CacheDebugLog.hpp"

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
            for (;;) {
                ChunkKey key;
                try {
                    key = popNext();
                } catch (const std::runtime_error&) {
                    return;
                }
                if (stop.stop_requested()) return;

                std::vector<uint8_t> data;
                if (fetchFunc_) {
                    try {
                        data = fetchFunc_(key);
                    } catch (const std::exception& e) {
                        if (auto* log = cacheDebugLog())
                            std::fprintf(log, "[IOPOOL] fetch exception lvl=%d (%d,%d,%d): %s\n",
                                         key.level, key.iz, key.iy, key.ix, e.what());
                        continue;
                    }
                }

                if (onComplete_) {
                    try {
                        onComplete_(key, std::move(data));
                    } catch (const std::exception& e) {
                        std::fprintf(stderr, "[IOPOOL] completion exception lvl=%d (%d,%d,%d): %s\n",
                                     key.level, key.iz, key.iy, key.ix, e.what());
                    }
                }
            }
        });
    }
}

IOPool::~IOPool() { stop(); }

void IOPool::setFetchFunc(FetchFunc fn) { fetchFunc_ = std::move(fn); }
void IOPool::setCompletionCallback(CompletionCallback cb) { onComplete_ = std::move(cb); }

void IOPool::submitInteractive(const ChunkKey& key)
{
    {
        std::lock_guard lock(mutex_);
        if (shutdown_ || seen_.contains(key)) return;
        seen_.insert(key);
        interactive_.push_back(key);
    }
    cv_.notify_one();
}

void IOPool::submitInteractive(const std::vector<ChunkKey>& keys)
{
    size_t added = 0;
    {
        std::lock_guard lock(mutex_);
        if (shutdown_) return;
        for (const auto& k : keys) {
            if (!seen_.contains(k)) {
                seen_.insert(k);
                interactive_.push_back(k);
                added++;
            }
        }
    }
    if (added > 0) cv_.notify_all();
}

void IOPool::submitPrefetch(const ChunkKey& key)
{
    {
        std::lock_guard lock(mutex_);
        if (shutdown_ || seen_.contains(key)) return;
        seen_.insert(key);
        prefetch_.push_back(key);
    }
    cv_.notify_one();
}

void IOPool::submitPrefetch(const std::vector<ChunkKey>& keys)
{
    size_t added = 0;
    {
        std::lock_guard lock(mutex_);
        if (shutdown_) return;
        for (const auto& k : keys) {
            if (!seen_.contains(k)) {
                seen_.insert(k);
                prefetch_.push_back(k);
                added++;
            }
        }
    }
    if (added > 0) cv_.notify_all();
}

ChunkKey IOPool::popNext()
{
    std::unique_lock lock(mutex_);
    cv_.wait(lock, [this] {
        return !interactive_.empty() || !prefetch_.empty() || shutdown_;
    });
    if (shutdown_ && interactive_.empty() && prefetch_.empty())
        throw std::runtime_error("IOPool shutdown");

    // Interactive queue first
    if (!interactive_.empty()) {
        ChunkKey k = interactive_.front();
        interactive_.pop_front();
        return k;
    }
    ChunkKey k = prefetch_.front();
    prefetch_.pop_front();
    return k;
}

void IOPool::cancelPending()
{
    std::lock_guard lock(mutex_);
    interactive_.clear();
    prefetch_.clear();
    // Don't clear seen_ — we still don't want to re-download chunks
}

size_t IOPool::interactiveCount() const noexcept
{
    std::lock_guard lock(mutex_);
    return interactive_.size();
}

size_t IOPool::prefetchCount() const noexcept
{
    std::lock_guard lock(mutex_);
    return prefetch_.size();
}

size_t IOPool::pendingCount() const noexcept
{
    std::lock_guard lock(mutex_);
    return interactive_.size() + prefetch_.size();
}

void IOPool::stop()
{
    {
        std::lock_guard lock(mutex_);
        if (shutdown_) return;
        shutdown_ = true;
    }
    cv_.notify_all();
    for (auto& w : workers_)
        w.request_stop();
    workers_.clear();
}

}  // namespace vc::cache
