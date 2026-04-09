#include "vc/core/cache/IOPool.hpp"

#include <algorithm>
#include <unordered_set>

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
                ShardKey shard;
                try {
                    shard = popNext();
                } catch (const std::runtime_error&) {
                    return;
                }
                if (stop.stop_requested()) return;

                FetchResult result;
                if (fetchFunc_) {
                    try {
                        result = fetchFunc_(shard);
                    } catch (const std::exception& e) {
                        std::lock_guard lock(mutex_);
                        shards_[shard].state = ShardState::Done;
                        continue;
                    }
                }

                {
                    std::lock_guard lock(mutex_);
                    shards_[shard].state = ShardState::Done;
                }

                if (onComplete_ && !result.empty()) {
                    try {
                        onComplete_(std::move(result));
                    } catch (const std::exception&) {}
                }
            }
        });
    }
}

IOPool::~IOPool() { stop(); }

void IOPool::setShardMapper(ShardMapper fn) { shardMapper_ = std::move(fn); }
void IOPool::setFetchFunc(FetchFunc fn) { fetchFunc_ = std::move(fn); }
void IOPool::setCompletionCallback(CompletionCallback cb) { onComplete_ = std::move(cb); }

void IOPool::submit(const std::vector<ChunkKey>& keys, Priority pri)
{
    if (keys.empty()) return;

    bool added = false;
    {
        std::lock_guard lock(mutex_);
        if (shutdown_) return;

        for (const auto& key : keys) {
            ShardKey sk = shardMapper_(key);
            auto it = shards_.find(sk);

            if (it != shards_.end()) {
                if (it->second.state == ShardState::InFlight)
                    continue;  // currently downloading, skip

                if (it->second.state == ShardState::Done) {
                    // Re-open: caller filtered hot cache, so data is needed again
                    it->second = {pri, ShardState::Queued};
                    auto& q = (pri == Priority::Interactive) ? interactiveQ_ : prefetchQ_;
                    q.push_back(sk);
                    added = true;
                    continue;
                }

                // Queued: promote priority if needed
                if (pri < it->second.priority) {
                    std::erase(prefetchQ_, sk);
                    it->second.priority = pri;
                    interactiveQ_.push_back(sk);
                    added = true;
                }
                continue;
            }

            // New shard
            shards_[sk] = {pri, ShardState::Queued};
            auto& q = (pri == Priority::Interactive) ? interactiveQ_ : prefetchQ_;
            q.push_back(sk);
            added = true;
        }
    }
    if (added) cv_.notify_all();
}

void IOPool::updateInteractive(const std::vector<ChunkKey>& keys)
{
    if (keys.empty()) return;

    {
        std::lock_guard lock(mutex_);
        if (shutdown_) return;

        // Build set of shard IDs for the new interactive viewport
        std::unordered_set<ShardKey, ShardKeyHash> wanted;
        wanted.reserve(keys.size());
        for (const auto& key : keys)
            wanted.insert(shardMapper_(key));

        // Walk current interactive queue: keep what's still wanted, demote the rest
        std::deque<ShardKey> newInteractiveQ;
        for (const auto& sk : interactiveQ_) {
            auto it = shards_.find(sk);
            if (it == shards_.end() || it->second.state != ShardState::Queued)
                continue;
            if (wanted.contains(sk)) {
                newInteractiveQ.push_back(sk);
                wanted.erase(sk);
            } else {
                // Demote to prefetch
                it->second.priority = Priority::Prefetch;
                prefetchQ_.push_back(sk);
            }
        }

        // Remaining wanted shards: promote from prefetch or add fresh
        for (const auto& sk : wanted) {
            auto it = shards_.find(sk);
            if (it != shards_.end()) {
                if (it->second.state == ShardState::InFlight)
                    continue;
                if (it->second.state == ShardState::Done) {
                    // Re-open: data needed again (evicted from hot cache)
                    it->second = {Priority::Interactive, ShardState::Queued};
                } else {
                    // Promote from prefetch
                    std::erase(prefetchQ_, sk);
                    it->second.priority = Priority::Interactive;
                }
            } else {
                // Brand new
                shards_[sk] = {Priority::Interactive, ShardState::Queued};
            }
            newInteractiveQ.push_back(sk);
        }

        interactiveQ_ = std::move(newInteractiveQ);
    }
    cv_.notify_all();
}

ShardKey IOPool::popNext()
{
    std::unique_lock lock(mutex_);
    cv_.wait(lock, [this] {
        return !interactiveQ_.empty() || !prefetchQ_.empty() || shutdown_;
    });
    if (shutdown_ && interactiveQ_.empty() && prefetchQ_.empty())
        throw std::runtime_error("IOPool shutdown");

    ShardKey sk;
    if (!interactiveQ_.empty()) {
        sk = interactiveQ_.front();
        interactiveQ_.pop_front();
    } else {
        sk = prefetchQ_.front();
        prefetchQ_.pop_front();
    }

    shards_[sk].state = ShardState::InFlight;
    return sk;
}

void IOPool::cancelPending()
{
    std::lock_guard lock(mutex_);
    // Remove queued entries but preserve Done/InFlight so we don't re-download
    for (const auto& sk : interactiveQ_) {
        auto it = shards_.find(sk);
        if (it != shards_.end() && it->second.state == ShardState::Queued)
            shards_.erase(it);
    }
    for (const auto& sk : prefetchQ_) {
        auto it = shards_.find(sk);
        if (it != shards_.end() && it->second.state == ShardState::Queued)
            shards_.erase(it);
    }
    interactiveQ_.clear();
    prefetchQ_.clear();
}

size_t IOPool::interactiveCount() const noexcept
{
    std::lock_guard lock(mutex_);
    return interactiveQ_.size();
}

size_t IOPool::prefetchCount() const noexcept
{
    std::lock_guard lock(mutex_);
    return prefetchQ_.size();
}

size_t IOPool::pendingCount() const noexcept
{
    std::lock_guard lock(mutex_);
    return interactiveQ_.size() + prefetchQ_.size();
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
