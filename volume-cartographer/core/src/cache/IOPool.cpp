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
                        shards_[shard] = ShardState::Done;
                        continue;
                    }
                }

                {
                    std::lock_guard lock(mutex_);
                    shards_[shard] = ShardState::Done;
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

void IOPool::submit(const std::vector<ChunkKey>& keys)
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
                if (it->second == ShardState::InFlight)
                    continue;

                if (it->second == ShardState::Done) {
                    it->second = ShardState::Queued;
                    queue_.push_back(sk);
                    added = true;
                }
                continue;
            }

            shards_[sk] = ShardState::Queued;
            queue_.push_back(sk);
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

        // Priority model: the most-recent call reflects what the user is
        // looking at *right now*. Its keys go to the front of the queue in
        // order. Previously-queued keys not in the new request stay at the
        // back as backlog (e.g., coarsest-level resident preload) so they
        // still complete after viewport demand is satisfied. In-flight work
        // keeps running regardless.
        std::unordered_set<ShardKey, ShardKeyHash> newWanted;
        newWanted.reserve(keys.size());
        for (const auto& key : keys)
            newWanted.insert(shardMapper_(key));

        std::deque<ShardKey> backlog;
        for (const auto& sk : queue_) {
            auto it = shards_.find(sk);
            if (it == shards_.end() || it->second != ShardState::Queued) continue;
            if (!newWanted.count(sk)) backlog.push_back(sk);
        }

        std::deque<ShardKey> front;
        std::unordered_set<ShardKey, ShardKeyHash> seen;
        seen.reserve(keys.size());
        for (const auto& key : keys) {
            ShardKey sk = shardMapper_(key);
            if (!seen.insert(sk).second) continue;

            auto it = shards_.find(sk);
            if (it != shards_.end()) {
                // Already in flight or finished — don't re-queue. Re-fetching
                // a finished chunk forces a redundant disk read + h265 decode
                // per render pass; block cache evictions are handled via the
                // blocking getBlockingBlock path on the sampler side.
                if (it->second == ShardState::InFlight
                    || it->second == ShardState::Done) continue;
                it->second = ShardState::Queued;
            } else {
                shards_[sk] = ShardState::Queued;
            }
            front.push_back(sk);
        }

        queue_.clear();
        queue_.insert(queue_.end(), front.begin(), front.end());
        queue_.insert(queue_.end(), backlog.begin(), backlog.end());
    }
    cv_.notify_all();
}

ShardKey IOPool::popNext()
{
    std::unique_lock lock(mutex_);
    cv_.wait(lock, [this] {
        return !queue_.empty() || shutdown_;
    });
    if (shutdown_ && queue_.empty())
        throw std::runtime_error("IOPool shutdown");

    ShardKey sk = queue_.front();
    queue_.pop_front();
    shards_[sk] = ShardState::InFlight;
    return sk;
}

void IOPool::cancelPending()
{
    std::lock_guard lock(mutex_);
    for (const auto& sk : queue_) {
        auto it = shards_.find(sk);
        if (it != shards_.end() && it->second == ShardState::Queued)
            shards_.erase(it);
    }
    queue_.clear();
}

size_t IOPool::pendingCount() const noexcept
{
    std::lock_guard lock(mutex_);
    return queue_.size();
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
