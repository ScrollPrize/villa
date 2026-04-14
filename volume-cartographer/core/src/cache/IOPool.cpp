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
            if (sk.level < 0 || sk.level >= kMaxLevels) continue;
            auto it = shards_.find(sk);

            if (it != shards_.end()) {
                if (it->second == ShardState::InFlight)
                    continue;

                if (it->second == ShardState::Done) {
                    it->second = ShardState::Queued;
                    queues_[sk.level].push_back(sk);
                    queueTotal_++;
                    added = true;
                }
                continue;
            }

            shards_[sk] = ShardState::Queued;
            queues_[sk.level].push_back(sk);
            queueTotal_++;
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
        // looking at *right now*. Within each level's queue, new keys go to
        // the front in order; queued keys not in the new request drop to
        // backlog at the back. The worker side (popNext) drains coarsest
        // level first, so low-res chunks stream in ahead of high-res even
        // when the caller mixes levels in one submission.
        std::unordered_set<ShardKey, ShardKeyHash> newWanted;
        newWanted.reserve(keys.size());
        for (const auto& key : keys)
            newWanted.insert(shardMapper_(key));

        std::array<std::deque<ShardKey>, kMaxLevels> front;
        std::array<std::deque<ShardKey>, kMaxLevels> backlog;

        for (int lvl = 0; lvl < kMaxLevels; ++lvl) {
            for (const auto& sk : queues_[lvl]) {
                auto it = shards_.find(sk);
                if (it == shards_.end() || it->second != ShardState::Queued) continue;
                if (!newWanted.count(sk)) backlog[lvl].push_back(sk);
            }
        }

        std::unordered_set<ShardKey, ShardKeyHash> seen;
        seen.reserve(keys.size());
        for (const auto& key : keys) {
            ShardKey sk = shardMapper_(key);
            if (sk.level < 0 || sk.level >= kMaxLevels) continue;
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
            front[sk.level].push_back(sk);
        }

        queueTotal_ = 0;
        for (int lvl = 0; lvl < kMaxLevels; ++lvl) {
            auto& q = queues_[lvl];
            q.clear();
            q.insert(q.end(), front[lvl].begin(), front[lvl].end());
            q.insert(q.end(), backlog[lvl].begin(), backlog[lvl].end());
            queueTotal_ += q.size();
        }
    }
    cv_.notify_all();
}

ShardKey IOPool::popNext()
{
    std::unique_lock lock(mutex_);
    cv_.wait(lock, [this] {
        return queueTotal_ > 0 || shutdown_;
    });
    if (shutdown_ && queueTotal_ == 0)
        throw std::runtime_error("IOPool shutdown");

    // Coarsest level first: level indices grow coarser (higher index).
    for (int lvl = kMaxLevels - 1; lvl >= 0; --lvl) {
        auto& q = queues_[lvl];
        if (q.empty()) continue;
        ShardKey sk = q.front();
        q.pop_front();
        queueTotal_--;
        shards_[sk] = ShardState::InFlight;
        return sk;
    }
    // Unreachable: queueTotal_ > 0 implies some queue is non-empty.
    throw std::runtime_error("IOPool queue inconsistency");
}

void IOPool::cancelPending()
{
    std::lock_guard lock(mutex_);
    for (auto& q : queues_) {
        for (const auto& sk : q) {
            auto it = shards_.find(sk);
            if (it != shards_.end() && it->second == ShardState::Queued)
                shards_.erase(it);
        }
        q.clear();
    }
    queueTotal_ = 0;
}

size_t IOPool::pendingCount() const noexcept
{
    std::lock_guard lock(mutex_);
    return queueTotal_;
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
