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
                bool fetchOk = false;
                if (fetchFunc_) {
                    try {
                        result = fetchFunc_(shard);
                        fetchOk = true;
                    } catch (const std::exception& e) {
                        // Exception = transient failure (HTTP 5xx, auth,
                        // decode). Drop the shard from the state map
                        // entirely so the next interactive request re-queues
                        // it, instead of permanently marking it Done.
                        std::lock_guard lock(mutex_);
                        shards_.erase(shard);
                        continue;
                    }
                }

                // On success record Done so back-to-back requests against
                // the same shard don't re-fetch. On empty-result success
                // (e.g. shard-sentinel missing-chunk) also Done — the
                // downstream fetchFunc_ already negative-cached that case.
                {
                    std::lock_guard lock(mutex_);
                    if (fetchOk) shards_[shard] = ShardState::Done;
                    else         shards_.erase(shard);
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

    int addedCount = 0;
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
                    ++addedCount;
                }
                continue;
            }

            shards_[sk] = ShardState::Queued;
            queues_[sk.level].push_back(sk);
            queueTotal_++;
            ++addedCount;
        }
    }
    // Wake exactly as many workers as we added items, capped at pool size.
    // notify_all() created a thundering herd on every submit — most workers
    // would re-sleep immediately after finding the queue empty.
    const int toWake = std::min(addedCount, numThreads_);
    for (int i = 0; i < toWake; ++i) cv_.notify_one();
}

void IOPool::updateInteractive(const std::vector<ChunkKey>& keys, int targetLevel)
{
    if (keys.empty()) return;

    size_t totalToWake = 0;
    {
        std::lock_guard lock(mutex_);
        if (shutdown_) return;
        if (targetLevel >= 0 && targetLevel < kMaxLevels
            && targetLevel != targetLevel_) {
            // Reset the served counters whenever the viewport target
            // changes so the new priority kicks in immediately instead of
            // being masked by historical counts.
            targetLevel_ = targetLevel;
            served_.fill(0);
        }

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
                // InFlight: a worker is on it, don't double-schedule.
                // Done: re-queue. The downstream fetchFunc (BlockPipeline
                // loader/encoder) already short-circuits when the blocks
                // or output are still in memory, so re-queueing a Done
                // shard is cheap when data is resident and correct when
                // it was evicted. Previously we skipped Done, which meant
                // evicted blocks could stay missing for the rest of the
                // session.
                if (it->second == ShardState::InFlight) continue;
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
        totalToWake = std::min<size_t>(queueTotal_, size_t(numThreads_));
    }
    for (size_t i = 0; i < totalToWake; ++i) cv_.notify_one();
}

ShardKey IOPool::popNext()
{
    std::unique_lock lock(mutex_);
    cv_.wait(lock, [this] {
        return queueTotal_ > 0 || shutdown_;
    });
    if (shutdown_ && queueTotal_ == 0)
        throw std::runtime_error("IOPool shutdown");

    // Weighted round-robin keyed on the current viewport's target level.
    // The target (the resolution the viewer is actually asking to display)
    // wins the most threads; levels adjacent to it get secondary share so
    // the coarse fallback and the zoom-out neighbours stay warm.
    //
    // Weight vs. distance-from-target: 0→6, 1→3, 2→2, 3+→1. Target at
    // level 1 (scroll at ~0.37x zoom) gives weights 3,6,3,2,1,1 for
    // levels 0..5 — level 1 gets ~38% of pops, 0 and 2 each ~19%, rest
    // ~6% each.
    auto weightFor = [&](int lvl) -> int {
        int d = lvl - targetLevel_;
        if (d < 0) d = -d;
        switch (d) {
            case 0: return 6;
            case 1: return 3;
            case 2: return 2;
            default: return 1;
        }
    };

    int bestLevel = -1;
    int bestWeight = 1;
    // Compare served/weight ratios using cross-multiplication to avoid
    // floating point. bestLevel wins when
    //   served[bestLevel] * weight[lvl] > served[lvl] * weight[bestLevel]
    // (candidate has a smaller served/weight).
    for (int lvl = 0; lvl < kMaxLevels; ++lvl) {
        if (queues_[lvl].empty()) continue;
        const int w = weightFor(lvl);
        if (bestLevel < 0) { bestLevel = lvl; bestWeight = w; continue; }
        const uint64_t lhs = served_[bestLevel] * uint64_t(w);
        const uint64_t rhs = served_[lvl]       * uint64_t(bestWeight);
        if (lhs > rhs) { bestLevel = lvl; bestWeight = w; }
    }
    if (bestLevel >= 0) {
        auto& q = queues_[bestLevel];
        ShardKey sk = q.front();
        q.pop_front();
        queueTotal_--;
        served_[bestLevel]++;
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
