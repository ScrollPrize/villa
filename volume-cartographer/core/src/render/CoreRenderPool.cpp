#include "vc/core/render/CoreRenderPool.hpp"

#include <algorithm>

#include "vc/core/render/TileRenderer.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/Surface.hpp"

#include <utils/thread_pool.hpp>

static constexpr uint64_t kEpochSlack = 5;

namespace vc::render {

CoreRenderPool::CoreRenderPool(int numThreads)
    : pool_(std::make_unique<utils::PriorityThreadPool>(numThreads))
{
}

CoreRenderPool::~CoreRenderPool()
{
    cancelAll();
}

void CoreRenderPool::submit(const TileRenderParams& params,
                            const std::shared_ptr<Surface>& surface,
                            const std::shared_ptr<Volume>& volume,
                            const std::shared_ptr<std::atomic<uint64_t>>& epochRef,
                            int controllerId)
{
    constexpr int kMaxQueuedTasks = 128;
    if (pendingCount_.load(std::memory_order_relaxed) > kMaxQueuedTasks)
        return;

    pendingCount_.fetch_add(1, std::memory_order_relaxed);

    int priority = params.submitPriority;

    pool_->submit(priority,
        [this, params, surface, volume, epochRef, controllerId]() {
            uint64_t currentEpoch = epochRef->load(std::memory_order_relaxed);
            if (currentEpoch > kEpochSlack && params.epoch < currentEpoch - kEpochSlack) {
                pendingCount_.fetch_sub(1, std::memory_order_relaxed);
                return;
            }

            TileRenderResult result = TileRenderer::renderTile(params, surface, volume.get());
            result.controllerId = controllerId;

            pushResult(std::move(result));
        });
}

std::vector<TileRenderResult> CoreRenderPool::drainCompleted(int maxResults, uint64_t minEpoch, int controllerId)
{
    resultSignalPending_.store(false, std::memory_order_relaxed);

    std::vector<TileRenderResult> all;
    {
        std::lock_guard<std::mutex> lock(resultsMutex_);
        std::swap(all, completedResults_);
    }

    std::vector<TileRenderResult> results;
    results.reserve(std::min(static_cast<int>(all.size()), maxResults));

    uint64_t effectiveMin = (minEpoch > kEpochSlack) ? minEpoch - kEpochSlack : 0;

    std::vector<TileRenderResult> remaining;
    remaining.reserve(all.size());

    for (auto& item : all) {
        if (item.controllerId != controllerId) {
            remaining.push_back(std::move(item));
            continue;
        }
        if (static_cast<int>(results.size()) < maxResults && item.epoch >= effectiveMin) {
            results.push_back(std::move(item));
        }
    }

    if (!remaining.empty()) {
        std::lock_guard<std::mutex> lock(resultsMutex_);
        completedResults_.insert(completedResults_.end(),
            std::make_move_iterator(remaining.begin()),
            std::make_move_iterator(remaining.end()));
    }

    return results;
}

void CoreRenderPool::cancelAll()
{
    pool_->cancel_pending();

    {
        std::lock_guard<std::mutex> lock(resultsMutex_);
        completedResults_.clear();
    }
}

bool CoreRenderPool::expireTimedOut()
{
    int pending = pendingCount_.load(std::memory_order_relaxed);
    if (pending <= 0)
        return false;

    if (pool_->active() > 0 || pool_->pending() > 0)
        return false;

    pendingCount_.store(0, std::memory_order_relaxed);
    return true;
}

int CoreRenderPool::pendingCount() const
{
    return pendingCount_.load(std::memory_order_relaxed);
}

void CoreRenderPool::setReadyCallback(std::function<void()> cb)
{
    readyCallback_ = std::move(cb);
}

void CoreRenderPool::pushResult(TileRenderResult result)
{
    {
        std::lock_guard<std::mutex> lock(resultsMutex_);
        completedResults_.push_back(std::move(result));
    }
    pendingCount_.fetch_sub(1, std::memory_order_acq_rel);

    if (!resultSignalPending_.exchange(true, std::memory_order_relaxed)) {
        if (readyCallback_)
            readyCallback_();
    }
}

} // namespace vc::render
