#include "RenderPool.hpp"

#include <chrono>

#include "vc/core/util/Surface.hpp"
#include "vc/core/types/Volume.hpp"

// ============================================================================
// RenderPool
// ============================================================================

RenderPool::RenderPool(int numThreads, QObject* parent)
    : QObject(parent)
    , pool_(std::make_unique<utils::PriorityThreadPool>(numThreads))
{
}

RenderPool::~RenderPool()
{
    // Do NOT call cancelAll() here — that calls pool_->cancel_pending()
    // which would kill other controllers' tasks if this pool is shared.
    // The unique_ptr<PriorityThreadPool> destructor handles shutdown.
}

void RenderPool::submit(const TileRenderParams& params,
                        const std::shared_ptr<Surface>& surface,
                        const std::shared_ptr<Volume>& volume,
                        const std::shared_ptr<std::atomic<uint64_t>>& epochRef,
                        int controllerId)
{
    pendingCount_.fetch_add(1, std::memory_order_relaxed);

    // Track submission time for timeout detection
    {
        std::lock_guard<std::mutex> lock(timeMutex_);
        auto now = std::chrono::steady_clock::now();
        if (!hasSubmissions_) {
            oldestSubmitTime_ = now;
            hasSubmissions_ = true;
        }
    }

    // Coarser pyramid levels (higher dsScaleIdx) get higher priority (lower value)
    // so fallback previews appear before fine tiles.
    int priority = -params.dsScaleIdx;

    // Submit without pool-level epoch filtering (the pool is shared across
    // multiple controllers with independent epoch counters).  Instead, check
    // the controller's epoch before and after rendering.
    // epochRef is captured by shared_ptr so the atomic outlives the controller.
    pool_->submit(priority,
        [this, params, surface, volume, epochRef, controllerId]() {
            // Pre-render staleness check
            if (params.epoch < epochRef->load(std::memory_order_relaxed)) {
                pendingCount_.fetch_sub(1, std::memory_order_relaxed);
                return;
            }

            TileRenderResult result = TileRenderer::renderTile(params, surface, volume.get());
            result.controllerId = controllerId;

            // Post-render staleness check
            if (params.epoch < epochRef->load(std::memory_order_relaxed)) {
                pendingCount_.fetch_sub(1, std::memory_order_relaxed);
                return;
            }

            pushResult(std::move(result));
        });
}

std::vector<TileRenderResult> RenderPool::drainCompleted(int maxResults, uint64_t minEpoch, int controllerId)
{
    std::vector<TileRenderResult> results;
    std::lock_guard<std::mutex> lock(resultsMutex_);

    results.reserve(std::min(static_cast<int>(completedResults_.size()), maxResults));

    auto it = completedResults_.begin();
    while (it != completedResults_.end() && static_cast<int>(results.size()) < maxResults) {
        if (it->controllerId != controllerId) {
            ++it;  // belongs to another controller, skip
            continue;
        }
        if (it->epoch >= minEpoch) {
            results.push_back(std::move(*it));
        }
        // Stale results (epoch < minEpoch) are silently discarded
        it = completedResults_.erase(it);
    }

    return results;
}

void RenderPool::cancelAll()
{
    pool_->cancel_pending();
    pool_->wait_idle();

    {
        std::lock_guard<std::mutex> lock(resultsMutex_);
        completedResults_.clear();
        pendingCount_.store(0, std::memory_order_relaxed);
    }
    {
        std::lock_guard<std::mutex> lock(timeMutex_);
        hasSubmissions_ = false;
    }
}

bool RenderPool::expireTimedOut(std::chrono::steady_clock::duration timeout)
{
    int pending = pendingCount_.load(std::memory_order_relaxed);
    if (pending <= 0)
        return false;

    // Only expire if the pool is idle (no workers actively running tasks)
    // AND no tasks are queued. This means the pending count is stuck.
    if (pool_->active() > 0 || pool_->pending() > 0)
        return false;

    std::lock_guard<std::mutex> lock(timeMutex_);
    if (!hasSubmissions_)
        return false;

    auto elapsed = std::chrono::steady_clock::now() - oldestSubmitTime_;
    if (elapsed < timeout)
        return false;

    // Pool is idle but pendingCount > 0 and oldest submission exceeded timeout.
    // This means some tasks were lost (e.g. skipped by epoch filtering in the
    // pool without going through pushResult). Reset the count.
    pendingCount_.store(0, std::memory_order_relaxed);
    hasSubmissions_ = false;
    return true;
}

int RenderPool::pendingCount() const
{
    return pendingCount_.load(std::memory_order_relaxed);
}

void RenderPool::pushResult(TileRenderResult result)
{
    {
        std::lock_guard<std::mutex> lock(resultsMutex_);
        completedResults_.push_back(std::move(result));
    }
    auto prev = pendingCount_.fetch_sub(1, std::memory_order_acq_rel);

    // When all pending tasks have completed, reset the submission tracker
    if (prev == 1) {
        std::lock_guard<std::mutex> lock(timeMutex_);
        hasSubmissions_ = false;
    }

    emit tileReady();
}
