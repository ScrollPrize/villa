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
    cancelAll();
}

void RenderPool::submit(const TileRenderParams& params,
                        const std::shared_ptr<Surface>& surface,
                        const std::shared_ptr<Volume>& volume,
                        const std::shared_ptr<std::atomic<uint64_t>>& epochRef,
                        int controllerId)
{
    // Cap queue depth: if too many tasks are pending, the pre-render epoch
    // check inside each lambda will naturally skip stale work.  We just need
    // to avoid submitting MORE tasks than the pool can ever process.
    constexpr int kMaxQueuedTasks = 128;
    if (pendingCount_.load(std::memory_order_relaxed) > kMaxQueuedTasks) {
        return;  // drop this submission — pool is saturated
    }

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
            // Relaxed staleness check: only skip renders that are very far
            // behind the current epoch.  During rapid interaction (zoom,
            // z-scroll) the epoch advances quickly but nearby-epoch renders
            // are still useful — showing slightly-stale data is better than
            // showing nothing.  The tile-level freshness check in setTile()
            // ensures newer results always replace older ones.
            constexpr uint64_t kEpochSlack = 5;
            uint64_t currentEpoch = epochRef->load(std::memory_order_relaxed);
            if (currentEpoch > kEpochSlack && params.epoch < currentEpoch - kEpochSlack) {
                pendingCount_.fetch_sub(1, std::memory_order_relaxed);
                return;
            }

            TileRenderResult result = TileRenderer::renderTile(params, surface, volume.get());
            result.controllerId = controllerId;
            // Convert QImage→QPixmap on worker thread to avoid GPU upload
            // stalls on the main thread during drain.
            if (!result.image.isNull()) {
                result.pixmap = QPixmap::fromImage(result.image, Qt::NoFormatConversion);
                result.image = QImage();  // release QImage memory
            }

            pushResult(std::move(result));
        });
}

std::vector<TileRenderResult> RenderPool::drainCompleted(int maxResults, uint64_t minEpoch, int controllerId)
{
    // Reset debounce flag so the next pushResult emits tileReady again
    resultSignalPending_.store(false, std::memory_order_relaxed);

    std::vector<TileRenderResult> results;
    std::lock_guard<std::mutex> lock(resultsMutex_);

    results.reserve(std::min(static_cast<int>(completedResults_.size()), maxResults));

    // Accept results from recent epochs (within kEpochSlack of minEpoch).
    // During rapid interaction the epoch advances quickly, but nearby-epoch
    // results are still visually useful and prevent gray/empty tiles.
    constexpr uint64_t kEpochSlack = 5;
    uint64_t effectiveMin = (minEpoch > kEpochSlack) ? minEpoch - kEpochSlack : 0;

    // Use reverse iteration with swap-and-pop to avoid O(n) shifts
    for (int i = static_cast<int>(completedResults_.size()) - 1;
         i >= 0 && static_cast<int>(results.size()) < maxResults; --i) {
        auto& item = completedResults_[i];
        if (item.controllerId != controllerId) {
            continue;  // belongs to another controller, skip
        }
        if (item.epoch >= effectiveMin) {
            results.push_back(std::move(item));
        }
        // Remove by swapping with back and popping (O(1) per element)
        completedResults_[i] = std::move(completedResults_.back());
        completedResults_.pop_back();
    }

    return results;
}

void RenderPool::cancelAll()
{
    pool_->cancel_pending();

    {
        std::lock_guard<std::mutex> lock(resultsMutex_);
        completedResults_.clear();
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

    // Debounce: only emit one tileReady per drain cycle. Without this,
    // 128 tiles completing in rapid succession queue 128 Qt events that
    // compete with input events and cause UI stalls.
    if (!resultSignalPending_.exchange(true, std::memory_order_relaxed)) {
        emit tileReady();
    }
}
