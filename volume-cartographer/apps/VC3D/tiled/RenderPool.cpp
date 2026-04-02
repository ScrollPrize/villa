#include "RenderPool.hpp"

#include <QImage>

#include "vc/core/util/Surface.hpp"
#include "vc/core/types/Volume.hpp"

static constexpr uint64_t kEpochSlack = 5;

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

    // Use the caller-provided priority which encodes both pyramid level
    // and spatial locality (chunk batching).  Coarser levels and
    // chunk-grouped tiles get lower values (= higher urgency).
    int priority = params.submitPriority;

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
            uint64_t currentEpoch = epochRef->load(std::memory_order_relaxed);
            if (currentEpoch > kEpochSlack && params.epoch < currentEpoch - kEpochSlack) {
                pendingCount_.fetch_sub(1, std::memory_order_relaxed);
                return;
            }

            TileRenderResult result = TileRenderer::renderTile(params, surface, volume.get());
            result.controllerId = controllerId;
            // Convert raw ARGB32 pixels -> QPixmap on worker thread to
            // avoid GPU upload stalls on the main thread during drain.
            if (!result.pixels.empty()) {
                QImage img(reinterpret_cast<const uchar*>(result.pixels.data()),
                           result.width, result.height,
                           result.width * 4, QImage::Format_RGB32);
                result.pixmap = QPixmap::fromImage(img, Qt::NoFormatConversion);
                result.pixels.clear();
                result.pixels.shrink_to_fit();
            }

            pushResult(std::move(result));
        });
}

std::vector<TileRenderResult> RenderPool::drainCompleted(int maxResults, uint64_t minEpoch, int controllerId)
{
    // Reset debounce flag so the next pushResult emits tileReady again
    resultSignalPending_.store(false, std::memory_order_relaxed);

    // Swap the entire vector out under lock — O(1) critical section.
    std::vector<TileRenderResult> all;
    {
        std::lock_guard<std::mutex> lock(resultsMutex_);
        std::swap(all, completedResults_);
    }

    // Filter outside the lock
    std::vector<TileRenderResult> results;
    results.reserve(std::min(static_cast<int>(all.size()), maxResults));

    // Accept results from recent epochs (within kEpochSlack of minEpoch).
    // During rapid interaction the epoch advances quickly, but nearby-epoch
    // results are still visually useful and prevent gray/empty tiles.
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
        // else: discard stale results for this controller
    }

    // Push non-matching items back under lock — O(1) if empty, O(n) insert
    // but only for items belonging to other controllers.
    if (!remaining.empty()) {
        std::lock_guard<std::mutex> lock(resultsMutex_);
        completedResults_.insert(completedResults_.end(),
            std::make_move_iterator(remaining.begin()),
            std::make_move_iterator(remaining.end()));
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

bool RenderPool::expireTimedOut()
{
    int pending = pendingCount_.load(std::memory_order_relaxed);
    if (pending <= 0)
        return false;

    // Only expire if the pool is idle (no workers actively running tasks)
    // AND no tasks are queued. This means the pending count is stuck
    // (tasks lost to epoch filtering without going through pushResult).
    if (pool_->active() > 0 || pool_->pending() > 0)
        return false;

    pendingCount_.store(0, std::memory_order_relaxed);
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
    pendingCount_.fetch_sub(1, std::memory_order_acq_rel);

    // Debounce: only emit one tileReady per drain cycle. Without this,
    // 128 tiles completing in rapid succession queue 128 Qt events that
    // compete with input events and cause UI stalls.
    if (!resultSignalPending_.exchange(true, std::memory_order_relaxed)) {
        emit tileReady();
    }
}
