#pragma once

#include <QObject>

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <vector>

#include "TileRenderer.hpp"
#include <utils/thread_pool.hpp>

class Surface;
class Volume;

// Background tile rendering pool.
// Workers render tiles off the main thread. Completed results are
// collected by the main thread via drainCompleted().
//
// Internally uses utils::PriorityThreadPool with epoch-based stale
// task filtering to replace QThreadPool.
class RenderPool : public QObject
{
    Q_OBJECT

public:
    explicit RenderPool(int numThreads = 2, QObject* parent = nullptr);
    ~RenderPool() override;

    // Submit a tile for background rendering.
    // The epochRef atomic is checked before and after rendering to skip stale tasks.
    // controllerId tags results so drainCompleted can filter by owner.
    void submit(const TileRenderParams& params,
                const std::shared_ptr<Surface>& surface,
                const std::shared_ptr<Volume>& volume,
                const std::shared_ptr<std::atomic<uint64_t>>& epochRef,
                int controllerId);

    // Take up to maxResults completed results belonging to controllerId.
    // Results with epoch < minEpoch are discarded.
    std::vector<TileRenderResult> drainCompleted(int maxResults, uint64_t minEpoch, int controllerId);

    // Cancel all pending work and clear results.  With a shared pool this
    // only resets bookkeeping — it does NOT call cancel_pending on the
    // underlying thread pool (that would kill other controllers' tasks).
    void cancelAll();

    // Number of pending + in-flight tasks
    int pendingCount() const;

    // Check for timed-out tasks. If pending tasks have been waiting longer
    // than the timeout and the pool is idle, reset the pending count.
    // Returns true if any tasks were expired.
    bool expireTimedOut(std::chrono::steady_clock::duration timeout = std::chrono::seconds(5));

signals:
    // Emitted (from worker thread) when a result is ready.
    // Connect with Qt::QueuedConnection to receive on main thread.
    void tileReady();

private:
    // Called from worker threads when a render completes.
    void pushResult(TileRenderResult result);

    std::unique_ptr<utils::PriorityThreadPool> pool_;
    std::mutex resultsMutex_;
    std::vector<TileRenderResult> completedResults_;
    std::atomic<int> pendingCount_{0};

    // Track the oldest submission time to detect timed-out renders.
    std::mutex timeMutex_;
    std::chrono::steady_clock::time_point oldestSubmitTime_;
    bool hasSubmissions_{false};
};
