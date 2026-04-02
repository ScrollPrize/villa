#pragma once

#include <QObject>

#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

#include "TileRenderer.hpp"
#include "vc/core/render/CoreRenderPool.hpp"

class Surface;
class Volume;

// Background tile rendering pool (Qt wrapper around CoreRenderPool).
// Workers render tiles off the main thread. Completed results are
// collected by the main thread via drainCompleted(), which converts
// raw pixel buffers to QPixmaps.
//
// Internally delegates to vc::render::CoreRenderPool for thread pool
// management, task submission, and result storage.
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
    // Converts raw pixel buffers to QPixmaps on the calling thread.
    std::vector<QtTileRenderResult> drainCompleted(int maxResults, uint64_t minEpoch, int controllerId);

    // Cancel all pending work and clear results.  With a shared pool this
    // only resets bookkeeping — it does NOT call cancel_pending on the
    // underlying thread pool (that would kill other controllers' tasks).
    void cancelAll();

    // Number of pending + in-flight tasks
    int pendingCount() const;

    // Reset stuck pending count when the pool is idle but pendingCount > 0
    // (tasks lost to epoch filtering without going through pushResult).
    // Returns true if the count was reset.
    bool expireTimedOut();

    // Access the underlying core pool (for ViewportRenderer integration)
    vc::render::CoreRenderPool& corePool() { return *corePool_; }
    const vc::render::CoreRenderPool& corePool() const { return *corePool_; }

signals:
    // Emitted (from worker thread) when a result is ready.
    // Connect with Qt::QueuedConnection to receive on main thread.
    void tileReady();

private:
    std::unique_ptr<vc::render::CoreRenderPool> corePool_;
};
