#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "vc/core/render/TileTypes.hpp"

namespace utils { class PriorityThreadPool; }
class Surface;
class Volume;

namespace vc::render {

// Background tile rendering pool (no Qt dependency).
// Workers render tiles off the main thread.  Completed results are
// collected by the caller via drainCompleted().
//
// Uses utils::PriorityThreadPool with epoch-based stale task filtering.
class CoreRenderPool {
public:
    explicit CoreRenderPool(int numThreads = 2);
    ~CoreRenderPool();

    // Submit a tile for background rendering.
    // epochRef is checked before/after rendering to skip stale tasks.
    // controllerId tags results so drainCompleted can filter by owner.
    void submit(const TileRenderParams& params,
                const std::shared_ptr<Surface>& surface,
                const std::shared_ptr<Volume>& volume,
                const std::shared_ptr<std::atomic<uint64_t>>& epochRef,
                int controllerId);

    // Take all completed results belonging to controllerId.
    // Results with epoch < minEpoch (minus slack) are discarded.
    std::vector<TileRenderResult> drainCompleted(uint64_t minEpoch, int controllerId);

    // Cancel all pending work and clear results.
    void cancelAll();

    // Number of pending + in-flight tasks.
    int pendingCount() const;

    // Reset stuck pending count when the pool is idle but pendingCount > 0.
    // Returns true if the count was reset.
    bool expireTimedOut();

    // Callback invoked from a worker thread when a result is ready.
    // Use to wake the main thread (e.g. ensureTickRunning).
    void setReadyCallback(std::function<void()> cb);

private:
    void pushResult(TileRenderResult result);

    std::unique_ptr<utils::PriorityThreadPool> pool_;
    std::mutex resultsMutex_;
    std::vector<TileRenderResult> completedResults_;
    std::atomic<int> pendingCount_{0};
    std::atomic<bool> resultSignalPending_{false};
    std::function<void()> readyCallback_;
};

} // namespace vc::render
