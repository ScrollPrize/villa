#pragma once

#include <QObject>
#include <QRectF>

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "RenderPool.hpp"
#include "TileScene.hpp"
#include "TiledViewerCamera.hpp"
#include "TileRenderer.hpp"

class QTimer;
class Surface;
class Volume;

// Orchestrates tile rendering: submits visible tiles to the background
// pool, drains completed results, and updates the tile scene.
//
// All public methods must be called from the main thread.
class TileRenderController : public QObject
{
    Q_OBJECT

public:
    explicit TileRenderController(TileScene* tileScene, RenderPool* sharedPool, QObject* parent = nullptr);
    ~TileRenderController() override;

    // Called when camera state changes (pan, zoom, slice offset).
    // Submits visible tiles to the background pool (skipping in-flight duplicates).
    void onCameraChanged(const TiledViewerCamera& camera,
                         const std::shared_ptr<Surface>& surface,
                         const std::shared_ptr<Volume>& volume,
                         const std::function<TileRenderParams(const WorldTileKey&)>& buildParams,
                         const QRectF& viewportRect);

    // Called when rendering parameters change (window/level, colormap, etc.)
    // Re-renders everything.
    void onParamsChanged(const TiledViewerCamera& camera,
                         const std::shared_ptr<Surface>& surface,
                         const std::shared_ptr<Volume>& volume,
                         const std::function<TileRenderParams(const WorldTileKey&)>& buildParams,
                         const QRectF& viewportRect);

    // Deferred render: stores state and processes on next drain tick.
    // Rapid calls coalesce into a single onCameraChanged().
    void scheduleRender(const TiledViewerCamera& camera,
                        const std::shared_ptr<Surface>& surface,
                        const std::shared_ptr<Volume>& volume,
                        const std::function<TileRenderParams(const WorldTileKey&)>& buildParams,
                        const QRectF& viewportRect);

    // Cancel all in-flight renders and clear results.
    void cancelAll();

    // Clear cached render state (_last* and _pending*) so stale callbacks
    // (e.g. chunk-arrival) cannot re-trigger renders with an old volume.
    void clearState();

    RenderPool* renderPool() const { return _renderPool; }

    // Progressive rendering: show coarse previews while full-res loads
    void setProgressiveEnabled(bool enabled) { _progressiveEnabled = enabled; }
    bool progressiveEnabled() const { return _progressiveEnabled; }

    // --- Dirty flags (set by viewer, processed each tick) ---
    void markOverlaysDirty();
    void markChunkArrived();
    void setOverlayCallback(std::function<void()> cb);

signals:
    // Emitted when drainResults() actually updated tile pixmaps.
    // Connect to viewport()->update() to guarantee repaints during
    // progressive refinement (chunk arrival while the view is idle).
    void sceneNeedsUpdate();

private slots:
    // Drain completed results from the render pool and update tile scene.
    void drainResults();

    // Unified tick: runs at ~60 Hz, handles all periodic work.
    void tick();

    // Start the tick timer if not already running.
    void ensureTickRunning();

private:
    TileScene* _tileScene;
    RenderPool* _renderPool;  // shared, not owned
    QTimer* _tickTimer;

    std::shared_ptr<std::atomic<uint64_t>> _currentEpoch = std::make_shared<std::atomic<uint64_t>>(0);
    int _controllerId;
    static inline std::atomic<int> _nextControllerId{0};
    QRectF _lastViewportRect;
    bool _progressiveEnabled = true;

    // Desired pyramid level for current render pass
    int _desiredLevel = 0;

    // Last render state for refinement re-submission
    TiledViewerCamera _lastCamera;
    std::shared_ptr<Surface> _lastSurface;
    std::shared_ptr<Volume> _lastVolume;
    std::function<TileRenderParams(const WorldTileKey&)> _lastBuildParams;

    // Pending state for coalescing rapid viewport changes
    bool _pendingDirty = false;
    TiledViewerCamera _pendingCamera;
    std::shared_ptr<Surface> _pendingSurface;
    std::shared_ptr<Volume> _pendingVolume;
    std::function<TileRenderParams(const WorldTileKey&)> _pendingBuildParams;
    QRectF _pendingViewportRect;

    // In-flight tile tracking to avoid duplicate submissions
    std::unordered_set<WorldTileKey, WorldTileKeyHash> _inFlightTiles;

    // Dirty flags set by the viewer, processed each tick
    bool _overlaysDirty = false;
    bool _chunkArrived = false;

    // Callback to notify the viewer
    std::function<void()> _overlayCallback;
};
