#pragma once

#include <QObject>
#include <QPixmap>
#include <QRectF>

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "SliceCache.hpp"
#include "RenderPool.hpp"
#include "TileScene.hpp"
#include "TiledViewerCamera.hpp"
#include "TileRenderer.hpp"

class QTimer;
class Surface;
class Volume;

// Orchestrates tile rendering: checks cache, submits misses to background
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
    // For each visible tile:
    //   - Cache hit -> apply pixmap immediately
    //   - Cache miss -> submit to background pool
    void onCameraChanged(const TiledViewerCamera& camera,
                         const std::shared_ptr<Surface>& surface,
                         const std::shared_ptr<Volume>& volume,
                         const std::function<TileRenderParams(const WorldTileKey&)>& buildParams,
                         const QRectF& viewportRect);

    // Called when rendering parameters change (window/level, colormap, etc.)
    // Clears the slice cache and re-renders everything.
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

    // Access the slice cache (for stats, manual invalidation, etc.)
    SliceCache& sliceCache() { return _cache; }

    // Hash rendering parameters for cache key generation.
    // Should be recomputed whenever window/level, colormap, composite settings change.
    void setParamsHash(uint64_t hash) { _paramsHash = hash; }
    uint64_t paramsHash() const { return _paramsHash; }

    // Progressive rendering: show coarse previews while full-res loads
    void setProgressiveEnabled(bool enabled) { _progressiveEnabled = enabled; }
    bool progressiveEnabled() const { return _progressiveEnabled; }

    // For the next epoch-changing render, keep the old frame visible until the
    // visible tile set for the new epoch has been resolved.
    void setAtomicNextEpochSwap(bool enabled) { _atomicNextEpochSwap = enabled; }

    // --- Dirty flags (set by viewer, processed each tick) ---
    void markOverlaysDirty();
    void markChunkArrived();
    void startZoomSettle();
    void cancelZoomSettle();
    void setOverlayCallback(std::function<void()> cb);
    void setZoomSettleCallback(std::function<void()> cb);

signals:
    // Emitted when drainResults() actually updated tile pixmaps.
    // Connect to viewport()->update() to guarantee repaints during
    // progressive refinement (chunk arrival while the view is idle).
    void sceneNeedsUpdate();

private slots:
    // Drain completed results from the render pool and update tile scene.
    void drainResults();

    // Unified tick: runs at ~30 Hz, handles all periodic work.
    void tick();

    // Start the tick timer if not already running.
    void ensureTickRunning();

private:
    struct PendingSwapTile {
        bool ready = false;
        bool hasPixmap = false;
        QPixmap pixmap;
        int8_t level = -1;
    };

    TileScene* _tileScene;
    SliceCache _cache;
    RenderPool* _renderPool;  // shared, not owned
    QTimer* _tickTimer;

    std::shared_ptr<std::atomic<uint64_t>> _currentEpoch = std::make_shared<std::atomic<uint64_t>>(0);
    int _controllerId;
    static inline std::atomic<int> _nextControllerId{0};
    uint64_t _paramsHash = 0;
    QRectF _lastViewportRect;
    bool _progressiveEnabled = true;
    bool _atomicNextEpochSwap = false;

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

    // Dirty flags set by the viewer, processed each tick
    bool _overlaysDirty = false;
    bool _chunkArrived = false;
    bool _zoomSettlePending = false;
    uint64_t _zoomSettleTicksLeft = 0;  // countdown in ticks (~6 ticks = 200ms)

    // Callbacks to notify the viewer
    std::function<void()> _overlayCallback;
    std::function<void()> _zoomSettleCallback;

    // Atomic epoch swap: collect all tiles for a new epoch before displaying
    bool _pendingSwapActive = false;
    uint64_t _pendingSwapEpoch = 0;
    std::unordered_map<WorldTileKey, PendingSwapTile, WorldTileKeyHash> _pendingSwapTiles;

    void beginPendingSwap(uint64_t epoch, const std::vector<WorldTileKey>& keys);
    void stagePendingSwapTile(const WorldTileKey& wk, const QPixmap& pixmap,
                              int8_t level, bool hasPixmap);
    bool tryCommitPendingSwap();
    static QPixmap placeholderPixmap();
};
