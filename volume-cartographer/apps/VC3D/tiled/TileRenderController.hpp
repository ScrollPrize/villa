#pragma once

#include <QObject>
#include <QRectF>

#include <cstdint>
#include <functional>
#include <memory>

#include "RenderPool.hpp"
#include "TileScene.hpp"
#include "TiledViewerCamera.hpp"
#include "TileRenderer.hpp"
#include "vc/core/render/ViewportRenderer.hpp"

class Surface;
class Volume;

// Orchestrates tile rendering: delegates to the platform-agnostic
// ViewportRenderer for tile scheduling, draining, and progressive
// refinement, and syncs results to TileScene's QGraphicsPixmapItems.
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

    // Same but with pre-computed visible keys (bypasses grid coordinate system).
    void onCameraChangedDirect(const TiledViewerCamera& camera,
                               const std::shared_ptr<Surface>& surface,
                               const std::shared_ptr<Volume>& volume,
                               const std::function<TileRenderParams(const WorldTileKey&)>& buildParams,
                               const std::vector<WorldTileKey>& visibleKeys);

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
    void setProgressiveEnabled(bool enabled) { _viewportRenderer.setProgressiveEnabled(enabled); }
    bool progressiveEnabled() const { return _viewportRenderer.progressiveEnabled(); }

    // --- Dirty flags (set by viewer, processed each tick) ---
    void markOverlaysDirty();
    void markChunkArrived(int chunkLevel = 0);
    void setOverlayCallback(std::function<void()> cb);

    // Rebuild ViewportRenderer's TileGrid (the authoritative grid).
    // Call alongside TileScene::rebuildGrid() which just resizes the framebuffer.
    void syncGridBounds(const ContentBounds& bounds, int viewportW, int viewportH);

    // Access the underlying ViewportRenderer
    vc::render::ViewportRenderer& viewportRenderer() { return _viewportRenderer; }
    const vc::render::ViewportRenderer& viewportRenderer() const { return _viewportRenderer; }

signals:
    // Emitted when drainResults() actually updated tile pixmaps.
    // Connect to viewport()->update() to guarantee repaints during
    // progressive refinement (chunk arrival while the view is idle).
    void sceneNeedsUpdate();

private slots:
    // Vsync-gated tick: runs at display refresh rate (60fps = 16ms).
    // Drains all completed render results, syncs to scene, triggers repaint.
    void tick();

    // Start the vsync timer if not already running.
    void ensureTickRunning();

private:
    TileScene* _tileScene;
    RenderPool* _renderPool;  // shared, not owned
    bool _tickPending = false;
    bool _anyUpdatedThisTick = false;
    QTimer* _vsyncTimer = nullptr;

    vc::render::ViewportRenderer _viewportRenderer;
};
