#include "TileRenderController.hpp"

#include <QDebug>
#include <QThread>
#include <QTimer>
#include <QImage>
#include <algorithm>
#include <cmath>
#include <vector>

#include "vc/core/cache/TieredChunkCache.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/Surface.hpp"

TileRenderController::TileRenderController(TileScene* tileScene, RenderPool* sharedPool, QObject* parent)
    : QObject(parent)
    , _tileScene(tileScene)
    , _renderPool(sharedPool)
    , _controllerId(_nextControllerId.fetch_add(1, std::memory_order_relaxed))
{
    // Tick timer (~60 Hz) handles periodic work; started on-demand, auto-stops
    // when idle to avoid burning CPU.
    _tickTimer = new QTimer(this);
    _tickTimer->setInterval(16);
    connect(_tickTimer, &QTimer::timeout, this, &TileRenderController::tick);
    // Not started here — ensureTickRunning() starts it when work arrives.

    // When a tile completes, wake the tick timer so it drains on the next cycle.
    // Don't call drainResults directly — the tick consolidates all drain + refinement
    // work, avoiding redundant lock acquisitions and double-draining.
    connect(_renderPool, &RenderPool::tileReady, this, &TileRenderController::ensureTickRunning,
            Qt::QueuedConnection);
}

TileRenderController::~TileRenderController()
{
    _tickTimer->stop();
    // Bump epoch so our in-flight tasks are discarded on completion.
    _currentEpoch->fetch_add(1, std::memory_order_relaxed);
}

void TileRenderController::ensureTickRunning()
{
    if (!_tickTimer->isActive())
        _tickTimer->start();
}

void TileRenderController::onCameraChanged(
    const TiledViewerCamera& camera,
    const std::shared_ptr<Surface>& surface,
    const std::shared_ptr<Volume>& volume,
    const std::function<TileRenderParams(const WorldTileKey&)>& buildParams,
    const QRectF& viewportRect)
{
    ensureTickRunning();
    bool epochChanged = (camera.epoch != _currentEpoch->load(std::memory_order_relaxed));

    // Early out if nothing changed (pan without movement, duplicate calls)
    if (!epochChanged && viewportRect == _lastViewportRect) {
        return;
    }

    _currentEpoch->store(camera.epoch, std::memory_order_relaxed);
    _desiredLevel = camera.dsScaleIdx;

    if (epochChanged) {
        _inFlightTiles.clear();  // new epoch → need fresh renders
    } else {
        // Same epoch but pool may have been flushed — prune in-flight set
        // against pool pending count to avoid permanent dedup blocks
        if (_renderPool->pendingCount() == 0) {
            _inFlightTiles.clear();
        }
    }

    // Store last render state for refinement retries
    _lastCamera = camera;
    _lastSurface = surface;
    _lastVolume = volume;
    _lastBuildParams = buildParams;
    _lastViewportRect = viewportRect;

    // Get visible tiles (+ buffer for smooth scrolling), sorted by spatial
    // chunk groups so tiles sharing the same volume chunks are submitted
    // consecutively and processed by the same worker thread (keeping the
    // ChunkSampler's per-thread cache warm).  Within each chunk group,
    // center-first ordering ensures the most important tiles render first.
    auto visibleKeys = _tileScene->visibleTiles(viewportRect, tiled_config::VISIBLE_BUFFER_TILES);
    // Compute chunk group size in tile units.  Chunk XY is typically 128
    // voxels; at pyramid level dsScaleIdx the effective chunk covers
    // 128 * 2^dsScaleIdx surface units.  Each tile covers worldTileSize
    // surface units.  Group tiles that fall within the same chunk.
    int chunkGroupSize = 4;  // default: group 4x4 tiles
    if (volume && volume->tieredCache()) {
        auto cs = volume->tieredCache()->chunkShape(camera.dsScaleIdx);
        float worldTileSize = _tileScene->bounds().worldTileSize;
        if (worldTileSize > 0) {
            // cs is {z,y,x}; use XY chunk extent in surface parameter units
            chunkGroupSize = std::max(1, static_cast<int>(cs[1] / worldTileSize));
        }
    }
    {
        float vcx = static_cast<float>(viewportRect.center().x()) / TileScene::TILE_PX;
        float vcy = static_cast<float>(viewportRect.center().y()) / TileScene::TILE_PX;
        int cgs = chunkGroupSize;
        // Floor-division helper for correct grouping with negative tile coords
        auto floorDiv = [](int a, int b) { return (a >= 0) ? a / b : (a - b + 1) / b; };
        std::sort(visibleKeys.begin(), visibleKeys.end(),
            [vcx, vcy, cgs, floorDiv](const WorldTileKey& a, const WorldTileKey& b) {
                // Primary key: chunk group — tiles in the same chunk group
                // are submitted consecutively to the pool.
                int gaCol = floorDiv(a.worldCol, cgs);
                int gaRow = floorDiv(a.worldRow, cgs);
                int gbCol = floorDiv(b.worldCol, cgs);
                int gbRow = floorDiv(b.worldRow, cgs);
                if (gaCol != gbCol || gaRow != gbRow) {
                    // Chunk group closer to viewport center wins
                    float gaCx = gaCol * cgs + cgs * 0.5f;
                    float gaCy = gaRow * cgs + cgs * 0.5f;
                    float gbCx = gbCol * cgs + cgs * 0.5f;
                    float gbCy = gbRow * cgs + cgs * 0.5f;
                    float dga = (gaCx - vcx) * (gaCx - vcx) + (gaCy - vcy) * (gaCy - vcy);
                    float dgb = (gbCx - vcx) * (gbCx - vcx) + (gbCy - vcy) * (gbCy - vcy);
                    return dga < dgb;
                }
                // Secondary: center-distance within the group
                float da = (a.worldCol - vcx) * (a.worldCol - vcx)
                         + (a.worldRow - vcy) * (a.worldRow - vcy);
                float db = (b.worldCol - vcx) * (b.worldCol - vcx)
                         + (b.worldRow - vcy) * (b.worldRow - vcy);
                return da < db;
            });
    }

    const uint64_t epoch = _currentEpoch->load(std::memory_order_relaxed);
    int submitOrder = 0;  // encodes chunk-grouped spatial locality in priority

    for (const auto& wk : visibleKeys) {
        // Submit to background pool for full-quality render (skip duplicates)
        if (_inFlightTiles.find(wk) == _inFlightTiles.end()) {
            TileRenderParams params = buildParams(wk);
            // Priority: primary = pyramid level (coarser first, negative),
            // secondary = submission order (preserves chunk-grouped sort).
            params.submitPriority = -params.dsScaleIdx * 10000 + submitOrder;
            _renderPool->submit(params, surface, volume, _currentEpoch, _controllerId);
            _inFlightTiles.insert(wk);
            ++submitOrder;
        }
    }
}

void TileRenderController::onParamsChanged(
    const TiledViewerCamera& camera,
    const std::shared_ptr<Surface>& surface,
    const std::shared_ptr<Volume>& volume,
    const std::function<TileRenderParams(const WorldTileKey&)>& buildParams,
    const QRectF& viewportRect)
{
    onCameraChanged(camera, surface, volume, buildParams, viewportRect);
}

void TileRenderController::scheduleRender(
    const TiledViewerCamera& camera,
    const std::shared_ptr<Surface>& surface,
    const std::shared_ptr<Volume>& volume,
    const std::function<TileRenderParams(const WorldTileKey&)>& buildParams,
    const QRectF& viewportRect)
{
    // Skip if camera state hasn't changed (avoids redundant copies)
    if (_pendingDirty && camera.epoch == _pendingCamera.epoch &&
        std::abs(camera.zOff - _pendingCamera.zOff) < 1e-6f &&
        std::abs(camera.scale - _pendingCamera.scale) < 1e-6f) {
        return;
    }
    _pendingCamera = camera;
    _pendingSurface = surface;
    _pendingVolume = volume;
    _pendingBuildParams = buildParams;
    _pendingViewportRect = viewportRect;
    _pendingDirty = true;
    ensureTickRunning();
}

void TileRenderController::cancelAll()
{
    // With a shared pool we can't cancel_pending() (it would kill other
    // controllers' tasks).  Instead bump the epoch so our in-flight and
    // queued tasks are discarded at completion time.
    _currentEpoch->fetch_add(1, std::memory_order_relaxed);
}

void TileRenderController::clearState()
{
    _lastSurface.reset();
    _lastVolume.reset();
    _lastBuildParams = nullptr;
    _lastViewportRect = QRectF();

    _pendingDirty = false;
    _pendingSurface.reset();
    _pendingVolume.reset();
    _pendingBuildParams = nullptr;

    _chunkArrived = false;
}

void TileRenderController::drainResults()
{
    // Take up to DRAIN_BATCH_SIZE results per drain cycle
    auto results = _renderPool->drainCompleted(tiled_config::DRAIN_BATCH_SIZE, _currentEpoch->load(std::memory_order_relaxed), _controllerId);

    bool anyUpdated = false;

    for (auto& result : results) {
        // Remove from in-flight tracking so tile can be re-submitted for refinement
        _inFlightTiles.erase(result.worldKey);

        // Apply pixmap to scene directly via world key
        if (!result.pixmap.isNull() &&
            _tileScene->setTileWorld(result.worldKey, result.pixmap, result.epoch,
                                     static_cast<int8_t>(result.actualLevel))) {
            anyUpdated = true;
        }
    }

    // Ensure the scene repaints after progressive refinement updates.
    // QGraphicsPixmapItem::setPixmap() should trigger this automatically,
    // but explicitly requesting an update guarantees the view refreshes
    // even when updates arrive while the view is idle.
    if (anyUpdated) {
        emit sceneNeedsUpdate();
    }

    // Process pending viewport change (coalesced)
    if (_pendingDirty) {
        _pendingDirty = false;
        onCameraChanged(_pendingCamera, _pendingSurface,
                        _pendingVolume, _pendingBuildParams, _pendingViewportRect);
        // Release references after dispatch
        _pendingSurface.reset();
        _pendingVolume.reset();
        _pendingBuildParams = nullptr;
    }
}

void TileRenderController::tick()
{
    bool moreWork = false;

    // 1. Drain completed render results
    drainResults();

    // 2. Check if chunks arrived → directly re-submit stale tiles using _last* state.
    //    Do NOT use the _pendingDirty mechanism — that path clears _pending* after
    //    dispatch, so a second chunk arrival would pass null state to onCameraChanged
    //    and corrupt _last*.
    bool chunksJustArrived = _chunkArrived;
    if (_chunkArrived) {
        _chunkArrived = false;
    }

    // 3. Progressive refinement: re-submit stale tiles when new chunks
    //    have arrived OR when the pool is idle and not actively interacting.
    bool poolIdle = _renderPool->pendingCount() == 0;
    bool shouldRefine = chunksJustArrived || (poolIdle && !_pendingDirty);
    if (_progressiveEnabled && shouldRefine) {
        if (_lastSurface && _lastVolume && _lastBuildParams) {
            auto stale = _tileScene->staleTilesInRect(
                _desiredLevel, _currentEpoch->load(std::memory_order_relaxed),
                _lastViewportRect, tiled_config::VISIBLE_BUFFER_TILES);
            if (!stale.empty()) {
                const uint64_t epoch = _currentEpoch->load(std::memory_order_relaxed);
                // Submit up to 32 stale tiles, skipping those already in-flight.
                // No sort needed — visibleTiles already returns center-first order.
                int submitted = 0;
                for (const auto& wk : stale) {
                    if (submitted >= 32) break;
                    if (_inFlightTiles.count(wk)) continue;
                    TileRenderParams params = _lastBuildParams(wk);
                    params.epoch = epoch;
                    // Refinement uses low priority (high value) so it
                    // doesn't compete with fresh camera-change renders.
                    params.submitPriority = 1000 + submitted;
                    _renderPool->submit(params, _lastSurface, _lastVolume, _currentEpoch, _controllerId);
                    _inFlightTiles.insert(wk);
                    ++submitted;
                }
                if (submitted > 0) moreWork = true;
            }
        }
    }

    // 4. Overlay update
    if (_overlaysDirty) {
        _overlaysDirty = false;
        if (_overlayCallback) _overlayCallback();
    }

    // Expire stuck pending counts (pool idle but pendingCount > 0 for too long)
    _renderPool->expireTimedOut();

    // Still have in-flight renders? Keep ticking to drain them.
    if (_renderPool->pendingCount() > 0 || _pendingDirty)
        moreWork = true;

    // Auto-stop when idle to avoid burning CPU
    if (!moreWork)
        _tickTimer->stop();
}

void TileRenderController::markOverlaysDirty()
{
    _overlaysDirty = true;
    ensureTickRunning();
}

void TileRenderController::markChunkArrived()
{
    _chunkArrived = true;
    // Do NOT bump _currentEpoch here.  Bumping the epoch invalidates ALL
    // in-flight renders (pre/post-render staleness checks in RenderPool and
    // the minEpoch filter in drainCompleted discard results whose epoch <
    // _currentEpoch).  Since chunks arrive continuously during progressive
    // loading, this caused a cascade where every chunk arrival killed all
    // queued renders, leaving tiles gray.
    //
    // Progressive refinement works without an epoch bump because
    // staleTilesInRect already detects tiles whose render level is coarser
    // than the desired level (m.level > desiredLevel), and setTile accepts
    // finer-level results at the same epoch (level < m.level passes).
    ensureTickRunning();
}

void TileRenderController::setOverlayCallback(std::function<void()> cb)
{
    _overlayCallback = std::move(cb);
}

