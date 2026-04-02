#include "vc/core/render/ViewportRenderer.hpp"

#include <algorithm>
#include <cmath>

#include "vc/core/cache/TieredChunkCache.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/Surface.hpp"

namespace vc::render {

ViewportRenderer::ViewportRenderer(int numThreads)
    : _renderPool(std::make_unique<CoreRenderPool>(numThreads))
    , _externalPool(nullptr)
    , _controllerId(_nextControllerId.fetch_add(1, std::memory_order_relaxed))
{
}

ViewportRenderer::ViewportRenderer(CoreRenderPool* externalPool)
    : _externalPool(externalPool)
    , _controllerId(_nextControllerId.fetch_add(1, std::memory_order_relaxed))
{
}

ViewportRenderer::~ViewportRenderer()
{
    _currentEpoch->fetch_add(1, std::memory_order_relaxed);
}

void ViewportRenderer::onCameraChanged(
    const TiledViewerCamera& camera,
    const std::shared_ptr<Surface>& surface,
    const std::shared_ptr<Volume>& volume,
    const std::function<TileRenderParams(const WorldTileKey&)>& buildParams,
    float vpLeft, float vpTop, float vpRight, float vpBottom)
{
    bool epochChanged = (camera.epoch != _currentEpoch->load(std::memory_order_relaxed));

    if (!epochChanged &&
        vpLeft == _lastVpL && vpTop == _lastVpT &&
        vpRight == _lastVpR && vpBottom == _lastVpB) {
        return;
    }

    _currentEpoch->store(camera.epoch, std::memory_order_relaxed);
    _desiredLevel = camera.dsScaleIdx;

    if (epochChanged) {
        _inFlightTiles.clear();
    } else {
        if (pool()->pendingCount() == 0)
            _inFlightTiles.clear();
    }

    _lastCamera = camera;
    _lastSurface = surface;
    _lastVolume = volume;
    _lastBuildParams = buildParams;
    _lastVpL = vpLeft;
    _lastVpT = vpTop;
    _lastVpR = vpRight;
    _lastVpB = vpBottom;

    auto visibleKeys = _tileGrid.visibleTiles(vpLeft, vpTop, vpRight, vpBottom,
                                               tiled_config::VISIBLE_BUFFER_TILES);

    // Compute chunk group size in tile units.
    int chunkGroupSize = 4;
    if (volume && volume->tieredCache()) {
        auto cs = volume->tieredCache()->chunkShape(camera.dsScaleIdx);
        float worldTileSize = _tileGrid.bounds().worldTileSize;
        if (worldTileSize > 0)
            chunkGroupSize = std::max(1, static_cast<int>(cs[1] / worldTileSize));
    }

    // Sort by chunk group (center-first within groups) for cache locality.
    {
        float vcx = (vpLeft + vpRight) * 0.5f / TileGrid::TILE_PX;
        float vcy = (vpTop + vpBottom) * 0.5f / TileGrid::TILE_PX;
        int cgs = chunkGroupSize;
        auto floorDiv = [](int a, int b) { return (a >= 0) ? a / b : (a - b + 1) / b; };
        std::sort(visibleKeys.begin(), visibleKeys.end(),
            [vcx, vcy, cgs, floorDiv](const WorldTileKey& a, const WorldTileKey& b) {
                int gaCol = floorDiv(a.worldCol, cgs);
                int gaRow = floorDiv(a.worldRow, cgs);
                int gbCol = floorDiv(b.worldCol, cgs);
                int gbRow = floorDiv(b.worldRow, cgs);
                if (gaCol != gbCol || gaRow != gbRow) {
                    float gaCx = gaCol * cgs + cgs * 0.5f;
                    float gaCy = gaRow * cgs + cgs * 0.5f;
                    float gbCx = gbCol * cgs + cgs * 0.5f;
                    float gbCy = gbRow * cgs + cgs * 0.5f;
                    float dga = (gaCx - vcx) * (gaCx - vcx) + (gaCy - vcy) * (gaCy - vcy);
                    float dgb = (gbCx - vcx) * (gbCx - vcx) + (gbCy - vcy) * (gbCy - vcy);
                    return dga < dgb;
                }
                float da = (a.worldCol - vcx) * (a.worldCol - vcx)
                         + (a.worldRow - vcy) * (a.worldRow - vcy);
                float db = (b.worldCol - vcx) * (b.worldCol - vcx)
                         + (b.worldRow - vcy) * (b.worldRow - vcy);
                return da < db;
            });
    }

    const uint64_t epoch = _currentEpoch->load(std::memory_order_relaxed);
    int submitOrder = 0;

    for (const auto& wk : visibleKeys) {
        if (_inFlightTiles.find(wk) == _inFlightTiles.end()) {
            TileRenderParams params = buildParams(wk);
            params.submitPriority = -params.dsScaleIdx * 10000 + submitOrder;
            pool()->submit(params, surface, volume, _currentEpoch, _controllerId);
            _inFlightTiles.insert(wk);
            ++submitOrder;
        }
    }
}

void ViewportRenderer::onParamsChanged(
    const TiledViewerCamera& camera,
    const std::shared_ptr<Surface>& surface,
    const std::shared_ptr<Volume>& volume,
    const std::function<TileRenderParams(const WorldTileKey&)>& buildParams,
    float vpLeft, float vpTop, float vpRight, float vpBottom)
{
    onCameraChanged(camera, surface, volume, buildParams, vpLeft, vpTop, vpRight, vpBottom);
}

void ViewportRenderer::scheduleRender(
    const TiledViewerCamera& camera,
    const std::shared_ptr<Surface>& surface,
    const std::shared_ptr<Volume>& volume,
    const std::function<TileRenderParams(const WorldTileKey&)>& buildParams,
    float vpLeft, float vpTop, float vpRight, float vpBottom)
{
    if (_pendingDirty && camera.epoch == _pendingCamera.epoch &&
        std::abs(camera.zOff - _pendingCamera.zOff) < 1e-6f &&
        std::abs(camera.scale - _pendingCamera.scale) < 1e-6f) {
        return;
    }
    _pendingCamera = camera;
    _pendingSurface = surface;
    _pendingVolume = volume;
    _pendingBuildParams = buildParams;
    _pendingVpL = vpLeft;
    _pendingVpT = vpTop;
    _pendingVpR = vpRight;
    _pendingVpB = vpBottom;
    _pendingDirty = true;
}

void ViewportRenderer::cancelAll()
{
    _currentEpoch->fetch_add(1, std::memory_order_relaxed);
}

void ViewportRenderer::clearState()
{
    _lastSurface.reset();
    _lastVolume.reset();
    _lastBuildParams = nullptr;
    _lastVpL = _lastVpT = _lastVpR = _lastVpB = 0;

    _pendingDirty = false;
    _pendingSurface.reset();
    _pendingVolume.reset();
    _pendingBuildParams = nullptr;

    _chunkArrived = false;
}

int ViewportRenderer::pendingCount() const
{
    return pool()->pendingCount();
}

void ViewportRenderer::setTileReadyCallback(std::function<void()> cb)
{
    pool()->setReadyCallback(std::move(cb));
}

void ViewportRenderer::setSceneUpdatedCallback(std::function<void()> cb)
{
    _sceneUpdatedCallback = std::move(cb);
}

void ViewportRenderer::setOverlayCallback(std::function<void()> cb)
{
    _overlayCallback = std::move(cb);
}

void ViewportRenderer::markChunkArrived()
{
    _chunkArrived = true;
}

void ViewportRenderer::markOverlaysDirty()
{
    _overlaysDirty = true;
}

void ViewportRenderer::drainResults()
{
    auto results = pool()->drainCompleted(
        tiled_config::DRAIN_BATCH_SIZE,
        _currentEpoch->load(std::memory_order_relaxed),
        _controllerId);

    bool anyUpdated = false;

    for (auto& result : results) {
        _inFlightTiles.erase(result.worldKey);

        if (!result.pixels.empty() &&
            _tileGrid.setTileWorld(result.worldKey, std::move(result.pixels),
                                   result.width, result.height,
                                   result.epoch,
                                   static_cast<int8_t>(result.actualLevel))) {
            anyUpdated = true;
        }
    }

    if (anyUpdated && _sceneUpdatedCallback)
        _sceneUpdatedCallback();

    if (_pendingDirty) {
        _pendingDirty = false;
        onCameraChanged(_pendingCamera, _pendingSurface,
                        _pendingVolume, _pendingBuildParams,
                        _pendingVpL, _pendingVpT, _pendingVpR, _pendingVpB);
        _pendingSurface.reset();
        _pendingVolume.reset();
        _pendingBuildParams = nullptr;
    }
}

bool ViewportRenderer::tick()
{
    bool moreWork = false;

    // 1. Drain completed render results
    int drainedBefore = pool()->pendingCount();
    drainResults();
    int drainedAfter = pool()->pendingCount();
    bool drainedSomething = drainedAfter < drainedBefore;

    // 2. Check if chunks arrived
    bool chunksJustArrived = _chunkArrived;
    if (_chunkArrived)
        _chunkArrived = false;

    // 3. Progressive refinement
    bool poolIdle = pool()->pendingCount() == 0;
    bool shouldRefine = chunksJustArrived || (poolIdle && !_pendingDirty);
    if (_progressiveEnabled && shouldRefine) {
        if (_lastSurface && _lastVolume && _lastBuildParams) {
            auto stale = _tileGrid.staleTilesInRect(
                _desiredLevel, _currentEpoch->load(std::memory_order_relaxed),
                _lastVpL, _lastVpT, _lastVpR, _lastVpB,
                tiled_config::VISIBLE_BUFFER_TILES);
            if (!stale.empty()) {
                const uint64_t epoch = _currentEpoch->load(std::memory_order_relaxed);
                int submitted = 0;
                for (const auto& wk : stale) {
                    if (submitted >= 32) break;
                    if (_inFlightTiles.count(wk)) continue;
                    TileRenderParams params = _lastBuildParams(wk);
                    params.epoch = epoch;
                    params.submitPriority = 1000 + submitted;
                    pool()->submit(params, _lastSurface, _lastVolume, _currentEpoch, _controllerId);
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

    pool()->expireTimedOut();

    if (pool()->pendingCount() > 0 || _pendingDirty)
        moreWork = true;

    if (!moreWork && !drainedSomething && !chunksJustArrived)
        return false;  // caller can stop ticking

    return true;
}

} // namespace vc::render
