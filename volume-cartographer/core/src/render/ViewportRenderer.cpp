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

ViewportRenderer::~ViewportRenderer() noexcept
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
    _currentEpoch->store(camera.epoch, std::memory_order_relaxed);
    _desiredLevel = camera.dsScaleIdx;
    _lastCamera = camera;
    _lastSurface = surface;
    _lastVolume = volume;
    _lastBuildParams = buildParams;
    _lastVpL = vpLeft;
    _lastVpT = vpTop;
    _lastVpR = vpRight;
    _lastVpB = vpBottom;

    // Get visible tiles and submit them all
    auto visibleKeys = _tileGrid.visibleTiles(vpLeft, vpTop, vpRight, vpBottom,
                                               tiled_config::VISIBLE_BUFFER_TILES);

    fprintf(stderr, "[cam] scale=%.3f visible=%zu vp=[%.0f,%.0f,%.0f,%.0f] pending=%d\n",
        camera.scale, visibleKeys.size(), vpLeft, vpTop, vpRight, vpBottom, pool()->pendingCount());

    for (const auto& wk : visibleKeys) {
        TileRenderParams params = buildParams(wk);
        pool()->submit(params, surface, volume, _currentEpoch, _controllerId);
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

    _chunkArrived.store(false, std::memory_order_relaxed);
}

int ViewportRenderer::pendingCount() const noexcept
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
    _chunkArrived.store(true, std::memory_order_release);
}

void ViewportRenderer::markOverlaysDirty()
{
    _overlaysDirty.store(true, std::memory_order_release);
}

void ViewportRenderer::drainResults()
{
    auto results = pool()->drainCompleted(
        _currentEpoch->load(std::memory_order_relaxed),
        _controllerId);

    bool anyUpdated = false;

    for (auto& result : results) {
        _inFlightTiles.erase(result.worldKey);

        if (!result.pixels.empty()) {
            // Update metadata (best-effort — may fail if grid bounds changed)
            _tileGrid.setTileMeta(
                result.worldKey, result.epoch,
                static_cast<int8_t>(result.actualLevel));

            // Always blit — display uses camera-relative positioning,
            // not grid bounds. Don't gate on metadata acceptance.
            anyUpdated = true;
            if (_resultCallback)
                _resultCallback(std::move(result));
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


    // 2. Check if chunks arrived (atomic test-and-clear to avoid TOCTOU
    //    with markChunkArrived() called from signal handlers on other threads)
    bool chunksJustArrived = _chunkArrived.exchange(false, std::memory_order_acq_rel);

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

    // 4. Overlay update (atomic test-and-clear)
    if (_overlaysDirty.exchange(false, std::memory_order_acq_rel)) {
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
