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

void ViewportRenderer::onCameraChanged(
    const TiledViewerCamera& camera,
    const std::shared_ptr<Surface>& surface,
    const std::shared_ptr<Volume>& volume,
    const std::function<TileRenderParams(const WorldTileKey&)>& buildParams,
    float vpLeft, float vpTop, float vpRight, float vpBottom)
{
    pool()->clearQueue();

    _desiredLevel = camera.dsScaleIdx;
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

    for (const auto& wk : visibleKeys) {
        TileRenderParams params = buildParams(wk);
        pool()->submit(params, surface, volume, _controllerId);
    }
}

void ViewportRenderer::onCameraChangedDirect(
    const TiledViewerCamera& camera,
    const std::shared_ptr<Surface>& surface,
    const std::shared_ptr<Volume>& volume,
    const std::function<TileRenderParams(const WorldTileKey&)>& buildParams,
    const std::vector<WorldTileKey>& visibleKeys)
{
    pool()->clearQueue();

    _desiredLevel = camera.dsScaleIdx;
    _lastCamera = camera;
    _lastSurface = surface;
    _lastVolume = volume;
    _lastBuildParams = buildParams;

    for (const auto& wk : visibleKeys) {
        TileRenderParams params = buildParams(wk);
        pool()->submit(params, surface, volume, _controllerId);
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
    pool()->clearQueue();
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
    auto results = pool()->takeResults(_controllerId);

    bool anyUpdated = false;

    for (auto& result : results) {
        if (!result.pixels.empty()) {
            _tileGrid.setTileMeta(
                result.worldKey, result.epoch,
                static_cast<int8_t>(result.actualLevel));

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
    // 1. Take completed results and blit them
    drainResults();

    // 2. Chunks arrived → re-render for finer data
    bool chunksJustArrived = _chunkArrived.exchange(false, std::memory_order_acq_rel);

    // 3. Progressive refinement: if pool is idle, submit stale tiles
    bool poolIdle = !pool()->busy();
    bool shouldRefine = chunksJustArrived || (poolIdle && !_pendingDirty);
    if (_progressiveEnabled && shouldRefine) {
        if (_lastSurface && _lastVolume && _lastBuildParams) {
            auto stale = _tileGrid.staleTilesInRect(
                _desiredLevel, 0,
                _lastVpL, _lastVpT, _lastVpR, _lastVpB,
                tiled_config::VISIBLE_BUFFER_TILES);
            if (!stale.empty()) {
                int submitted = 0;
                for (const auto& wk : stale) {
                    if (submitted >= 32) break;
                    TileRenderParams params = _lastBuildParams(wk);
                    params.submitPriority = 1000 + submitted;
                    pool()->submit(params, _lastSurface, _lastVolume, _controllerId);
                    ++submitted;
                }
                if (submitted > 0) return true;
            }
        }
    }

    // 4. Overlay update
    if (_overlaysDirty.exchange(false, std::memory_order_acq_rel)) {
        if (_overlayCallback) _overlayCallback();
    }

    if (pool()->busy() || _pendingDirty)
        return true;

    return false;
}

} // namespace vc::render
