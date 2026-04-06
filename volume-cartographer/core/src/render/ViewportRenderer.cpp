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
    pool()->takeResults(_controllerId);
    ++_batch;

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
        params.batch = _batch;
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
    pool()->takeResults(_controllerId); // discard completed stale results
    ++_batch;

    _desiredLevel = camera.dsScaleIdx;
    _lastCamera = camera;
    _lastSurface = surface;
    _lastVolume = volume;
    _lastBuildParams = buildParams;
    _lastVisibleKeys = visibleKeys;

    for (const auto& wk : visibleKeys) {
        TileRenderParams params = buildParams(wk);
        params.batch = _batch;
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
    onCameraChanged(camera, surface, volume, buildParams, vpLeft, vpTop, vpRight, vpBottom);
}

void ViewportRenderer::cancelAll()
{
    pool()->clearQueue();
    pool()->takeResults(_controllerId); // discard this view's stale results
}

void ViewportRenderer::clearState()
{
    _lastSurface.reset();
    _lastVolume.reset();
    _lastBuildParams = nullptr;
    _lastVpL = _lastVpT = _lastVpR = _lastVpB = 0;
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

void ViewportRenderer::markChunkArrived(int chunkLevel)
{
    // Only trigger re-render if the chunk could improve what's displayed.
    // Chunks coarser than the desired level won't help. Chunks at the
    // desired level or 1-2 levels coarser are useful (best-effort fallback).
    if (chunkLevel <= _desiredLevel + 2) {
        _chunkArrived.store(true, std::memory_order_release);
    }
}

void ViewportRenderer::markOverlaysDirty()
{
    _overlaysDirty.store(true, std::memory_order_release);
}

bool ViewportRenderer::tick()
{
    // Drain completed results and blit them
    auto results = pool()->takeResults(_controllerId);
    bool anyUpdated = false;
    for (auto& result : results) {
        if (!result.pixels.empty()) {
            anyUpdated = true;
            if (_resultCallback)
                _resultCallback(std::move(result));
        }
    }
    if (anyUpdated && _sceneUpdatedCallback)
        _sceneUpdatedCallback();

    return pool()->busy();
}

} // namespace vc::render
