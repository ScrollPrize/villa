#include "TileRenderController.hpp"

#include <QTimer>
#include <QImage>

#include "vc/core/types/Volume.hpp"
#include "vc/core/util/Surface.hpp"

TileRenderController::TileRenderController(TileScene* tileScene, RenderPool* sharedPool, QObject* parent)
    : QObject(parent)
    , _tileScene(tileScene)
    , _renderPool(sharedPool)
    , _viewportRenderer(&sharedPool->corePool())
{
    // 60fps vsync timer — runs only while there's work to do.
    _vsyncTimer = new QTimer(this);
    _vsyncTimer->setTimerType(Qt::PreciseTimer);
    _vsyncTimer->setInterval(16);  // ~60 Hz
    connect(_vsyncTimer, &QTimer::timeout, this, &TileRenderController::tick);

    // Wake the timer when workers complete tiles.
    connect(_renderPool, &RenderPool::tileReady, this, &TileRenderController::ensureTickRunning,
            Qt::QueuedConnection);

    // Direct pixel→QPixmap conversion during drain — skips TileGrid pixel storage.
    // Blit rendered tile pixels directly into the single framebuffer.
    // No per-tile QImage/QPixmap creation — just a memcpy into the right position.
    _viewportRenderer.setResultCallback([this](TileRenderResult&& result) {
        if (result.pixels.empty()) return;
        // Discard stale tiles from previous zoom levels — their worldCol
        // means a different surface position at the current worldTileSize.
        if (std::abs(result.scale - _tileScene->camScale()) > 0.01f) return;
        _tileScene->blitTile(result.worldKey, result.pixels.data(),
                             result.width, result.height);
        _anyUpdatedThisTick = true;
    });
}

TileRenderController::~TileRenderController()
{
}

void TileRenderController::ensureTickRunning()
{
    if (!_vsyncTimer->isActive()) {
        // Fire immediately on first wake, then repeat at 16ms
        tick();
        _vsyncTimer->start();
    }
}

void TileRenderController::onCameraChanged(
    const TiledViewerCamera& camera,
    const std::shared_ptr<Surface>& surface,
    const std::shared_ptr<Volume>& volume,
    const std::function<TileRenderParams(const WorldTileKey&)>& buildParams,
    const QRectF& viewportRect)
{
    ensureTickRunning();
    _viewportRenderer.onCameraChanged(
        camera, surface, volume, buildParams,
        static_cast<float>(viewportRect.left()),
        static_cast<float>(viewportRect.top()),
        static_cast<float>(viewportRect.right()),
        static_cast<float>(viewportRect.bottom()));
}

void TileRenderController::onCameraChangedDirect(
    const TiledViewerCamera& camera,
    const std::shared_ptr<Surface>& surface,
    const std::shared_ptr<Volume>& volume,
    const std::function<TileRenderParams(const WorldTileKey&)>& buildParams,
    const std::vector<WorldTileKey>& visibleKeys)
{
    ensureTickRunning();
    _viewportRenderer.onCameraChangedDirect(camera, surface, volume, buildParams, visibleKeys);
}

void TileRenderController::onParamsChanged(
    const TiledViewerCamera& camera,
    const std::shared_ptr<Surface>& surface,
    const std::shared_ptr<Volume>& volume,
    const std::function<TileRenderParams(const WorldTileKey&)>& buildParams,
    const QRectF& viewportRect)
{
    ensureTickRunning();
    _viewportRenderer.onParamsChanged(
        camera, surface, volume, buildParams,
        static_cast<float>(viewportRect.left()),
        static_cast<float>(viewportRect.top()),
        static_cast<float>(viewportRect.right()),
        static_cast<float>(viewportRect.bottom()));
}

void TileRenderController::scheduleRender(
    const TiledViewerCamera& camera,
    const std::shared_ptr<Surface>& surface,
    const std::shared_ptr<Volume>& volume,
    const std::function<TileRenderParams(const WorldTileKey&)>& buildParams,
    const QRectF& viewportRect)
{
    _viewportRenderer.scheduleRender(
        camera, surface, volume, buildParams,
        static_cast<float>(viewportRect.left()),
        static_cast<float>(viewportRect.top()),
        static_cast<float>(viewportRect.right()),
        static_cast<float>(viewportRect.bottom()));
    ensureTickRunning();
}

void TileRenderController::cancelAll()
{
    _viewportRenderer.cancelAll();
}

void TileRenderController::clearState()
{
    _viewportRenderer.clearState();
}

void TileRenderController::markOverlaysDirty()
{
    _viewportRenderer.markOverlaysDirty();
    ensureTickRunning();
}

void TileRenderController::markChunkArrived()
{
    _viewportRenderer.markChunkArrived();
    ensureTickRunning();
}

void TileRenderController::setOverlayCallback(std::function<void()> cb)
{
    _viewportRenderer.setOverlayCallback(std::move(cb));
}

void TileRenderController::syncGridBounds(const ContentBounds& bounds, int viewportW, int viewportH)
{
    _viewportRenderer.tileGrid().rebuildGrid(bounds, viewportW, viewportH);
}

void TileRenderController::tick()
{
    _anyUpdatedThisTick = false;
    bool moreWork = _viewportRenderer.tick();

    if (_anyUpdatedThisTick) {
        _tileScene->flush();
        emit sceneNeedsUpdate();
    }

    if (!moreWork)
        _vsyncTimer->stop();
}
