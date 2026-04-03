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
    // One memcpy into QImage (which owns its buffer), then QPixmap::fromImage.
    _viewportRenderer.setResultCallback([this](TileRenderResult&& result) {
        if (result.pixels.empty()) return;

        QImage img(result.width, result.height, QImage::Format_RGB32);
        const size_t srcStride = static_cast<size_t>(result.width) * 4;
        const size_t dstStride = static_cast<size_t>(img.bytesPerLine());
        if (srcStride == dstStride) {
            std::memcpy(img.bits(), result.pixels.data(),
                        srcStride * static_cast<size_t>(result.height));
        } else {
            for (int y = 0; y < result.height; y++) {
                std::memcpy(img.scanLine(y),
                            reinterpret_cast<const uchar*>(result.pixels.data()) + static_cast<size_t>(y) * srcStride,
                            srcStride);
            }
        }
        QPixmap pixmap = QPixmap::fromImage(std::move(img), Qt::NoFormatConversion);

        _tileScene->setTilePixmapOnly(result.worldKey, pixmap);
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

    if (_anyUpdatedThisTick)
        emit sceneNeedsUpdate();

    if (!moreWork)
        _vsyncTimer->stop();
}
