#include "TileRenderController.hpp"

#include <QDebug>
#include <QThread>
#include <QTimer>
#include <QImage>
#include <algorithm>
#include <cmath>
#include <vector>

#include "vc/core/types/Volume.hpp"
#include "vc/core/util/Surface.hpp"

TileRenderController::TileRenderController(TileScene* tileScene, RenderPool* sharedPool, QObject* parent)
    : QObject(parent)
    , _tileScene(tileScene)
    , _renderPool(sharedPool)
    , _viewportRenderer(&sharedPool->corePool())
{
    // When a tile completes, wake the tick so it drains on the next cycle.
    // RenderPool emits tileReady() from the core pool's ready callback.
    // Don't use _viewportRenderer.setTileReadyCallback() here — that would
    // overwrite the shared RenderPool's callback, breaking other controllers.
    connect(_renderPool, &RenderPool::tileReady, this, &TileRenderController::ensureTickRunning,
            Qt::QueuedConnection);
}

TileRenderController::~TileRenderController()
{
}

void TileRenderController::ensureTickRunning()
{
    if (!_tickPending) {
        _tickPending = true;
        QTimer::singleShot(0, this, &TileRenderController::tick);
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
    _syncedVersions.clear();
}

bool TileRenderController::syncTilesToScene()
{
    auto& grid = _viewportRenderer.tileGrid();
    const auto& bounds = grid.bounds();
    if (bounds.totalCols <= 0 || bounds.totalRows <= 0)
        return false;

    bool anyUpdated = false;

    grid.forEachTile([&](const TileKey& key) {
        int idx = key.row * bounds.totalCols + key.col;
        uint64_t ver = grid.tileVersion(bounds.worldKeyAt(key.col, key.row));
        if (ver == 0)
            return;  // tile not yet rendered

        auto it = _syncedVersions.find(idx);
        if (it != _syncedVersions.end() && it->second == ver)
            return;  // already synced this version

        WorldTileKey wk = bounds.worldKeyAt(key.col, key.row);
        const uint32_t* pixels = grid.tilePixels(wk);
        if (!pixels)
            return;

        auto [w, h] = grid.tileSize(wk);
        if (w <= 0 || h <= 0)
            return;

        // Convert raw ARGB32 pixels -> QPixmap
        QImage img(w, h, QImage::Format_RGB32);
        const int srcStride = w * 4;
        const int dstStride = img.bytesPerLine();
        if (srcStride == dstStride) {
            std::memcpy(img.bits(), pixels, static_cast<size_t>(srcStride) * h);
        } else {
            for (int y = 0; y < h; y++) {
                std::memcpy(img.scanLine(y),
                            reinterpret_cast<const uchar*>(pixels) + y * srcStride,
                            srcStride);
            }
        }
        QPixmap pixmap = QPixmap::fromImage(std::move(img), Qt::NoFormatConversion);

        // Get metadata from grid for the staleness check
        const auto& meta = grid.metaAt(idx);

        if (_tileScene->setTileWorld(wk, pixmap, meta.epoch, meta.level)) {
            anyUpdated = true;
        }

        _syncedVersions[idx] = ver;
    });

    return anyUpdated;
}

void TileRenderController::tick()
{
    _tickPending = false;

    // Delegate all scheduling, draining, and progressive refinement to ViewportRenderer.
    bool moreWork = _viewportRenderer.tick();

    // Sync updated tiles from ViewportRenderer's TileGrid to TileScene's QGraphicsPixmapItems.
    bool anyUpdated = syncTilesToScene();

    if (anyUpdated)
        emit sceneNeedsUpdate();

    // Schedule next tick if there's more work
    if (moreWork)
        ensureTickRunning();
}
