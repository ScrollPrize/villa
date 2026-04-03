#include "TileScene.hpp"

#include <algorithm>
#include <cstring>

TileScene::TileScene(QGraphicsScene* scene)
    : _scene(scene)
{
}

void TileScene::rebuildGrid(const ContentBounds& bounds, int viewportW, int viewportH)
{
    _grid.rebuildGrid(bounds, viewportW, viewportH);

    // Scene rect = viewport size. No scrolling — we blit directly.
    _scene->setSceneRect(0, 0, viewportW, viewportH);

    // Framebuffer = viewport size (always)
    if (_framebuffer.isNull() || _framebuffer.width() != viewportW || _framebuffer.height() != viewportH) {
        _framebuffer = QImage(std::max(1, viewportW), std::max(1, viewportH), QImage::Format_RGB32);
        _framebuffer.fill(QColor(64, 64, 64));
    }

    if (!_displayItem) {
        _displayItem = _scene->addPixmap(QPixmap::fromImage(_framebuffer));
        _displayItem->setZValue(0);
    }
    // Display item always at (0,0) — it IS the viewport
    _displayItem->setPos(0, 0);
    _dirty = true;
}

void TileScene::blitTile(const WorldTileKey& wk, const uint32_t* pixels, int w, int h)
{
    if (_framebuffer.isNull()) return;
    const auto& b = _grid.bounds();
    if (b.scale <= 0 || b.worldTileSize <= 0) return;

    // Viewport-relative position: (tileSurf - camera) * scale + viewport/2
    float tileSurfX = static_cast<float>(wk.worldCol) * b.worldTileSize;
    float tileSurfY = static_cast<float>(wk.worldRow) * b.worldTileSize;
    float vpCx = static_cast<float>(_framebuffer.width()) * 0.5f;
    float vpCy = static_cast<float>(_framebuffer.height()) * 0.5f;
    int dstX = static_cast<int>((tileSurfX - _camSurfX) * _camScale + vpCx);
    int dstY = static_cast<int>((tileSurfY - _camSurfY) * _camScale + vpCy);

    int fbW = _framebuffer.width();
    int fbH = _framebuffer.height();
    if (dstX >= fbW || dstY >= fbH || dstX + w <= 0 || dstY + h <= 0) return;

    int srcStartX = std::max(0, -dstX);
    int srcStartY = std::max(0, -dstY);
    int copyW = std::min(w - srcStartX, fbW - std::max(0, dstX));
    int copyH = std::min(h - srcStartY, fbH - std::max(0, dstY));
    if (copyW <= 0 || copyH <= 0) return;

    int dstStartX = std::max(0, dstX);
    int dstStartY = std::max(0, dstY);

    for (int y = 0; y < copyH; y++) {
        const uint32_t* srcRow = pixels + (srcStartY + y) * w + srcStartX;
        uchar* dstRow = _framebuffer.scanLine(dstStartY + y) + dstStartX * 4;
        std::memcpy(dstRow, srcRow, static_cast<size_t>(copyW) * 4);
    }
    _dirty = true;
}

void TileScene::flush()
{
    if (!_dirty || !_displayItem) return;
    _displayItem->setPixmap(QPixmap::fromImage(_framebuffer, Qt::NoFormatConversion));
    _dirty = false;
}

bool TileScene::setTilePixmapOnly(const WorldTileKey& wk, const QPixmap& pixmap)
{
    QImage img = pixmap.toImage().convertToFormat(QImage::Format_RGB32);
    if (img.isNull()) return false;
    const auto* pixels = reinterpret_cast<const uint32_t*>(img.constBits());
    blitTile(wk, pixels, img.width(), img.height());
    return true;
}

bool TileScene::tileNeedsContent(const WorldTileKey& wk) const
{
    return _grid.tileNeedsContent(wk);
}

void TileScene::resetMetadata()
{
    _grid.resetMetadata();
}

void TileScene::clearAll()
{
    if (!_framebuffer.isNull())
        _framebuffer.fill(QColor(64, 64, 64));
    _dirty = true;
    resetMetadata();
}

void TileScene::sceneCleared()
{
    _displayItem = nullptr;
    _grid.clear();
}

QPointF TileScene::surfaceToScene(float surfX, float surfY) const
{
    if (_framebuffer.isNull()) return {0, 0};
    float vpCx = static_cast<float>(_framebuffer.width()) * 0.5f;
    float vpCy = static_cast<float>(_framebuffer.height()) * 0.5f;
    return {static_cast<qreal>((surfX - _camSurfX) * _camScale + vpCx),
            static_cast<qreal>((surfY - _camSurfY) * _camScale + vpCy)};
}

cv::Vec2f TileScene::sceneToSurface(const QPointF& scenePos) const
{
    if (_framebuffer.isNull() || _camScale <= 0) return {0, 0};
    float vpCx = static_cast<float>(_framebuffer.width()) * 0.5f;
    float vpCy = static_cast<float>(_framebuffer.height()) * 0.5f;
    float surfX = (static_cast<float>(scenePos.x()) - vpCx) / _camScale + _camSurfX;
    float surfY = (static_cast<float>(scenePos.y()) - vpCy) / _camScale + _camSurfY;
    return {surfX, surfY};
}

std::vector<WorldTileKey> TileScene::visibleTiles(const QRectF& viewportSceneRect,
                                                    int buffer) const
{
    return _grid.visibleTiles(
        static_cast<float>(viewportSceneRect.left()),
        static_cast<float>(viewportSceneRect.top()),
        static_cast<float>(viewportSceneRect.right()),
        static_cast<float>(viewportSceneRect.bottom()),
        buffer);
}

int TileScene::worstVisibleLevel(const QRectF& viewportSceneRect) const
{
    return _grid.worstVisibleLevel(
        static_cast<float>(viewportSceneRect.left()),
        static_cast<float>(viewportSceneRect.top()),
        static_cast<float>(viewportSceneRect.right()),
        static_cast<float>(viewportSceneRect.bottom()));
}

std::vector<WorldTileKey> TileScene::staleTilesInRect(int desiredLevel, uint64_t epoch,
                                                        const QRectF& viewportSceneRect,
                                                        int buffer) const
{
    return _grid.staleTilesInRect(
        desiredLevel, epoch,
        static_cast<float>(viewportSceneRect.left()),
        static_cast<float>(viewportSceneRect.top()),
        static_cast<float>(viewportSceneRect.right()),
        static_cast<float>(viewportSceneRect.bottom()),
        buffer);
}
