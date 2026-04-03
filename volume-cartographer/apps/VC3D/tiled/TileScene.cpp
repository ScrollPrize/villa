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

    const auto& b = _grid.bounds();
    float padX = _grid.padX();
    float padY = _grid.padY();

    float contentPxW = static_cast<float>(b.totalCols) * TILE_PX;
    float contentPxH = static_cast<float>(b.totalRows) * TILE_PX;
    float sceneW = std::max(contentPxW, static_cast<float>(viewportW));
    float sceneH = std::max(contentPxH, static_cast<float>(viewportH));
    _scene->setSceneRect(0, 0, static_cast<qreal>(sceneW), static_cast<qreal>(sceneH));

    // Framebuffer covers the full content grid.
    // Only reallocate if it needs to GROW — never shrink (avoids clearing on
    // every zoom step when windowing changes totalCols).
    int fbW = std::max(1, static_cast<int>(contentPxW));
    int fbH = std::max(1, static_cast<int>(contentPxH));

    if (_framebuffer.isNull() || _framebuffer.width() < fbW || _framebuffer.height() < fbH) {
        // Grow to at least the needed size, with some headroom to avoid
        // repeated reallocations during zoom
        int newW = std::max(fbW, _framebuffer.width());
        int newH = std::max(fbH, _framebuffer.height());
        QImage newFb(newW, newH, QImage::Format_RGB32);
        newFb.fill(QColor(64, 64, 64));
        // Copy old content if we had any
        if (!_framebuffer.isNull()) {
            int copyW = std::min(_framebuffer.width(), newW);
            int copyH = std::min(_framebuffer.height(), newH);
            for (int y = 0; y < copyH; y++) {
                std::memcpy(newFb.scanLine(y), _framebuffer.constScanLine(y),
                            static_cast<size_t>(copyW) * 4);
            }
        }
        _framebuffer = std::move(newFb);
    }

    // Create or reposition the single display item
    if (!_displayItem) {
        _displayItem = _scene->addPixmap(QPixmap::fromImage(_framebuffer));
        _displayItem->setZValue(0);
    }
    _displayItem->setPos(static_cast<qreal>(padX), static_cast<qreal>(padY));
    _dirty = true;
}

void TileScene::blitTile(const WorldTileKey& wk, const uint32_t* pixels, int w, int h)
{
    const auto& b = _grid.bounds();
    int col, row;
    if (!b.gridPosition(wk, col, row)) return;

    // Tile position in the framebuffer
    int dstX = col * TILE_PX;
    int dstY = row * TILE_PX;

    // Clip to framebuffer
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

    // Blit row by row
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
    // Legacy path: convert QPixmap to pixels and blit
    QImage img = pixmap.toImage().convertToFormat(QImage::Format_RGB32);
    if (img.isNull()) return false;
    // The pixel data is ARGB32 = RGB32 in Qt
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
    auto v = _grid.surfaceToGrid(surfX, surfY);
    return {static_cast<qreal>(v[0]), static_cast<qreal>(v[1])};
}

cv::Vec2f TileScene::sceneToSurface(const QPointF& scenePos) const
{
    return _grid.gridToSurface(static_cast<float>(scenePos.x()),
                               static_cast<float>(scenePos.y()));
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
