#include "TileScene.hpp"

#include <algorithm>
#include <cstring>

TileScene::TileScene(QGraphicsScene* scene)
    : _scene(scene)
{
}

void TileScene::rebuildGrid(const ContentBounds& bounds, int viewportW, int viewportH)
{
    _worldTileSize = bounds.worldTileSize;

    _scene->setSceneRect(0, 0, viewportW, viewportH);

    if (_framebuffer.isNull() || _framebuffer.width() != viewportW || _framebuffer.height() != viewportH) {
        _framebuffer = QImage(std::max(1, viewportW), std::max(1, viewportH), QImage::Format_RGB32);
        _framebuffer.fill(QColor(64, 64, 64));
    }

    if (!_displayItem) {
        _displayItem = _scene->addPixmap(QPixmap::fromImage(_framebuffer));
        _displayItem->setZValue(0);
    }
    _displayItem->setPos(0, 0);
    _dirty = true;
}

void TileScene::blitTile(const WorldTileKey& wk, const uint32_t* pixels, int w, int h)
{
    if (_framebuffer.isNull() || _worldTileSize <= 0 || _camScale <= 0) return;

    float tileSurfX = static_cast<float>(wk.worldCol) * _worldTileSize;
    float tileSurfY = static_cast<float>(wk.worldRow) * _worldTileSize;
    float vpCx = static_cast<float>(_framebuffer.width()) * 0.5f;
    float vpCy = static_cast<float>(_framebuffer.height()) * 0.5f;
    int dstX = static_cast<int>((tileSurfX - _camSurfX) * _camScale + vpCx);
    int dstY = static_cast<int>((tileSurfY - _camSurfY) * _camScale + vpCy);

    static float lastLogScale = -1;
    static int blitLog = 0;
    static int blitVisible = 0;
    static int blitClipped = 0;
    if (std::abs(_camScale - lastLogScale) > 0.001f) {
        if (lastLogScale > 0)
            fprintf(stderr, "[blit-summary] scale=%.3f visible=%d clipped=%d\n", lastLogScale, blitVisible, blitClipped);
        lastLogScale = _camScale;
        blitLog = 0;
        blitVisible = 0;
        blitClipped = 0;
    }

    int fbW = _framebuffer.width();
    int fbH = _framebuffer.height();
    if (dstX >= fbW || dstY >= fbH || dstX + w <= 0 || dstY + h <= 0) {
        blitClipped++;
        return;
    }

    int srcStartX = std::max(0, -dstX);
    int srcStartY = std::max(0, -dstY);
    int copyW = std::min(w - srcStartX, fbW - std::max(0, dstX));
    int copyH = std::min(h - srcStartY, fbH - std::max(0, dstY));
    if (copyW <= 0 || copyH <= 0) { blitClipped++; return; }

    blitVisible++;
    if (blitLog++ < 5) {
        fprintf(stderr, "[blit] scale=%.3f wts=%.1f dst=(%d,%d) copy=%dx%d fb=%dx%d wk=(%d,%d)\n",
            _camScale, _worldTileSize, dstX, dstY, copyW, copyH, fbW, fbH, wk.worldCol, wk.worldRow);
    }

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

void TileScene::clearAll()
{
    if (!_framebuffer.isNull())
        _framebuffer.fill(QColor(64, 64, 64));
    _dirty = true;
}

void TileScene::sceneCleared()
{
    _displayItem = nullptr;
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
