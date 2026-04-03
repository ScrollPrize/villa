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
        _prevFramebuffer = QImage();  // invalidate — sizes don't match
        _blendAlpha = 1.0f;
    }

    if (!_displayItem) {
        _displayItem = _scene->addPixmap(QPixmap::fromImage(_framebuffer));
        _displayItem->setZValue(0);
    }
    _displayItem->setPos(0, 0);
    _dirty = true;
}

void TileScene::setCamera(float surfX, float surfY, float scale)
{
    bool changed = (std::abs(surfX - _camSurfX) > 0.01f ||
                    std::abs(surfY - _camSurfY) > 0.01f ||
                    std::abs(scale - _camScale) > 1e-6f);

    if (changed && !_framebuffer.isNull()) {
        // Snapshot current framebuffer for crossfade
        _prevFramebuffer = _framebuffer.copy();
        _blendAlpha = 0.0f;
    }

    _camSurfX = surfX;
    _camSurfY = surfY;
    _camScale = scale;
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

bool TileScene::flush()
{
    if (!_displayItem) return false;

    bool blending = _blendAlpha < 1.0f;

    if (blending && !_prevFramebuffer.isNull() &&
        _prevFramebuffer.size() == _framebuffer.size())
    {
        _blendAlpha = std::min(_blendAlpha + BLEND_STEP, 1.0f);

        // Fixed-point alpha: 0..256
        int alpha = static_cast<int>(_blendAlpha * 256.0f);
        int invAlpha = 256 - alpha;

        if (_blendBuffer.size() != _framebuffer.size())
            _blendBuffer = QImage(_framebuffer.size(), QImage::Format_RGB32);

        int h = _framebuffer.height();
        int w = _framebuffer.width();
        for (int y = 0; y < h; y++) {
            const auto* prev = reinterpret_cast<const uint32_t*>(_prevFramebuffer.constScanLine(y));
            const auto* curr = reinterpret_cast<const uint32_t*>(_framebuffer.constScanLine(y));
            auto* out = reinterpret_cast<uint32_t*>(_blendBuffer.scanLine(y));
            for (int x = 0; x < w; x++) {
                uint32_t p = prev[x];
                uint32_t c = curr[x];
                uint32_t rb_p = p & 0x00FF00FFu;
                uint32_t g_p  = p & 0x0000FF00u;
                uint32_t rb_c = c & 0x00FF00FFu;
                uint32_t g_c  = c & 0x0000FF00u;
                uint32_t rb = ((rb_p * invAlpha + rb_c * alpha) >> 8) & 0x00FF00FFu;
                uint32_t g  = ((g_p * invAlpha + g_c * alpha) >> 8) & 0x0000FF00u;
                out[x] = 0xFF000000u | rb | g;
            }
        }
        _displayItem->setPixmap(QPixmap::fromImage(_blendBuffer, Qt::NoFormatConversion));
    }
    else if (_dirty)
    {
        _displayItem->setPixmap(QPixmap::fromImage(_framebuffer, Qt::NoFormatConversion));
    }
    else
    {
        return false;
    }

    _dirty = false;
    return _blendAlpha < 1.0f;  // true = still blending, keep ticking
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
