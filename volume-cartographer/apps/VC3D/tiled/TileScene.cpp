#include "TileScene.hpp"

#include <algorithm>
#include <cstring>
#include <cstdint>

TileScene::TileScene(QGraphicsScene* scene)
    : _scene(scene)
{
}

void TileScene::rebuildGrid(int viewportW, int viewportH)
{
    _scene->setSceneRect(0, 0, viewportW, viewportH);

    if (_framebuffer.isNull() || _framebuffer.width() != viewportW || _framebuffer.height() != viewportH) {
        _framebuffer = QImage(std::max(1, viewportW), std::max(1, viewportH), QImage::Format_RGB32);
        _framebuffer.fill(QColor(64, 64, 64));
    }
    _dirty = true;
}

void TileScene::flush()
{
    if (!_dirty) return;
    // No QPixmap conversion needed — drawBackground paints the QImage directly
    _dirty = false;
}

void TileScene::setCamera(float surfX, float surfY, float scale)
{
    bool changed = std::abs(surfX - _camSurfX) > 0.01f ||
                   std::abs(surfY - _camSurfY) > 0.01f ||
                   std::abs(scale - _camScale) > 1e-6f;
    _camSurfX = surfX; _camSurfY = surfY; _camScale = scale;
    if (changed) clearAll();
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

