#pragma once

#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QImage>
#include <QPixmap>
#include <QPointF>
#include <cstdint>
#include <opencv2/core.hpp>

// Single-framebuffer display: one QGraphicsPixmapItem sized to the viewport.
// Rendering writes directly to the framebuffer; this class just manages the
// QImage and provides coordinate conversions.
class TileScene
{
public:
    explicit TileScene(QGraphicsScene* scene);

    // Resize the framebuffer (call when viewport changes).
    void rebuildGrid(int viewportW, int viewportH);

    // Push the framebuffer to the display item (call once per render).
    void flush();

    // Clear framebuffer to gray.
    void clearAll();

    // Call after the QGraphicsScene is cleared externally.
    void sceneCleared();

    // Coordinate conversions (viewport-relative)
    QPointF surfaceToScene(float surfX, float surfY) const;
    cv::Vec2f sceneToSurface(const QPointF& scenePos) const;

    // Direct framebuffer access for full-viewport rendering.
    uint32_t* framebufferBits() { return _framebuffer.isNull() ? nullptr : reinterpret_cast<uint32_t*>(_framebuffer.bits()); }
    int framebufferStride() const { return _framebuffer.isNull() ? 0 : _framebuffer.bytesPerLine() / 4; }
    int framebufferWidth() const { return _framebuffer.width(); }
    int framebufferHeight() const { return _framebuffer.height(); }
    void markDirty() { _dirty = true; }
    QImage& rawFramebuffer() { return _framebuffer; }
    const QImage& constFramebuffer() const { return _framebuffer; }

    // Set camera position for coordinate conversions.
    // Clears framebuffer on zoom change.
    void setCamera(float surfX, float surfY, float scale);

    void setCamZOff(float z) noexcept { _camZOff = z; }

private:
    QGraphicsScene* _scene;
    QGraphicsPixmapItem* _displayItem = nullptr;
    QImage _framebuffer;
    bool _dirty = false;

    // Camera state for viewport-relative positioning
    float _camSurfX = 0, _camSurfY = 0, _camScale = 1.0f, _camZOff = 0;

    // Staging buffer — render here, then memcpy to QImage
    std::vector<uint32_t> _staging;
};
