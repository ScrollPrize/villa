#pragma once

#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QImage>
#include <QPixmap>
#include <QPointF>
#include <cstdint>
#include <opencv2/core.hpp>

#include "vc/core/render/TileTypes.hpp"

// Single-framebuffer display: one QGraphicsPixmapItem sized to the viewport.
// The tiled rendering pipeline (TileGrid, ViewportRenderer) lives in the core
// layer; this class just composites rendered tiles into a single QImage.
class TileScene
{
public:
    explicit TileScene(QGraphicsScene* scene);

    // Resize the framebuffer (call when viewport or bounds change).
    void rebuildGrid(const ContentBounds& bounds, int viewportW, int viewportH);

    // Blit a rendered tile into the framebuffer at the correct position.
    void blitTile(const WorldTileKey& wk, const uint32_t* pixels, int w, int h);

    // Push the framebuffer to the display item (call once per tick).
    void flush();

    // Clear framebuffer to gray.
    void clearAll();

    // Call after the QGraphicsScene is cleared externally.
    void sceneCleared();

    // Coordinate conversions (viewport-relative)
    QPointF surfaceToScene(float surfX, float surfY) const;
    cv::Vec2f sceneToSurface(const QPointF& scenePos) const;

    [[nodiscard]] float camScale() const noexcept { return _camScale; }
    [[nodiscard]] float camZOff() const noexcept { return _camZOff; }
    void setCamZOff(float z) noexcept { _camZOff = z; }

    // Set camera position for viewport-relative blitting.
    // Shifts framebuffer to compensate for pan; clears on zoom change.
    void setCamera(float surfX, float surfY, float scale);

private:
    QGraphicsScene* _scene;
    QGraphicsPixmapItem* _displayItem = nullptr;
    QImage _framebuffer;
    bool _dirty = false;

    // Camera state for viewport-relative positioning
    float _camSurfX = 0, _camSurfY = 0, _camScale = 1.0f, _camZOff = 0;
    float _worldTileSize = 0;
};
