#pragma once

#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QImage>
#include <QPixmap>
#include <QRectF>
#include <vector>
#include <cstdint>
#include <opencv2/core.hpp>

#include "vc/core/render/TileTypes.hpp"
#include "vc/core/render/TileGrid.hpp"

// Single-framebuffer display: one QGraphicsPixmapItem sized to the viewport.
// The tiled rendering pipeline (TileGrid, ViewportRenderer) runs internally;
// this class composites rendered tiles into a single QImage for display.
class TileScene
{
public:
    static constexpr int TILE_PX = vc::render::TileGrid::TILE_PX;

    explicit TileScene(QGraphicsScene* scene);

    // Update grid bounds and resize the framebuffer.
    void rebuildGrid(const ContentBounds& bounds, int viewportW, int viewportH);

    // Blit a rendered tile's pixels into the framebuffer at the correct position.
    // Called from the result callback for each completed tile.
    void blitTile(const WorldTileKey& wk, const uint32_t* pixels, int w, int h);

    // Push the framebuffer to the display item (call once per tick).
    void flush();

    // Reset all tile metadata (on full invalidation)
    void resetMetadata();

    // Clear framebuffer to gray
    void clearAll();

    // Call after the QGraphicsScene is cleared externally.
    void sceneCleared();

    // Legacy compatibility
    bool setTilePixmapOnly(const WorldTileKey& wk, const QPixmap& pixmap);
    bool tileNeedsContent(const WorldTileKey& wk) const;
    void clearRetained() {}
    bool hasRetainedItems() const { return false; }
    int unfilledTileCount() const { return _grid.unfilledTileCount(); }

    // Content bounds
    const ContentBounds& bounds() const { return _grid.bounds(); }
    int cols() const { return _grid.cols(); }
    int rows() const { return _grid.rows(); }

    // Convert surface parameter coordinates to scene pixel coordinates
    QPointF surfaceToScene(float surfX, float surfY) const;

    // Convert scene pixel coordinates to surface parameter coordinates
    cv::Vec2f sceneToSurface(const QPointF& scenePos) const;

    // Get world tile keys visible in the given viewport scene rect (+ buffer tiles).
    std::vector<WorldTileKey> visibleTiles(const QRectF& viewportSceneRect,
                                            int buffer = tiled_config::VISIBLE_BUFFER_TILES) const;

    int worstVisibleLevel(const QRectF& viewportSceneRect) const;

    std::vector<WorldTileKey> staleTilesInRect(int desiredLevel, uint64_t epoch,
                                                const QRectF& viewportSceneRect,
                                                int buffer = tiled_config::VISIBLE_BUFFER_TILES) const;

    template<typename Func>
    void forEachTile(Func&& fn) const {
        _grid.forEachTile(std::forward<Func>(fn));
    }

    vc::render::TileGrid& grid() { return _grid; }
    const vc::render::TileGrid& grid() const { return _grid; }

private:
    QGraphicsScene* _scene;
    vc::render::TileGrid _grid;
    QGraphicsPixmapItem* _displayItem = nullptr;  // single display item
    QImage _framebuffer;                           // viewport-sized ARGB32 buffer
    bool _dirty = false;                           // framebuffer changed since last flush
};
