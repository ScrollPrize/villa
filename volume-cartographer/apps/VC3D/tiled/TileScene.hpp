#pragma once

#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QPixmap>
#include <QRectF>
#include <vector>
#include <cstdint>
#include <opencv2/core.hpp>

#include "vc/core/render/TileTypes.hpp"
#include "vc/core/render/TileGrid.hpp"

// Manages a grid of QGraphicsPixmapItems covering the full content on a QGraphicsScene.
// Data/logic operations are delegated to the platform-agnostic TileGrid in core;
// this class keeps only QGraphicsPixmapItem management for display.
class TileScene
{
public:
    static constexpr int TILE_PX = vc::render::TileGrid::TILE_PX;

    explicit TileScene(QGraphicsScene* scene);

    // Rebuild the grid to cover the given content bounds.
    // Retains old items as background layer and creates new ones.
    // viewportW/H are used to pad the scene rect so centerOn() works when
    // content is smaller than the viewport.
    void rebuildGrid(const ContentBounds& bounds, int viewportW, int viewportH);

    // Set tile with staleness check (uses grid-local coordinates).
    // Returns true if pixmap was applied.
    bool setTile(const TileKey& key, const QPixmap& pixmap,
                 uint64_t epoch, int8_t level);

    // Set tile by world key (converts to grid position internally).
    bool setTileWorld(const WorldTileKey& wk, const QPixmap& pixmap,
                      uint64_t epoch, int8_t level);

    // Set only the QGraphicsPixmapItem pixmap without round-tripping pixels
    // back into TileGrid.  Use when the grid already has the pixel data
    // (e.g. syncTilesToScene reads from grid, converts to QPixmap, and just
    // needs to update the display item).
    bool setTilePixmapOnly(const WorldTileKey& wk, const QPixmap& pixmap);

    // Returns true if the tile at wk has no rendered content (level == -1).
    // Used to decide whether a synchronous coarse preview is needed.
    bool tileNeedsContent(const WorldTileKey& wk) const;

    // Reset all tile metadata (on full invalidation)
    void resetMetadata();

    // Set all tiles to a gray placeholder
    void clearAll();

    // Call after the QGraphicsScene is cleared externally.
    void sceneCleared();

    // Remove and delete retained (background) items from previous grid rebuilds.
    void clearRetained();
    bool hasRetainedItems() const { return !_retainedItems.empty(); }

    // Number of tiles that have no rendered content (level == -1).
    // Tracked incrementally -- O(1) to query.
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

    // Returns the coarsest (worst) actual pyramid level among visible tiles,
    // or -1 if no tiles have been rendered yet.
    int worstVisibleLevel(const QRectF& viewportSceneRect) const;

    // Returns world keys of tiles whose rendered level is worse than desiredLevel,
    // limited to tiles visible in the given viewport rect.
    std::vector<WorldTileKey> staleTilesInRect(int desiredLevel, uint64_t epoch,
                                                const QRectF& viewportSceneRect,
                                                int buffer = tiled_config::VISIBLE_BUFFER_TILES) const;

    // Iterate all tile keys (grid-local)
    template<typename Func>
    void forEachTile(Func&& fn) const {
        _grid.forEachTile(std::forward<Func>(fn));
    }

    // Access the underlying platform-agnostic grid
    vc::render::TileGrid& grid() { return _grid; }
    const vc::render::TileGrid& grid() const { return _grid; }

private:
    QGraphicsPixmapItem* itemAt(int col, int row) const;

    QGraphicsScene* _scene;
    vc::render::TileGrid _grid;
    std::vector<QGraphicsPixmapItem*> _items; // row-major: [row * totalCols + col]
    std::vector<QGraphicsPixmapItem*> _retainedItems; // old items kept as background during transition
};
