#pragma once

#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QPixmap>
#include <QRectF>
#include <vector>
#include <cstdint>
#include <cmath>
#include <opencv2/core.hpp>

// A tile key identifying a tile by its grid position (grid-local coordinates)
struct TileKey {
    int col = 0;
    int row = 0;

    bool operator==(const TileKey& o) const { return col == o.col && row == o.row; }
};

// A tile key in world-aligned coordinates (fixed surface parameter grid)
struct WorldTileKey {
    int worldCol = 0;
    int worldRow = 0;

    bool operator==(const WorldTileKey& o) const {
        return worldCol == o.worldCol && worldRow == o.worldRow;
    }
};

// Hash for WorldTileKey (needed for unordered containers)
struct WorldTileKeyHash {
    size_t operator()(const WorldTileKey& k) const {
        return std::hash<int>()(k.worldCol) ^ (std::hash<int>()(k.worldRow) << 16);
    }
};

// Content bounds describing the full tile grid covering all content
struct ContentBounds {
    int firstWorldCol = 0;  // world column of leftmost tile
    int firstWorldRow = 0;  // world row of topmost tile
    int totalCols = 0;      // number of tile columns
    int totalRows = 0;      // number of tile rows
    float worldTileSize = 0; // surface units per tile = TILE_PX / scale
    float scale = 0;        // current zoom scale

    bool operator==(const ContentBounds& o) const {
        return firstWorldCol == o.firstWorldCol && firstWorldRow == o.firstWorldRow &&
               totalCols == o.totalCols && totalRows == o.totalRows;
    }
    bool operator!=(const ContentBounds& o) const { return !(*this == o); }

    // Map grid position to world tile key
    WorldTileKey worldKeyAt(int gridCol, int gridRow) const {
        return {firstWorldCol + gridCol, firstWorldRow + gridRow};
    }

    // Map world tile key to grid position. Returns false if out of range.
    bool gridPosition(const WorldTileKey& wk, int& outCol, int& outRow) const {
        outCol = wk.worldCol - firstWorldCol;
        outRow = wk.worldRow - firstWorldRow;
        return outCol >= 0 && outCol < totalCols && outRow >= 0 && outRow < totalRows;
    }
};

// Shared configuration constants for the tiled renderer subsystem.
// TILE_PX lives in TileScene (tightly coupled to grid layout).
namespace tiled_config {
    constexpr int VISIBLE_BUFFER_TILES = 2;   // extra tiles around viewport for smooth scrolling
    constexpr int MAX_COARSER_LEVELS   = 8;   // fallback levels searched in SliceCache::getBest()
    constexpr int DRAIN_BATCH_SIZE     = 128; // max results drained per tick cycle
}

// Per-tile metadata for staleness checks during progressive rendering.
struct TileMetadata {
    uint64_t epoch = 0;
    int8_t level = -1;    // pyramid level of current pixmap (-1 = placeholder)
};

// Manages a grid of QGraphicsPixmapItems covering the full content on a QGraphicsScene.
// Tiles are permanently positioned at world coordinates. The QGraphicsView viewport
// scrolls over this scene to handle panning (no grid shifting or popping).
class TileScene
{
public:
    static constexpr int TILE_PX = 512;

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
    // Tracked incrementally — O(1) to query.
    int unfilledTileCount() const { return _unfilledCount; }

    // Content bounds
    const ContentBounds& bounds() const { return _bounds; }
    int cols() const { return _bounds.totalCols; }
    int rows() const { return _bounds.totalRows; }

    // Convert surface parameter coordinates to scene pixel coordinates
    QPointF surfaceToScene(float surfX, float surfY) const;

    // Convert scene pixel coordinates to surface parameter coordinates
    cv::Vec2f sceneToSurface(const QPointF& scenePos) const;

    // Get world tile keys visible in the given viewport scene rect (+ buffer tiles).
    std::vector<WorldTileKey> visibleTiles(const QRectF& viewportSceneRect,
                                            int buffer = tiled_config::VISIBLE_BUFFER_TILES) const;

    // Returns world keys of tiles whose rendered level is worse than desiredLevel,
    // limited to tiles visible in the given viewport rect.
    std::vector<WorldTileKey> staleTilesInRect(int desiredLevel, uint64_t epoch,
                                                const QRectF& viewportSceneRect,
                                                int buffer = tiled_config::VISIBLE_BUFFER_TILES) const;

    // Iterate all tile keys (grid-local)
    template<typename Func>
    void forEachTile(Func&& fn) const {
        for (int r = 0; r < _bounds.totalRows; ++r) {
            for (int c = 0; c < _bounds.totalCols; ++c) {
                fn(TileKey{c, r});
            }
        }
    }

private:
    QGraphicsPixmapItem* itemAt(int col, int row) const;

    // Convert viewport scene rect to grid-local col/row range (clamped)
    void visibleGridRange(const QRectF& viewportSceneRect, int buffer,
                          int& firstCol, int& firstRow,
                          int& lastCol, int& lastRow) const;

    QGraphicsScene* _scene;
    std::vector<QGraphicsPixmapItem*> _items; // row-major: [row * totalCols + col]
    std::vector<TileMetadata> _meta;
    std::vector<QGraphicsPixmapItem*> _retainedItems; // old items kept as background during transition
    int _unfilledCount = 0;  // tiles with level == -1 (tracked incrementally)
    ContentBounds _bounds;
    float _padX = 0;  // scene padding when content < viewport
    float _padY = 0;
};
