#pragma once

#include <cstdint>
#include <cmath>
#include <utility>
#include <vector>
#include <opencv2/core.hpp>

#include "vc/core/render/TileTypes.hpp"

namespace vc::render {

// Platform-agnostic tile grid: manages tile metadata, pixel storage, coordinate
// transforms, and visibility queries.  Owns NO Qt types.  The Qt-side TileScene
// delegates data/logic operations here and keeps only QGraphicsPixmapItem
// management for display.
class TileGrid {
public:
    static constexpr int TILE_PX = 512;

    // Rebuild the grid to cover the given content bounds.
    // viewportW/H are used to compute padding so centering works when content
    // is smaller than the viewport.  Returns true if the grid was actually
    // rebuilt (bounds changed); false if only padding was updated or nothing
    // changed.
    bool rebuildGrid(const ContentBounds& bounds, int viewportW, int viewportH);

    // Set tile with staleness check (grid-local coordinates).
    // Stores pixels and updates metadata.  Returns true if accepted.
    bool setTile(const TileKey& key, std::vector<uint32_t>&& pixels,
                 int width, int height, int8_t level,
                 std::chrono::steady_clock::time_point submitTime = {},
                 std::chrono::steady_clock::time_point renderDone = {});

    // Set tile by world key (converts to grid position internally).
    bool setTileWorld(const WorldTileKey& wk, std::vector<uint32_t>&& pixels,
                      int width, int height, int8_t level,
                      std::chrono::steady_clock::time_point submitTime = {},
                      std::chrono::steady_clock::time_point renderDone = {});

    // Update metadata only (no pixel storage). Returns true if accepted.
    bool setTileMeta(const WorldTileKey& wk, int8_t level);

    // Returns true if the tile at wk has no rendered content (level == -1).
    bool tileNeedsContent(const WorldTileKey& wk) const;

    // Reset all tile metadata (on full invalidation).
    void resetMetadata();

    // Clear all tile data and metadata (full reset).
    void clear();

    // Number of tiles that have no rendered content (level == -1).
    [[nodiscard]] int unfilledTileCount() const noexcept { return _unfilledCount; }

    // Content bounds
    [[nodiscard]] const ContentBounds& bounds() const noexcept { return _bounds; }
    [[nodiscard]] int cols() const noexcept { return _bounds.totalCols; }
    [[nodiscard]] int rows() const noexcept { return _bounds.totalRows; }

    // Padding
    [[nodiscard]] float padX() const noexcept { return _padX; }
    [[nodiscard]] float padY() const noexcept { return _padY; }

    // Convert surface parameter coordinates to grid pixel coordinates.
    cv::Vec2f surfaceToGrid(float surfX, float surfY) const;

    // Convert grid pixel coordinates to surface parameter coordinates.
    cv::Vec2f gridToSurface(float gridX, float gridY) const;

    // World tile keys visible in the given viewport rect (+ buffer tiles).
    std::vector<WorldTileKey> visibleTiles(float vpL, float vpT, float vpR, float vpB,
                                           int buffer = tiled_config::VISIBLE_BUFFER_TILES) const;

    // Coarsest (worst) actual pyramid level among visible tiles, or -1.
    int worstVisibleLevel(float vpL, float vpT, float vpR, float vpB) const;

    // World keys of tiles whose rendered level is worse than desiredLevel,
    // limited to tiles visible in the given viewport rect.
    std::vector<WorldTileKey> staleTilesInRect(int desiredLevel,
                                               float vpL, float vpT, float vpR, float vpB,
                                               int buffer = tiled_config::VISIBLE_BUFFER_TILES) const;

    // Access rendered tile pixels (nullptr if not yet rendered or out of range).
    const uint32_t* tilePixels(const WorldTileKey& wk) const;
    std::pair<int,int> tileSize(const WorldTileKey& wk) const;

    // Per-tile version counter, incremented on each successful setTile.
    uint64_t tileVersion(const WorldTileKey& wk) const;

    // Metadata access (for staleness checks by TileScene)
    [[nodiscard]] const TileMetadata& metaAt(int idx) const noexcept { return _meta[idx]; }

    // Timing access (for profiling)
    std::chrono::steady_clock::time_point tileSubmitTime(const WorldTileKey& wk) const;
    std::chrono::steady_clock::time_point tileRenderDone(const WorldTileKey& wk) const;

    // Iterate all tile keys (grid-local)
    template<typename Func>
    void forEachTile(Func&& fn) const {
        for (int r = 0; r < _bounds.totalRows; ++r)
            for (int c = 0; c < _bounds.totalCols; ++c)
                fn(TileKey{c, r});
    }

private:
    void visibleGridRange(float vpL, float vpT, float vpR, float vpB, int buffer,
                          int& firstCol, int& firstRow,
                          int& lastCol, int& lastRow) const;

    ContentBounds _bounds;
    float _padX = 0;
    float _padY = 0;
    int _unfilledCount = 0;

    std::vector<TileMetadata> _meta;

    struct TileData {
        std::vector<uint32_t> pixels;
        int width = 0;
        int height = 0;
        uint64_t version = 0;
        std::chrono::steady_clock::time_point submitTime;
        std::chrono::steady_clock::time_point renderDone;
    };
    std::vector<TileData> _tiles;
    uint64_t _nextVersion = 1;
};

} // namespace vc::render
