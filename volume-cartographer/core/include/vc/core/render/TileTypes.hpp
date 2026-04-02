#pragma once

#include <cstdint>
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/core.hpp>

#include "vc/core/util/Compositing.hpp"

class Volume;

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
    constexpr int DRAIN_BATCH_SIZE     = 128; // max results drained per tick cycle
}

// Per-tile metadata for staleness checks during progressive rendering.
struct TileMetadata {
    uint64_t epoch = 0;
    int8_t level = -1;    // pyramid level of current pixmap (-1 = placeholder)
};

// Parameters for a single tile render call.
// Collected on the main thread, passed to the renderer.
struct TileRenderParams {
    // --- Tile identity ---
    WorldTileKey worldKey;
    uint64_t epoch = 0;

    // --- Tile geometry ---
    // Surface parameter space ROI for this tile (same for both surface types)
    cv::Rect2f surfaceROI;
    int tileW = 0;   // tile width in pixels
    int tileH = 0;   // tile height in pixels

    // --- Camera state ---
    float scale = 1.0f;       // user zoom level
    float dsScale = 1.0f;     // pyramid downscale factor at dsScaleIdx
    int dsScaleIdx = 0;       // pyramid level index (0 = finest)
    float zOff = 0.0f;        // Z-axis slice offset

    // --- Render settings ---
    float windowLow = 0.0f;            // window/level low bound
    float windowHigh = 255.0f;         // window/level high bound
    bool stretchValues = false;         // auto-stretch intensity range
    std::string colormapId;             // colormap identifier (empty = grayscale)
    std::shared_ptr<Volume> overlayVolume; // optional overlay volume sampled with the same coords
    float overlayOpacity = 0.0f;        // overlay alpha in [0,1]
    float overlayWindowLow = 0.0f;      // overlay threshold/window low
    float overlayWindowHigh = 255.0f;   // overlay window high
    std::string overlayColormapId;      // overlay colormap identifier
    bool useFastInterpolation = false;  // nearest-neighbor instead of trilinear
    bool isPlaneSurface = false;        // pre-computed: surface is PlaneSurface (avoids RTTI)
    CompositeRenderSettings compositeSettings;  // multi-layer composite params

    // --- Pool scheduling ---
    int submitPriority = 0;  // lower = higher urgency; set by TileRenderController
};

// Result from rendering a single tile.
// TileRenderer fills the raw pixels buffer; the Qt layer converts to QPixmap
// for display, then releases the pixel data.
struct TileRenderResult {
    WorldTileKey worldKey;
    std::vector<uint32_t> pixels;  // ARGB32 buffer (tileW * tileH)
    int width = 0;                 // tile width in pixels
    int height = 0;                // tile height in pixels
    uint64_t epoch = 0;

    // Camera state snapshot for cache key reconstruction
    float scale = 1.0f;
    float zOff = 0.0f;
    int dsScaleIdx = 0;

    // Actual pyramid level used (may differ from requested if best-effort)
    int actualLevel = 0;

    // Identifies which controller submitted this task (for shared pool routing)
    int controllerId = -1;
};
