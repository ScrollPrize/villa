#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <opencv2/core.hpp>
#include <QImage>

#include "TiledViewerCamera.hpp"
#include "TileScene.hpp"
#include "vc/core/util/Compositing.hpp"

class Surface;
class PlaneSurface;
class QuadSurface;
class Volume;
namespace vc { class VcDataset; }

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
    bool useFastInterpolation = false;  // nearest-neighbor instead of trilinear
    CompositeRenderSettings compositeSettings;  // multi-layer composite params
};

// Result from rendering a single tile
struct TileRenderResult {
    WorldTileKey worldKey;
    QImage image;        // Format_RGB32 QImage (produced directly on worker thread)
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

// Stateless tile renderer. Thread-safe (no Qt objects, no mutable state).
// Extracted from CVolumeViewer::render_area() logic.
class TileRenderer
{
public:
    // Render a single tile synchronously.
    // All inputs passed by value/pointer - no shared mutable state.
    // Volume owns its cache; no external ChunkCache needed.
    static TileRenderResult renderTile(
        const TileRenderParams& params,
        const std::shared_ptr<Surface>& surface,
        Volume* volume);

private:
    // Generate view coordinates for the tile
    static void generateTileCoords(
        cv::Mat_<cv::Vec3f>& coords,
        const TileRenderParams& params,
        const std::shared_ptr<Surface>& surface);
};
