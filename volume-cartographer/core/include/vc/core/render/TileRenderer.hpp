#pragma once

#include <memory>
#include <opencv2/core.hpp>

#include "vc/core/render/TileTypes.hpp"

class Surface;
class PlaneSurface;
class QuadSurface;
class Volume;

namespace vc {

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

}  // namespace vc
