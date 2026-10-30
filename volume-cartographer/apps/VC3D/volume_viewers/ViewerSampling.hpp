#pragma once

#include "OpenedVolumes.hpp"
#include "vc/core/types/Sampling.hpp"

#include <vc_reader/grid.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

class QuadSurface;
class Volume;
struct CompositeParams;

namespace vc3d {

// The working levels of one volume that a viewer samples, opened through the
// shared OpenedVolumes. Index: working level; an absent level has no array.
struct ViewerLevels {
    std::vector<OpenedVolumes::Level> levels;

    int count() const { return static_cast<int>(levels.size()); }
    bool has(int level) const;
    // Working level-0 voxels one index of `level` spans, per axis (ZYX).
    std::array<double, 3> voxelExtent(int level) const;
    // `level` for vcr::grid: working level-0 XYZ points to its indices.
    vcr::grid::Level gridLevel(int level) const;
};

// Opens every present working level of `volume`. Levels that fail to open are absent.
ViewerLevels openViewerLevels(OpenedVolumes& opened, const Volume& volume);

// Level choices of the viewer: the coarsest level whose largest voxel extent is at
// most 1 / (pixelsPerBaseVoxel * qualityBias), or the finest level when none is.
int sourceLevelForView(const ViewerLevels& levels, float pixelsPerBaseVoxel, float qualityBias);
double maximumBaseVoxelExtent(const ViewerLevels& levels, int level);
// How many coarser levels may fill a viewport's holes: up to the first level one
// of whose chunks spans the larger viewport edge.
int fallbackLevelCountForViewport(const ViewerLevels& levels, int startLevel, int viewportWidth,
                                  int viewportHeight, float pixelsPerBaseVoxel,
                                  int maximumFallbackLevels = 5);

// The vertex grid of a QuadSurface for vcr::grid, with its interpolation mode,
// quad validity and components. It views the surface's points: rebuild it after
// any edit.
std::shared_ptr<const vcr::grid::Surface> gridSurface(const QuadSurface& surface);

// What a frame samples: a plane, or a raster over a grid surface.
struct ViewerGeometry {
    std::optional<vcr::grid::Plane> plane;
    std::shared_ptr<const vcr::grid::Surface> surface;
    vcr::grid::Raster raster;

    int width() const { return plane ? plane->width : raster.width; }
    int height() const { return plane ? plane->height : raster.height; }
    // On the GPU when every requested level of `levels` has a CUDA reader.
    vcr::grid::Result sample(const vcr::grid::Request& request, const ViewerLevels& levels) const;
};

// A blocking read of what a frame could not sample from the caches: its geometry
// and layers at one level, for the pixels that lacked it.
struct ViewerRead {
    ViewerGeometry geometry;
    std::shared_ptr<const ViewerLevels> levels;  // keeps the level's array open
    vcr::grid::Request request;
    std::vector<std::uint8_t> pixels;
};

// The reads one frame asks for: its coarsest fallback level first (a quick
// preview), then its start level.
struct ViewerLoad {
    std::vector<ViewerRead> coarse, target;

    bool empty() const { return coarse.empty() && target.empty(); }
};

// What a frame samples of one volume: its layers (offsets along the normal),
// optionally reduced to one value, from the start level with up to
// `fallbackLevels` coarser ones filling holes.
struct ViewerSample {
    std::shared_ptr<const ViewerLevels> levels;
    int startLevel = 0;
    int fallbackLevels = 0;
    vc::Sampling sampling = vc::Sampling::Nearest;
    std::vector<float> offsets{0.0f};
    std::optional<vcr::composite::Options> composite;
};

// Samples `what` over `geometry` from the caches alone and adds the reads of what
// it lacked to `load`.
vcr::grid::Result sampleCached(const ViewerGeometry& geometry, const ViewerSample& what, ViewerLoad* load);

// Runs the reads (coarse first, then target), which bring their chunks into the
// caches. `afterPart` runs after each part that had reads; returning false stops.
void loadPending(const ViewerLoad& load, const std::function<bool()>& afterPart);

// The composite reduction of a viewer composite method, or nullopt when the method
// is not a vcr::composite reduction (e.g. "volumetric").
std::optional<vcr::composite::Options> compositeOptions(const CompositeParams& params);

// Layer `layer` of a result as 8-bit display values and coverage (pixels with a value).
void toDisplay(const vcr::grid::Result& result, int layer, uint8_t* values, uint8_t* coverage);

}  // namespace vc3d
