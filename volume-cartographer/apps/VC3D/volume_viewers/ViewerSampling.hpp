#pragma once

#include "OpenedVolumes.hpp"

#include <vc_reader/grid.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <opencv2/core.hpp>

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
    // Set by a failed GPU frame: these levels render on the CPU from then on (gridLevel
    // gives vcr::grid no reader). Device bricks never load after a failure, so a
    // cached-only GPU frame would stay empty.
    std::shared_ptr<std::atomic_bool> gpuFailed = std::make_shared<std::atomic_bool>(false);

    int count() const { return static_cast<int>(levels.size()); }
    bool has(int level) const;
    // `level` for vcr::grid: working level-0 XYZ points to its indices, with its CUDA
    // reader unless a GPU frame has failed.
    vcr::grid::Level gridLevel(int level) const;
    // Every level for vcr::grid's level choice; absent levels have no array.
    std::vector<vcr::grid::Level> gridLevels() const;
    // A frame's request: the start level, then up to `fallbackLevels` coarser ones.
    vcr::grid::Request request(int startLevel, int fallbackLevels) const;
    // The relative size of one working level-0 voxel along X, Y, Z, finest axis 1.
    std::array<float, 3> vertexSpacing() const;
};

// Opens every present working level of `volume`. Levels that fail to open are absent.
ViewerLevels openViewerLevels(OpenedVolumes& opened, const Volume& volume);

// Pixels per working voxel of a flattened view: vcr::grid::pixelsPerVoxel from the
// surface's tangents (UV units need not span one voxel), else `nominalPixelsPerVoxel`.
float surfacePixelsPerVoxel(const QuadSurface& surface, const vcr::grid::Raster& viewport,
                            float nominalPixelsPerVoxel);

// An owned native snapshot of the whole grid; retained by background jobs.
std::shared_ptr<const vcr::grid::Surface> gridSurface(const QuadSurface& surface, const std::array<float, 3>& spacing = {1, 1, 1});

// The grid vertices a frame of `raster`, grown by the given pixels per side, interpolates,
// padded so its positions and normals match the whole grid's.
cv::Rect gridWindowFor(const QuadSurface& surface, const vcr::grid::Raster& raster, int left, int top, int right, int bottom);

// The reads frames could not sample from the caches, with the levels they read
// (kept open until the reads ran).
struct ViewerLoad {
    vcr::grid::Pending pending;
    std::vector<std::shared_ptr<const ViewerLevels>> levels;

    bool empty() const { return pending.empty(); }
};

// vcr::grid::sampleCached over `levels`, adding what the frame lacked to `load`. A GPU
// failure renders `levels` on the CPU from then on.
vcr::grid::Result sampleCached(const std::shared_ptr<const ViewerLevels>& levels, const vcr::grid::Frame& frame,
                               const vcr::grid::Request& request, ViewerLoad* load);

// vcr::grid::load: coarse reads, then target ones; `afterPart` runs after each part
// that had reads. Once `cancel` is set no further chunk read starts and loading stops.
// A GPU failure renders the load's levels on the CPU from then on.
void loadPending(const ViewerLoad& load, const std::function<void()>& afterPart,
                 const std::atomic<bool>* cancel = nullptr);

} // namespace vc3d
