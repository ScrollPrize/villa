#pragma once

// Chunk-footprint estimation for one rendered frame, used to raise a private
// chunk pool's floor so a single render cannot thrash its own cap.
//
// Deliberately Qt-free, header-only and pure so it can be tested directly: an
// earlier version of this arithmetic measured the frame's extent in *level-0*
// voxels (framebuffer pixels / camera scale), which grows without bound as a
// view zooms out even though the frame then touches *fewer* chunks, because it
// is sampled at a coarser pyramid level. That inflated a "bounded" pool's
// capacity into the hundreds of GB and stopped it evicting.
//
// The fix is structural rather than a comment: none of these functions take a
// camera scale, so a scale-dependent span cannot be passed in. Spans are in
// voxels of the level the frame is sampled at, which is framebuffer pixels --
// the render level tracks the zoom, so one pixel is about one voxel there.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace vc3d {

// Bytes of decoded chunks a frame spanning `spanVoxelsXyz` touches, for a volume
// with `chunkShapeZyx` (zarr storage order) and `bytesPerVoxel`. The +1 per axis
// covers the frame's unaligned start offset within a chunk.
inline std::size_t chunkFootprintBytes(const std::array<int, 3>& chunkShapeZyx,
                                       std::size_t bytesPerVoxel,
                                       const std::array<double, 3>& spanVoxelsXyz)
{
    if (chunkShapeZyx[0] <= 0 || chunkShapeZyx[1] <= 0 || chunkShapeZyx[2] <= 0 ||
        bytesPerVoxel == 0) {
        return 0;
    }
    const std::array<int, 3> chunkExtentXyz{chunkShapeZyx[2], chunkShapeZyx[1],
                                            chunkShapeZyx[0]};
    double chunks = 1.0;
    for (int axis = 0; axis < 3; ++axis) {
        // Check before clamping: std::max(0.0, NaN) returns 0.0, which would
        // hide a non-finite span instead of rejecting it.
        if (!std::isfinite(spanVoxelsXyz[axis]))
            return 0;
        const double span = std::max(0.0, spanVoxelsXyz[axis]);
        chunks *= std::ceil(span / double(chunkExtentXyz[axis])) + 1.0;
    }
    const double chunkBytes = double(chunkShapeZyx[0]) * double(chunkShapeZyx[1]) *
                              double(chunkShapeZyx[2]) * double(bytesPerVoxel);
    const double total = chunks * chunkBytes;
    if (!std::isfinite(total) || total <= 0.0)
        return 0;
    return std::size_t(
        std::min(total, double(std::numeric_limits<std::size_t>::max() / 4)));
}

// A plane view's frame. `basisX` / `basisY` / `normal` are the plane's unit
// vectors in volume space -- `seg xz` and `seg yz` carry rotation and tilt, so
// the frame's pixel extent is projected onto the volume axes rather than
// assuming axis alignment. `layerThicknessVoxels` is the composite layer stack's
// depth along the normal (1 when compositing is off).
struct PlaneFrameGeometry {
    int fbW = 0;
    int fbH = 0;
    std::array<double, 3> basisX{1.0, 0.0, 0.0};
    std::array<double, 3> basisY{0.0, 1.0, 0.0};
    std::array<double, 3> normal{0.0, 0.0, 1.0};
    double layerThicknessVoxels = 1.0;
};

inline std::size_t planeFrameChunkFootprintBytes(const std::array<int, 3>& chunkShapeZyx,
                                                 std::size_t bytesPerVoxel,
                                                 const PlaneFrameGeometry& frame)
{
    std::array<double, 3> spanXyz{};
    for (int axis = 0; axis < 3; ++axis) {
        spanXyz[axis] = std::abs(frame.basisX[axis]) * double(frame.fbW) +
                        std::abs(frame.basisY[axis]) * double(frame.fbH) +
                        std::abs(frame.normal[axis]) * frame.layerThicknessVoxels;
    }
    const std::size_t boxBytes = chunkFootprintBytes(chunkShapeZyx, bytesPerVoxel, spanXyz);

    // The product above is the frame's axis-aligned bounding box, which badly
    // over-counts an oblique plane: the frame is a 2D sheet, so it only cuts a
    // staircase of chunks through that box, not the whole volume of it. A
    // diagonal 1500x1000 frame on 128^3 chunks has a 2197-chunk bbox but crosses
    // a few hundred. Bound it by its in-plane cell count instead, with a small
    // allowance for straddling boundaries in the remaining axis.
    const int inPlaneX = chunkShapeZyx[2];
    const int inPlaneY = chunkShapeZyx[1];
    const double cellsX = std::ceil(double(frame.fbW) / double(inPlaneX)) + 1.0;
    const double cellsY = std::ceil(double(frame.fbH) / double(inPlaneY)) + 1.0;
    const double depthCells =
        std::ceil(std::max(0.0, frame.layerThicknessVoxels) / double(chunkShapeZyx[0])) + 3.0;
    const double chunkBytes = double(chunkShapeZyx[0]) * double(chunkShapeZyx[1]) *
                              double(chunkShapeZyx[2]) * double(bytesPerVoxel);
    const double sheetTotal = cellsX * cellsY * depthCells * chunkBytes;
    if (!std::isfinite(sheetTotal) || sheetTotal <= 0.0)
        return boxBytes;
    const std::size_t sheetBytes = std::size_t(
        std::min(sheetTotal, double(std::numeric_limits<std::size_t>::max() / 4)));
    return std::min(boxBytes, sheetBytes);
}

// A generated-surface (flattened) frame. A curved sheet has no closed form, so
// use the axis-aligned equivalent of its pixel extent and let the configured
// floor dominate.
inline std::size_t surfaceFrameChunkFootprintBytes(const std::array<int, 3>& chunkShapeZyx,
                                                   std::size_t bytesPerVoxel,
                                                   int fbW,
                                                   int fbH,
                                                   double layerThicknessVoxels)
{
    return chunkFootprintBytes(chunkShapeZyx, bytesPerVoxel,
                               {double(fbW), double(fbH), layerThicknessVoxels});
}

// One round of concurrent SurfaceCache tile fills. A tile is
// `tileSize x tileSize` samples of surface space swept over the whole normal
// band; in the sampled level's grid space that is about
// `tileSize x tileSize x band`, so the chunk count barely varies with level.
inline std::size_t surfaceTileFillChunkFootprintBytes(
    const std::array<int, 3>& chunkShapeZyx,
    std::size_t bytesPerVoxel,
    int tileSize,
    int bandVoxels,
    std::size_t concurrentTiles)
{
    const std::size_t perTile = chunkFootprintBytes(
        chunkShapeZyx, bytesPerVoxel,
        {double(tileSize), double(tileSize), double(bandVoxels)});
    if (perTile == 0 || concurrentTiles == 0)
        return 0;
    if (perTile > std::numeric_limits<std::size_t>::max() / concurrentTiles)
        return std::numeric_limits<std::size_t>::max() / 4;
    return perTile * concurrentTiles;
}

} // namespace vc3d
