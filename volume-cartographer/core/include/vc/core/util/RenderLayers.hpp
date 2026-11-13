#pragma once

#include "vc/core/util/RenderCanvas.hpp"

#include <vc_reader/grid.hpp>

#include <opencv2/core/mat.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

class QuadSurface;
namespace vcr::catalog { struct Affine; }
namespace vcr::zarr { struct LevelPlacement; }

// How vc_render_tifxyz samples a surface's layers from a volume, for the tool and
// for code that needs the pixels of a render without writing one.

namespace vc {

// Edge of the tiles vc_render_tifxyz samples and writes: its Zarr chunks and TIFF tiles.
inline constexpr int kRenderTile = 128;

// Surface coordinates (working-level voxels, XYZ) to indices of the rendered
// pyramid level: index = scale * c + offset per axis, from the two levels' placements.
struct RenderLevelMap {
    cv::Vec3f scale{1, 1, 1}, offset{0, 0, 0};

    // Rendered-level voxels per working level-0 voxel (the mean over the axes: the
    // renderer's scale is one number). A render at --scale s has render scale
    // s * voxelScale().
    [[nodiscard]] float voxelScale() const { return (scale[0] + scale[1] + scale[2]) / 3.0f; }
};

[[nodiscard]] RenderLevelMap renderLevelMap(const vcr::zarr::LevelPlacement& working,
                                            const vcr::zarr::LevelPlacement& rendered);

// The surface with its vertices in rendered-level index space (XYZ): the segmentation
// scale, the affine and the level placement applied to each vertex once. Bilinear and
// Hermite patches commute with affine maps, so positions are those of the mapped
// vertex grid; normals are its patch normals and offsets along them are in
// rendered-level voxels. A copy: later edits of `surf` do not reach it.
struct RenderSurface {
    cv::Mat_<cv::Vec3f> vertices;
    std::unique_ptr<vcr::grid::Surface> grid;
};

[[nodiscard]] RenderSurface renderSurface(const QuadSurface& surf, vcr::grid::Interpolation interpolation, float scaleSeg,
                                          bool hasAffine, const vcr::catalog::Affine& aff, const RenderLevelMap& level);

// The offsets of `numSlices` layers along the unit surface normal, centered on the
// surface, in rendered-level voxels: the vertices are already in rendered-level index
// space and normals are unit length, so a step of sliceStep advances sliceStep
// rendered-level voxels. Do not scale them by the level's scale: the spacing along
// the normal would then be level 0's while the spacing in the plane is the rendered
// level's, an anisotropic stack that does not match the isotropic .zattrs scale.
// With accumOffsets, every layer lists its sub-steps (its offset plus each of them).
[[nodiscard]] std::vector<float> renderLayerOffsets(int numSlices, double sliceStep,
                                                    const std::vector<float>& accumOffsets = {});

// The raster of the pixels `pixels` of a canvas of size `canvas` rendered at
// `renderScale`: the renderer's arithmetic for one tile. It reads the scale and the
// origin of `surf`, so after shiftSurfaceOrigin the raster is no longer where
// RenderCanvas places the pixels on the grid.
[[nodiscard]] vcr::grid::Raster renderRaster(const QuadSurface& surf, cv::Size canvas, float renderScale,
                                             const cv::Rect& pixels);

// A tile's layers at `offsets` along the normal (negated with flipNormals, which
// reverses the layer order), linear or nearest, optionally reduced to one image; NaN
// where the surface has no point. `level` reads the rendered level with unit scale:
// the level's placement is in `surface`.
[[nodiscard]] vcr::grid::Result sampleRenderTile(const RenderSurface& surface, const vcr::grid::Raster& raster,
                                                 const vcr::grid::Level& level, const std::vector<float>& offsets,
                                                 bool flipNormals, int order,
                                                 const std::optional<vcr::composite::Options>& composite = std::nullopt,
                                                 const std::atomic<bool>* cancel = nullptr);

// One image of a result, result.width * result.height values: pixels without a
// value read 0; integer intensities round to nearest, ties to even.
void renderLayerPixels(const vcr::grid::Result& result, int layer, std::uint8_t* out);
void renderLayerPixels(const vcr::grid::Result& result, int layer, std::uint16_t* out);

// The layers at `offsets` of the canvas pixels `pixels`: a (layers, height, width)
// array, uint8 or uint16 as the volume, equal to the same pixels of a
// vc_render_tifxyz render of the whole canvas (no crop, rotation or flip). The
// window is sampled in the renderer's tiles, each whole and with the renderer's
// raster, so the values do not depend on where the window starts. Tiles run on
// the volume's thread pool. `surface` is renderSurface(surf, ...) for the canvas's
// level; `surf` gives the rasters, so its scale and origin must not change during
// the call. Throws std::invalid_argument when `pixels` is empty or not inside the
// canvas, the canvas is not the surface's (another grid or scale, or an origin
// moved by shiftSurfaceOrigin), the level carries a placement, or the volume is not
// uint8 or uint16, and vcr::Cancelled when `cancel` is set before the last chunk
// is read.
[[nodiscard]] cv::Mat sampleRenderLayers(const QuadSurface& surf, const RenderSurface& surface,
                                         const vcr::grid::Level& level, const RenderCanvas& canvas, const cv::Rect& pixels,
                                         const std::vector<float>& offsets, bool flipNormals = false,
                                         const std::atomic<bool>* cancel = nullptr);

}  // namespace vc
