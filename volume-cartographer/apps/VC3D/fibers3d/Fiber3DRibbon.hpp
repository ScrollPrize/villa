#pragma once

#include "Fiber3DScene.hpp"
#include "Fiber3DCT.hpp"
#include <atomic>
#include <functional>
#include <memory>
#include <opencv2/core.hpp>
#include "vc/lasagna/LineViewBuilder.hpp"

class Volume;

namespace vc3d::fibers3d::ribbon
{

using NormalAt = std::function<std::optional<cv::Vec3d>(const cv::Vec3d&)>;
using ModelForPoints = std::function<vc::lasagna::LineModel(const std::vector<cv::Vec3d>&)>;
struct Row {
    cv::Vec3d center, tangent, normal, side;
    std::array<cv::Vec3d, vc::lasagna::kLineViewCrossSampleCount> surface;
    bool sampledNormal = false;
};
using Strip = std::vector<Row>;
struct Frames {
    std::vector<Strip> strips;
    size_t sampledNormals = 0, rows = 0;
};

// No geometry is connected across runs, invalid coordinates or explicitly
// missing annotation spans. Geometry is the Line Annotation lineSurface grid,
// interpolated at one voxel along its parameterization for CT texture sampling.
Frames makeFrames(
    const Fiber& fiber,
    const std::optional<Bounds>& clip,
    const NormalAt& normalAt = {},
    const std::atomic_bool* cancelled = nullptr,
    const ModelForPoints& modelForPoints = {},
    const std::function<cv::Vec3f(const cv::Vec3f&)>& centerReference = {});
// Reuse an already-built Line Annotation surface (including manual normals).
Frames framesFromSurface(const cv::Mat_<cv::Vec3f>& grid, const cv::Mat_<cv::Vec3f>& sideGrid, double spacing, bool sampledNormals, const std::atomic_bool* cancelled = nullptr);
cv::Vec3d position(const Row& row, double offsetVoxels, double rotationDegrees);

using Tile = ct::Tile;
struct Image {
    QString key;
    std::vector<Tile> tiles;
    size_t sampledNormals = 0, rows = 0;
    int widthVoxels = 0, rotationDegrees = 0;
    int level = 0, errors = 0;
    size_t validTexels = 0, coveredTexels = 0, requestedLevelTexels = 0;
    bool complete = true;
};

// One trilinear sample per texel, with nonblocking viewer-cache sampling at
// the zoom's level and coarse fallback. No depth slab or MIP. Bounded tiles
// share a border row; their mesh and sampling coordinates agree.
// Throws on unsupported CT type, missing level, cancellation or the 8M-texel cap.
std::shared_ptr<Image> sampleProgressive(const Frames&, Volume&, int widthVoxels, int rotationDegrees, int level, const std::atomic_bool* cancelled = nullptr);

}  // namespace vc3d::fibers3d::ribbon
