#pragma once

#include "Fiber3DScene.hpp"
#include <atomic>
#include <functional>
#include <memory>
#include <opencv2/core.hpp>

class Volume;

namespace vc3d::fibers3d::ribbon
{

using NormalAt = std::function<std::optional<cv::Vec3d>(const cv::Vec3d&)>;
struct Row {
    cv::Vec3d center, tangent, normal;
    bool sampledNormal = false;
};
using Strip = std::vector<Row>;
struct Frames {
    std::vector<Strip> strips;
    size_t sampledNormals = 0, rows = 0;
};

// No geometry is connected across runs, invalid coordinates or explicitly
// missing annotation spans. Original vertices are retained; long segments are
// subdivided to <= 1 CT voxel. Uses the same frame transport as Line Annotation.
Frames makeFrames(const Fiber& fiber, const std::optional<Bounds>& clip, const NormalAt& normalAt = {}, const std::atomic_bool* cancelled = nullptr);
cv::Vec3d across(const Row& row, double rotationDegrees);

struct Tile {
    std::vector<Point> left, right;
    int width = 0, height = 0;
    // Interleaved grayscale / validity, independent of display window/level.
    std::vector<uint8_t> pixels;
};
struct Image {
    QString key;
    std::vector<Tile> tiles;
    size_t sampledNormals = 0, rows = 0;
    int widthVoxels = 0, rotationDegrees = 0;
};

// One trilinear CT sample per texel at level 0: no depth slab or MIP.
// Bounded tiles share a border row; their mesh and sampling coordinates agree.
// Throws on unsupported CT type, missing L0, cancellation or the 8M-texel cap.
std::shared_ptr<Image> sample(const Frames& frames, Volume& volume, int widthVoxels, int rotationDegrees, const std::atomic_bool* cancelled = nullptr);

}  // namespace vc3d::fibers3d::ribbon
