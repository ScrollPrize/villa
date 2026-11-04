#pragma once
#include "scene3d/Scene3DGeometry.hpp"
#include <vector>
#include <cstdint>

namespace vc3d::fibers3d { using scene3d::Point; using scene3d::Bounds; }
#include <atomic>
#include <opencv2/core.hpp>

class OpenedVolumes;
class Volume;

namespace vc3d::fibers3d::ct
{
struct Tile {
    std::vector<Point> left, right;
    int width = 0, height = 0;
    // Interleaved grayscale / validity, independent of display window/level.
    std::vector<uint8_t> pixels;
};

// Row endpoints and all sample coordinates use active level-0 CT voxels.
// Bounded, cancellable trilinear sampling; no slab, MIP or extra disk cache. Blocking
// reads: call off the GUI thread.
// Blocking CT sampling, with the same uint16 conversion and bounds mask.
cv::Mat_<cv::Vec2b> sampleCoordinates(const cv::Mat_<cv::Vec3f>&, OpenedVolumes&, const Volume&, int level);
void sampleTile(Tile&, OpenedVolumes&, const Volume&, int level, const std::atomic_bool* cancelled = nullptr);
// The present level with the finest voxels no smaller than `spacing` level-0 voxels; the
// finest present level when all are coarser.
int levelForSpacing(OpenedVolumes&, const Volume&, double spacing);
}  // namespace vc3d::fibers3d::ct
