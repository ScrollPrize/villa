#pragma once
#include "scene3d/Scene3DGeometry.hpp"
#include <vector>
#include <cstdint>

namespace vc3d::fibers3d { using scene3d::Point; using scene3d::Bounds; }
#include <atomic>
#include <opencv2/core.hpp>

class Volume;
namespace vc::render
{
class IChunkedArray;
}

namespace vc3d::fibers3d::ct
{
struct Tile {
    std::vector<Point> left, right;
    int width = 0, height = 0;
    // Interleaved grayscale / validity, independent of display window/level.
    std::vector<uint8_t> pixels;
};

// Row endpoints and all sample coordinates use active level-0 CT voxels.
// Bounded, cancellable trilinear sampling; no slab, MIP or extra disk cache.
// Shared blocking CT sampling, with the same uint16 conversion and bounds mask.
cv::Mat_<cv::Vec2b> sampleCoordinates(const cv::Mat_<cv::Vec3f>&, Volume&, int level);
// Shared display normalization for raw uint8/uint16 CT data.
cv::Mat_<uint8_t> displayBytes(const cv::Mat&);
void sampleTile(Tile&, Volume&, int level, const std::atomic_bool* cancelled = nullptr);

struct Progress {
    size_t valid = 0, covered = 0, requestedLevel = 0;
    int errors = 0;
};
// Interactive sampling through the shared viewer cache. Never waits for I/O;
// uncovered pixels remain transparent and cached coarse levels fill previews.
Progress sampleTileProgressive(Tile&, vc::render::IChunkedArray&, int level, const std::atomic_bool* cancelled = nullptr);
}  // namespace vc3d::fibers3d::ct
