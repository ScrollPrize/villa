#pragma once
#include "scene3d/Scene3DGeometry.hpp"
#include <vector>
#include <cstdint>

namespace vc3d::fibers3d { using scene3d::Point; using scene3d::Bounds; }
#include <opencv2/core.hpp>
#include <atomic>
#include <memory>

class Volume;

namespace vc3d::fibers3d::surface
{
struct Vertex {
    float position[3], normal[3];
    float uv[2]{};
};
struct Texture {
    int width = 0, height = 0;
    int level = 0; // actual OME-Zarr level sampled
    std::vector<uint8_t> pixels; // interleaved CT gray/validity
};
struct Mesh {
    Point origin{};
    Bounds bounds{};
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::shared_ptr<const Texture> texture;
};
// A display snapshot, in native CT coordinates. No resampling, hole filling,
// reconstruction, or modification of the source QuadSurface.
std::shared_ptr<Mesh> build(const cv::Mat_<cv::Vec3f>& points, size_t maxBytes, const std::atomic_bool& cancelled);
std::shared_ptr<Texture> sampleTexture(const cv::Mat_<cv::Vec3f>& points, cv::Vec2f scale,
                                       Volume&, const std::atomic_bool& cancelled, int level, int maxEdge = 2048);
}  // namespace vc3d::fibers3d::surface
