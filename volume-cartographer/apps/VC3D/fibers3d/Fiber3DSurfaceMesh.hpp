#pragma once
#include "Fiber3DScene.hpp"
#include <opencv2/core.hpp>
#include <atomic>
#include <memory>

namespace vc3d::fibers3d::surface
{
struct Vertex {
    float position[3], normal[3];
};
struct Mesh {
    Point origin{};
    Bounds bounds{};
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
};
// A display snapshot, in native CT coordinates. No resampling, hole filling,
// reconstruction, or modification of the source QuadSurface.
std::shared_ptr<Mesh> build(const cv::Mat_<cv::Vec3f>& points, size_t maxBytes, const std::atomic_bool& cancelled);
}  // namespace vc3d::fibers3d::surface
