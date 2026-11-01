#pragma once
#include "Fiber3DSurfaceMesh.hpp"
#include "Fiber3DRenderGeometry.hpp"

namespace vc3d::fibers3d::surface
{
struct Index {
    struct Block {
        Bounds bounds;
        size_t begin, end;
    };
    struct Node {
        Bounds bounds;
        size_t begin, end, left = 0, right = 0;
    };
    std::shared_ptr<const Mesh> mesh;
    std::vector<Block> blocks;
    std::vector<Node> nodes;
};
std::shared_ptr<Index> index(std::shared_ptr<const Mesh>, const std::atomic_bool&);
double segmentTriangleDistance(const Point& from, const Point& to, const Point& a, const Point& b, const Point& c);
// Intersect with an optional existing visibility filter. Whole source fibres
// are retained if any original, clipped segment is within radius of a triangle.
render::Visibility nearby(const render::Geometry&, const Index&, double radius, const render::Visibility&, const std::atomic_bool&);
}  // namespace vc3d::fibers3d::surface
