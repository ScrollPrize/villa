#pragma once
#include "Fiber3DCT.hpp"
#include <memory>

namespace vc3d::fibers3d::slices
{
struct Plane {
    int axis = 0; // XY, XZ, YZ: identities, even when the linked view is tilted.
    Point origin{}, right{}, down{}; // top-left sample, full horizontal/vertical spans
    int width = 0, height = 0;
    bool volumetric = false;
    Point normal{0, 0, 1}; // signed native-CT normal, including Reverse
    int front = 0, behind = 0;
    bool operator==(const Plane&) const = default;
};
struct Slab {
    Point origin{}, right{}, down{}, through{}; // outer texture bounds, in CT voxels
    int width = 0, height = 0, depth = 0;
    std::vector<uint8_t> pixels; // interleaved gray/validity, X fastest
};
struct Image {
    Plane plane;
    ct::Tile tile;
    int level = 0;
    bool preview = false;
    std::optional<Slab> slab;
};
using Images = std::vector<std::shared_ptr<const Image>>;

std::shared_ptr<Image> sample(const Plane&, Volume&, const std::atomic_bool* cancelled = nullptr, int maxTextureEdge = 1024);
std::array<Point, 8> corners(const Image&);

struct Vertex { Point position; double u = 0, v = 0; };
struct Polygon { size_t image = 0; std::vector<Vertex> vertices; };
// A tiny BSP splits the three planes at their intersections. Traversal then
// gives correct alpha order from either side, including intersecting slices.
struct Tree {
    Polygon polygon;
    Point normal{};
    std::unique_ptr<Tree> front, back;
};
std::unique_ptr<Tree> partition(const Images&);
std::vector<const Polygon*> backToFront(const Tree*, const Point& eye);
}
