#include "Fiber3DSlices.hpp"
#include "Fiber3DGeometry.hpp"
#include "vc/core/types/Volume.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vc3d::fibers3d::slices
{
namespace
{
cv::Vec3d vec(const Point& p) { return {p[0], p[1], p[2]}; }
Point point(cv::Vec3d p) { return {p[0], p[1], p[2]}; }
Point at(const Plane& p, double u, double v) { return point(vec(p.origin) + vec(p.right) * u + vec(p.down) * v); }
void insert(std::unique_ptr<Tree>& tree, Polygon polygon)
{
    if (polygon.vertices.size() < 3)
        return;
    if (!tree) {
        auto n = (vec(polygon.vertices[1].position) - vec(polygon.vertices[0].position))
                     .cross(vec(polygon.vertices[2].position) - vec(polygon.vertices[0].position));
        const double length = cv::norm(n);
        if (length < 1e-12)
            return;
        tree = std::make_unique<Tree>();
        tree->normal = point(n / length);
        tree->polygon = std::move(polygon);
        return;
    }
    const auto origin = vec(tree->polygon.vertices[0].position), normal = vec(tree->normal);
    Polygon front{polygon.image, {}}, back{polygon.image, {}};
    bool coplanar = true;
    for (size_t i = 0; i < polygon.vertices.size(); ++i) {
        const auto& a = polygon.vertices[i];
        const auto& b = polygon.vertices[(i + 1) % polygon.vertices.size()];
        const double da = (vec(a.position) - origin).dot(normal), db = (vec(b.position) - origin).dot(normal);
        coplanar = coplanar && std::abs(da) <= 1e-8;
        if (da >= -1e-8) front.vertices.push_back(a);
        if (da <= 1e-8) back.vertices.push_back(a);
        if ((da < -1e-8 && db > 1e-8) || (da > 1e-8 && db < -1e-8)) {
            const double t = da / (da - db);
            Vertex cut{interpolate(a.position, b.position, t), a.u + (b.u - a.u) * t, a.v + (b.v - a.v) * t};
            front.vertices.push_back(cut);
            back.vertices.push_back(cut);
        }
    }
    // Coplanar planes need only one copy. A consistent order is sufficient.
    if (coplanar)
        back.vertices.clear();
    insert(tree->front, std::move(front));
    insert(tree->back, std::move(back));
}
}

std::shared_ptr<Image> sample(const Plane& plane, Volume& volume, const std::atomic_bool* cancelled, int maxTextureEdge)
{
    if (plane.width < 2 || plane.height < 2 || plane.width > 1024 || plane.height > 1024 ||
        !finite(plane.origin) || !finite(plane.right) || !finite(plane.down) ||
        cv::norm(vec(plane.right).cross(vec(plane.down))) < 1e-8)
        throw std::invalid_argument("Invalid CT slice plane");
    if (maxTextureEdge < 2 || maxTextureEdge > 1024)
        throw std::invalid_argument("Invalid CT slice texture limit");
    auto image = std::make_shared<Image>();
    image->plane = plane;
    auto& tile = image->tile;
    const double reduction = std::max(1., double(std::max(plane.width, plane.height) - 1) / (maxTextureEdge - 1));
    tile.width = std::max(2, int(std::ceil((plane.width - 1) / reduction)) + 1);
    tile.height = std::max(2, int(std::ceil((plane.height - 1) / reduction)) + 1);
    image->preview = tile.width < plane.width || tile.height < plane.height;
    const double spacing = std::min(cv::norm(vec(plane.right)) / (tile.width - 1), cv::norm(vec(plane.down)) / (tile.height - 1));
    const int requested = int(std::clamp(std::floor(std::log2(std::max(1., spacing))), 0., 20.));
    image->level = volume.firstPresentScaleLevel();
    for (int level : volume.presentScaleLevels())
        if (level <= requested && level > image->level) image->level = level;
    for (int r = 0; r < tile.height; ++r) {
        const double v = double(r) / (tile.height - 1);
        tile.left.push_back(at(plane, 0, v));
        tile.right.push_back(at(plane, 1, v));
    }
    ct::sampleTile(tile, volume, image->level, cancelled);
    return image;
}

std::unique_ptr<Tree> partition(const Images& images)
{
    std::unique_ptr<Tree> tree;
    for (size_t i = 0; i < images.size(); ++i) {
        const auto& p = images[i]->plane;
        insert(tree, Polygon{i, {{at(p, 0, 0), 0, 0}, {at(p, 1, 0), 1, 0}, {at(p, 1, 1), 1, 1}, {at(p, 0, 1), 0, 1}}});
    }
    return tree;
}
std::vector<const Polygon*> backToFront(const Tree* tree, const Point& eye)
{
    std::vector<const Polygon*> result;
    auto visit = [&](auto&& self, const Tree* node) -> void {
        if (!node) return;
        const bool front = (vec(eye) - vec(node->polygon.vertices[0].position)).dot(vec(node->normal)) >= 0;
        self(self, front ? node->back.get() : node->front.get());
        result.push_back(&node->polygon);
        self(self, front ? node->front.get() : node->back.get());
    };
    visit(visit, tree);
    return result;
}
}
