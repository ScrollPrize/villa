#include "Fiber3DSurfaceMesh.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vc3d::fibers3d::surface
{
std::shared_ptr<Mesh> build(const cv::Mat_<cv::Vec3f>& points, size_t maxBytes, const std::atomic_bool& cancelled)
{
    auto mesh = std::make_shared<Mesh>();
    if (cancelled.load())
        return {};
    if (points.rows < 2 || points.cols < 2)
        return mesh;
    // Bound allocations before copying points or allocating the index map.
    constexpr size_t bytesPerPoint = sizeof(Vertex) + 6 * sizeof(uint32_t);
    if (points.total() > maxBytes / bytesPerPoint || points.total() >= UINT32_MAX)
        throw std::runtime_error("Surface exceeds the 3D memory budget. Increase Memory budget in Settings.");
    if (size_t(points.rows - 1) * (points.cols - 1) > size_t(INT_MAX) / 6)
        throw std::runtime_error("Surface exceeds the 3D renderer's index capacity.");
    const auto valid = [](cv::Vec3f p) { return p[0] != -1.f && std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]); };
    std::vector<uint32_t> ids(points.total(), UINT32_MAX);
    mesh->vertices.reserve(points.total());
    mesh->indices.reserve(size_t(points.rows - 1) * (points.cols - 1) * 6);
    const double inf = std::numeric_limits<double>::infinity();
    mesh->bounds = {{inf, inf, inf}, {-inf, -inf, -inf}};
    bool hasOrigin = false;
    auto vertex = [&](int y, int x) {
        auto& id = ids[size_t(y) * points.cols + x];
        if (id != UINT32_MAX)
            return id;
        const auto p = points(y, x);
        if (!hasOrigin) {
            mesh->origin = {p[0], p[1], p[2]};
            hasOrigin = true;
        }
        Vertex v{};
        for (int k = 0; k < 3; ++k) {
            v.position[k] = float(double(p[k]) - mesh->origin[k]);
            mesh->bounds.lower[k] = std::min(mesh->bounds.lower[k], double(p[k]));
            mesh->bounds.upper[k] = std::max(mesh->bounds.upper[k], double(p[k]));
        }
        id = uint32_t(mesh->vertices.size());
        mesh->vertices.push_back(v);
        return id;
    };
    auto triangle = [&](uint32_t a, uint32_t b, uint32_t c) {
        const auto* p = mesh->vertices[a].position;
        const auto* q = mesh->vertices[b].position;
        const auto* r = mesh->vertices[c].position;
        const cv::Vec3d u(double(q[0]) - p[0], double(q[1]) - p[1], double(q[2]) - p[2]);
        const cv::Vec3d v(double(r[0]) - p[0], double(r[1]) - p[1], double(r[2]) - p[2]);
        auto n = u.cross(v);
        const auto length = cv::norm(n);
        if (!(length > 0) || !std::isfinite(length))
            return;
        n /= length;
        mesh->indices.insert(mesh->indices.end(), {a, b, c});
        for (auto id : {a, b, c})
            for (int k = 0; k < 3; ++k)
                mesh->vertices[id].normal[k] += float(n[k]);
    };
    for (int y = 0; y + 1 < points.rows; ++y) {
        if (cancelled.load())
            return {};
        for (int x = 0; x + 1 < points.cols; ++x) {
            // Match QuadSurface's four-corner validity rule. In particular,
            // never bridge an invalid cell, including a single missing point.
            if (!valid(points(y, x)) || !valid(points(y, x + 1)) || !valid(points(y + 1, x)) || !valid(points(y + 1, x + 1)))
                continue;
            const auto a = vertex(y, x), b = vertex(y, x + 1), c = vertex(y + 1, x), d = vertex(y + 1, x + 1);
            triangle(a, b, d);
            triangle(a, d, c);
        }
    }
    return mesh;
}
}  // namespace vc3d::fibers3d::surface
