#include "Fiber3DSurfaceProximity.hpp"

namespace vc3d::fibers3d::surface
{
namespace
{
Point subtract(const Point& a, const Point& b)
{
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
double dot(const Point& a, const Point& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
Point cross(const Point& a, const Point& b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
Point vertex(const Mesh& mesh, uint32_t id)
{
    const auto& p = mesh.vertices[id].position;
    return {mesh.origin[0] + p[0], mesh.origin[1] + p[1], mesh.origin[2] + p[2]};
}
void check(const std::atomic_bool& cancelled)
{
    if (cancelled.load(std::memory_order_relaxed))
        throw std::runtime_error("Cancelled");
}
bool overlaps(const Bounds& a, const Bounds& b)
{
    for (int k = 0; k < 3; ++k)
        if (a.lower[k] > b.upper[k] || a.upper[k] < b.lower[k])
            return false;
    return true;
}
Bounds padded(Bounds b, double radius)
{
    for (int k = 0; k < 3; ++k) {
        b.lower[k] -= radius;
        b.upper[k] += radius;
    }
    return b;
}
}  // namespace
double segmentTriangleDistance(const Point& from, const Point& to, const Point& a, const Point& b, const Point& c)
{
    const auto ab = subtract(b, a), bc = subtract(c, b), ca = subtract(a, c), n = cross(ab, subtract(c, a));
    const double n2 = dot(n, n);
    double best =
        std::min({render::segmentDistance(from, to, a, b), render::segmentDistance(from, to, b, c), render::segmentDistance(from, to, c, a)});
    if (!(n2 > 0))
        return best;
    // Half-space tests apply to the projection onto the triangle plane too.
    const auto inside = [&](const Point& p) {
        return dot(cross(ab, subtract(p, a)), n) >= 0 && dot(cross(bc, subtract(p, b)), n) >= 0 && dot(cross(ca, subtract(p, c)), n) >= 0;
    };
    const double start = dot(subtract(from, a), n), end = dot(subtract(to, a), n);
    if (inside(from))
        best = std::min(best, std::abs(start) / std::sqrt(n2));
    if (inside(to))
        best = std::min(best, std::abs(end) / std::sqrt(n2));
    if (start != end) {
        const double t = start / (start - end);
        if (t >= 0 && t <= 1 && inside(interpolate(from, to, t)))
            return 0;
    }
    return best;
}
std::shared_ptr<Index> index(std::shared_ptr<const Mesh> mesh, const std::atomic_bool& cancelled)
{
    auto out = std::make_shared<Index>();
    out->mesh = std::move(mesh);
    // Consecutive grid cells stay spatially local. Index blocks rather than
    // six million separate triangles, retaining exact tests at the leaves.
    constexpr size_t blockIndices = 96;
    for (size_t begin = 0; begin < out->mesh->indices.size(); begin += blockIndices) {
        check(cancelled);
        const auto end = std::min(begin + blockIndices, out->mesh->indices.size());
        auto bounds = render::emptyBounds();
        for (size_t i = begin; i < end; ++i)
            render::expand(bounds, vertex(*out->mesh, out->mesh->indices[i]));
        out->blocks.push_back({bounds, begin, end});
    }
    auto build = [&](auto&& self, size_t begin, size_t end) -> size_t {
        check(cancelled);
        Index::Node node{render::emptyBounds(), begin, end};
        for (size_t i = begin; i < end; ++i) {
            render::expand(node.bounds, out->blocks[i].bounds.lower);
            render::expand(node.bounds, out->blocks[i].bounds.upper);
        }
        const auto id = out->nodes.size();
        out->nodes.push_back(node);
        if (end - begin > 4) {
            int axis = 0;
            for (int k = 1; k < 3; ++k)
                if (node.bounds.upper[k] - node.bounds.lower[k] > node.bounds.upper[axis] - node.bounds.lower[axis])
                    axis = k;
            size_t middle = begin + (end - begin) / 2;
            std::nth_element(out->blocks.begin() + begin, out->blocks.begin() + middle, out->blocks.begin() + end, [axis](const auto& a, const auto& b) {
                const double ac = a.bounds.lower[axis] + a.bounds.upper[axis], bc = b.bounds.lower[axis] + b.bounds.upper[axis];
                return ac != bc ? ac < bc : a.begin < b.begin;
            });
            const auto left = self(self, begin, middle), right = self(self, middle, end);
            out->nodes[id].left = left;
            out->nodes[id].right = right;
        }
        return id;
    };
    if (!out->blocks.empty())
        build(build, 0, out->blocks.size());
    return out;
}
render::Visibility nearby(const render::Geometry& g, const Index& index, double radius, const render::Visibility& previous, const std::atomic_bool& cancelled)
{
    if (!std::isfinite(radius) || radius < 0)
        throw std::invalid_argument("Invalid surface proximity distance");
    std::vector<uint8_t> keep(g.scene->fibers.size(), 0);
    if (index.nodes.empty())
        return render::visibility(g, std::move(keep));
    const auto& mesh = *index.mesh;
    const auto bounds = padded(mesh.bounds, radius);
    for (const auto& chunk : g.chunks) {
        check(cancelled);
        if (keep[chunk.fiber] || !previous.includes(chunk.fiber) || !overlaps(bounds, chunk.bounds))
            continue;
        for (size_t i = chunk.first; i < chunk.end; ++i) {
            render::Segment segment;
            if (!render::segment(g, chunk, i, segment))
                continue;
            auto search = render::emptyBounds();
            render::expand(search, segment.a);
            render::expand(search, segment.b);
            search = padded(search, radius);
            auto visit = [&](auto&& self, size_t ni) -> bool {
                const auto& node = index.nodes[ni];
                if (!overlaps(node.bounds, search))
                    return false;
                check(cancelled);
                if (node.left != node.right)
                    return self(self, node.left) || self(self, node.right);
                for (size_t bi = node.begin; bi < node.end; ++bi) {
                    const auto& block = index.blocks[bi];
                    if (!overlaps(block.bounds, search))
                        continue;
                    for (size_t ti = block.begin; ti < block.end; ti += 3) {
                        const auto a = vertex(mesh, mesh.indices[ti]), b = vertex(mesh, mesh.indices[ti + 1]),
                                   c = vertex(mesh, mesh.indices[ti + 2]);
                        if (segmentTriangleDistance(segment.a, segment.b, a, b, c) <= radius)
                            return true;
                    }
                }
                return false;
            };
            if (visit(visit, 0)) {
                keep[chunk.fiber] = 1;
                break;
            }
        }
    }
    check(cancelled);
    return render::visibility(g, std::move(keep));
}
}  // namespace vc3d::fibers3d::surface
