#pragma once

#include "Fiber3DGeometry.hpp"
#include <QMatrix4x4>
#include <atomic>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <tuple>

namespace vc3d::fibers3d::render
{
// Display-only approximations. Level zero and picking use the original clipped
// segments. Higher levels never connect different runs, styles or invalid spans.
inline constexpr std::array<double, 7> tolerances{0, .125, .5, 2, 8, 32, 128};
struct Vertex {
    float a[3], b[3];
    float length, family, style, fiber;
    float arc;
    float direction[3];
};
struct Range {
    size_t first = 0, count = 0;
};
inline Bounds emptyBounds()
{
    const double inf = std::numeric_limits<double>::infinity();
    return {{inf, inf, inf}, {-inf, -inf, -inf}};
}
inline void expand(Bounds& bounds, const Point& p)
{
    for (int k = 0; k < 3; ++k) {
        bounds.lower[k] = std::min(bounds.lower[k], p[k]);
        bounds.upper[k] = std::max(bounds.upper[k], p[k]);
    }
}
inline double distance(const Point& a, const Point& b)
{
    return std::hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
}
struct Chunk {
    Bounds bounds = emptyBounds();
    size_t fiber = 0, run = 0, first = 0, end = 0;
    double arc = 0;
    std::array<Range, tolerances.size()> levels;
};
struct Node {
    Bounds bounds = emptyBounds();
    size_t begin = 0, end = 0, left = 0, right = 0;
    bool leaf() const { return left == right; }
};
struct Geometry {
    std::shared_ptr<const Scene> scene;
    Bounds bounds = emptyBounds();
    Point origin{};
    std::vector<Chunk> chunks;
    std::vector<Node> nodes;
    std::array<std::vector<Vertex>, tolerances.size()> levels;
    std::vector<Vertex> markers;
    std::vector<Bounds> fiberBounds;
    std::vector<std::vector<size_t>> fiberChunks;
    std::pair<double, double> lengths;
};
struct Segment {
    Point a, b;
    size_t fiber;
    int64_t originalIndex;
    double begin, end;
};
inline bool segment(const Geometry& g, const Chunk& c, size_t index, Segment& out)
{
    const auto& run = g.scene->fibers[c.fiber].runs[c.run];
    const auto& a = run.points[index];
    const auto& b = run.points[index + 1];
    double lo = 0, hi = 1;
    if (g.scene->clipBounds && !clipSegment(a, b, *g.scene->clipBounds, lo, hi))
        return false;
    out = {interpolate(a, b, lo), interpolate(a, b, hi), c.fiber, run.firstPoint + int64_t(index), lo, hi};
    return true;
}
inline Vertex vertex(const Geometry& g, const Point& a, const Point& b, size_t fi, float style, double arc)
{
    const auto& f = g.scene->fibers[fi];
    const auto t = axialDirection(a, b);
    return {{float(a[0] - g.origin[0]), float(a[1] - g.origin[1]), float(a[2] - g.origin[2])}, {float(b[0] - g.origin[0]), float(b[1] - g.origin[1]), float(b[2] - g.origin[2])}, std::isfinite(f.lengthMm) && f.lengthMm >= 0 ? float(f.lengthMm) : -1.f, f.family == 'H' ? 1.f : (f.family == 'V' ? 2.f : 0.f), style, float(fi), float(arc), {float(t[0]), float(t[1]), float(t[2])}};
}
inline double distanceToSegment(const Point& p, const Point& a, const Point& b)
{
    double ab2 = 0, dot = 0;
    for (int k = 0; k < 3; ++k) {
        ab2 += (b[k] - a[k]) * (b[k] - a[k]);
        dot += (p[k] - a[k]) * (b[k] - a[k]);
    }
    return distance(p, interpolate(a, b, ab2 > 0 ? std::clamp(dot / ab2, 0., 1.) : 0.));
}
// RDP using distance to the finite chord (not its infinite supporting line).
// This also preserves hairpins/backtracking. Original endpoints are retained.
inline void simplify(const std::vector<Point>& points, size_t a, size_t b, double tolerance, std::vector<size_t>& out)
{
    double error = 0;
    size_t split = a;
    for (size_t i = a + 1; i < b; ++i) {
        const double d = distanceToSegment(points[i], points[a], points[b]);
        if (d > error) {
            error = d;
            split = i;
        }
    }
    if (b > a + 1 && error > tolerance) {
        simplify(points, a, split, tolerance, out);
        simplify(points, split, b, tolerance, out);
    } else {
        out.push_back(b);
    }
}
inline size_t buildNode(Geometry& g, size_t begin, size_t end)
{
    const size_t index = g.nodes.size();
    Node node;
    node.begin = begin;
    node.end = end;
    for (size_t i = begin; i < end; ++i) {
        expand(node.bounds, g.chunks[i].bounds.lower);
        expand(node.bounds, g.chunks[i].bounds.upper);
    }
    g.nodes.push_back(node);
    if (end - begin > 16) {
        int axis = 0;
        for (int k = 1; k < 3; ++k)
            if (node.bounds.upper[k] - node.bounds.lower[k] > node.bounds.upper[axis] - node.bounds.lower[axis])
                axis = k;
        const size_t middle = begin + (end - begin) / 2;
        std::nth_element(g.chunks.begin() + begin, g.chunks.begin() + middle, g.chunks.begin() + end, [axis](const Chunk& a, const Chunk& b) {
            const double ac = a.bounds.lower[axis] + a.bounds.upper[axis], bc = b.bounds.lower[axis] + b.bounds.upper[axis];
            if (ac != bc)
                return ac < bc;
            return std::tie(a.fiber, a.run, a.first) < std::tie(b.fiber, b.run, b.first);
        });
        const auto left = buildNode(g, begin, middle), right = buildNode(g, middle, end);
        g.nodes[index].left = left;
        g.nodes[index].right = right;
    }
    return index;
}
inline std::shared_ptr<Geometry> build(std::shared_ptr<const Scene> scene, const std::atomic_bool* cancelled = nullptr)
{
    auto check = [&] {
        if (cancelled && cancelled->load(std::memory_order_relaxed))
            throw std::runtime_error("Cancelled");
    };
    auto g = std::make_shared<Geometry>();
    g->scene = std::move(scene);
    g->lengths = lengthRange(*g->scene, {});
    g->fiberBounds.resize(g->scene->fibers.size(), emptyBounds());
    g->fiberChunks.resize(g->scene->fibers.size());
    for (size_t fi = 0; fi < g->scene->fibers.size(); ++fi) {
        check();
        const auto& fiber = g->scene->fibers[fi];
        for (size_t ri = 0; ri < fiber.runs.size(); ++ri) {
            const auto& run = fiber.runs[ri];
            double arc = 0, previousHi = -1;
            int previousStyle = -1;
            bool active = false;
            for (size_t i = 0; i + 1 < run.points.size(); ++i) {
                if (i % 4096 == 0)
                    check();
                const auto& a = run.points[i];
                const auto& b = run.points[i + 1];
                if (!finite(a) || !finite(b)) {
                    arc = 0;
                    active = false;
                    continue;
                }
                const double len = distance(a, b);
                double lo = 0, hi = 1;
                const bool visible = !g->scene->clipBounds || clipSegment(a, b, *g->scene->clipBounds, lo, hi);
                if (visible && len > 0 && hi > lo) {
                    const int style = i < run.styles.size() ? int(run.styles[i]) : 0;
                    if (!active || style != previousStyle || lo != 0 || previousHi != 1 || g->chunks.back().end - g->chunks.back().first == 128) {
                        Chunk c;
                        c.fiber = fi;
                        c.run = ri;
                        c.first = i;
                        c.arc = arc;
                        g->chunks.push_back(c);
                    }
                    auto& c = g->chunks.back();
                    c.end = i + 1;
                    expand(c.bounds, interpolate(a, b, lo));
                    expand(c.bounds, interpolate(a, b, hi));
                    previousHi = hi;
                    previousStyle = style;
                    active = true;
                } else {
                    active = false;
                }
                arc += len;
            }
        }
    }
    if (g->chunks.empty())
        return g;
    buildNode(*g, 0, g->chunks.size());
    check();
    g->bounds = g->nodes.front().bounds;
    for (int k = 0; k < 3; ++k)
        g->origin[k] = (g->bounds.lower[k] + g->bounds.upper[k]) * .5;
    size_t exactCount = 0;
    for (const auto& c : g->chunks)
        exactCount += c.end - c.first;
    g->levels[0].reserve(exactCount);
    std::vector<Point> points;
    std::vector<double> arcs;
    std::vector<size_t> kept;
    points.reserve(129);
    arcs.reserve(129);
    kept.reserve(129);
    for (size_t ci = 0; ci < g->chunks.size(); ++ci) {
        check();
        auto& c = g->chunks[ci];
        g->fiberChunks[c.fiber].push_back(ci);
        expand(g->fiberBounds[c.fiber], c.bounds.lower);
        expand(g->fiberBounds[c.fiber], c.bounds.upper);
        points.clear();
        arcs.clear();
        const auto& run = g->scene->fibers[c.fiber].runs[c.run];
        const float style = c.first < run.styles.size() ? float(run.styles[c.first]) : 0;
        double arc = c.arc;
        for (size_t i = c.first; i < c.end; ++i) {
            Segment s;
            segment(*g, c, i, s);
            const double len = distance(run.points[i], run.points[i + 1]);
            if (points.empty()) {
                points.push_back(s.a);
                arcs.push_back(arc + len * s.begin);
            }
            points.push_back(s.b);
            arcs.push_back(arc + len * s.end);
            arc += len;
        }
        for (size_t level = 0; level < tolerances.size(); ++level) {
            kept.clear();
            kept.push_back(0);
            if (level == 0)
                for (size_t i = 1; i < points.size(); ++i)
                    kept.push_back(i);
            else
                simplify(points, 0, points.size() - 1, tolerances[level], kept);
            auto& vertices = g->levels[level];
            c.levels[level] = {vertices.size(), kept.size() - 1};
            for (size_t i = 1; i < kept.size(); ++i)
                vertices.push_back(vertex(*g, points[kept[i - 1]], points[kept[i]], c.fiber, style, arcs[kept[i - 1]]));
        }
    }
    size_t markerCount = 0;
    for (const auto& f : g->scene->fibers)
        markerCount += f.markers.size();
    g->markers.reserve(markerCount);
    for (size_t fi = 0; fi < g->scene->fibers.size(); ++fi) {
        check();
        for (const auto& marker : g->scene->fibers[fi].markers) {
            double lo, hi;
            if (finite(marker.point) && (!g->scene->clipBounds || clipSegment(marker.point, marker.point, *g->scene->clipBounds, lo, hi)))
                g->markers.push_back(vertex(*g, marker.point, marker.point, fi, 10.f + float(marker.kind), 0));
        }
    }
    return g;
}

// Conservative clip-space box tests, expanded for stroke width (and picking).
// Projection derivatives bound screen error everywhere inside the box, including
// perspective. Near/eye-plane crossings force exact detail; never divide by zero.
struct View {
    std::array<std::array<double, 4>, 4> m{};
    Point origin{};
    double width = 1, height = 1, margin = 0;
    View(const QMatrix4x4& matrix, Point origin_, double w, double h, double padding)
        : origin(origin_), width(w), height(h), margin(padding)
    {
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                m[r][c] = matrix(r, c);
    }
    std::pair<double, double> extent(const Bounds& b, const std::array<double, 4>& row) const
    {
        double center = row[3], radius = 0;
        for (int k = 0; k < 3; ++k) {
            center += row[k] * ((b.lower[k] + b.upper[k]) * .5 - origin[k]);
            radius += std::abs(row[k]) * (b.upper[k] - b.lower[k]) * .5;
        }
        // Include float vertex/projection roundoff in the GPU. A CPU cull must
        // not reject a stroke touching a clip plane after float conversion.
        radius += 32 * std::numeric_limits<float>::epsilon() * (1 + std::abs(center) + radius + std::abs(row[3]));
        return {center - radius, center + radius};
    }
    // -1 outside, 0 intersects, 1 inside. x/y can describe a picking rectangle.
    int classify(const Bounds& b, double xmin = -1, double xmax = 1, double ymin = -1, double ymax = 1) const
    {
        const double limits[3][2] = {{xmin - 2 * margin / width, xmax + 2 * margin / width}, {ymin - 2 * margin / height, ymax + 2 * margin / height}, {-1, 1}};
        int result = 1;
        for (int axis = 0; axis < 3; ++axis)
            for (int side = 0; side < 2; ++side) {
                std::array<double, 4> plane;
                for (int k = 0; k < 4; ++k)
                    plane[k] = side ? limits[axis][side] * m[3][k] - m[axis][k] : m[axis][k] - limits[axis][side] * m[3][k];
                const auto [lo, hi] = extent(b, plane);
                if (hi < 0)
                    return -1;
                if (lo < 0)
                    result = 0;
            }
        return result;
    }
    double pixelsPerUnit(const Bounds& b) const
    {
        const double wmin = extent(b, m[3]).first;
        std::array<double, 4> near;
        for (int i = 0; i < 4; ++i)
            near[i] = m[2][i] + m[3][i];
        if (wmin <= 0 || extent(b, near).first <= 0)
            return std::numeric_limits<double>::infinity();
        const double wn = std::hypot(m[3][0], m[3][1], m[3][2]);
        double squared = 0;
        for (int axis = 0; axis < 2; ++axis) {
            const auto [lo, hi] = extent(b, m[axis]);
            const double rate = (axis ? height : width) * .5 *
                                (std::hypot(m[axis][0], m[axis][1], m[axis][2]) / wmin + std::max(std::abs(lo), std::abs(hi)) * wn / (wmin * wmin));
            squared += rate * rate;
        }
        return std::sqrt(squared);
    }
};
struct Draw {
    size_t level;
    Range range;
    size_t begin, end;
};
inline std::vector<Draw> draws(const Geometry& g, const View& view, double pixelError)
{
    std::vector<Draw> result;
    if (g.nodes.empty())
        return result;
    auto append = [&](size_t begin, size_t end, const Bounds& bounds) {
        const double scale = view.pixelsPerUnit(bounds);
        size_t level = 0;
        for (size_t i = 1; i < tolerances.size(); ++i)
            if (tolerances[i] * scale <= pixelError)
                level = i;
        const auto first = g.chunks[begin].levels[level], last = g.chunks[end - 1].levels[level];
        Range range{first.first, last.first + last.count - first.first};
        if (!result.empty() && result.back().level == level && result.back().range.first + result.back().range.count == range.first) {
            result.back().range.count += range.count;
            result.back().end = end;
        } else
            result.push_back({level, range, begin, end});
    };
    auto visit = [&](auto&& self, size_t i) -> void {
        const auto& n = g.nodes[i];
        const int state = view.classify(n.bounds);
        if (state < 0)
            return;
        if (state == 1) {
            append(n.begin, n.end, n.bounds);
            return;
        }
        if (n.leaf()) {
            for (size_t c = n.begin; c < n.end; ++c)
                if (view.classify(g.chunks[c].bounds) >= 0)
                    append(c, c + 1, g.chunks[c].bounds);
        } else {
            self(self, n.left);
            self(self, n.right);
        }
    };
    visit(visit, 0);
    return result;
}
inline std::vector<size_t> pickChunks(const Geometry& g, const View& view, double x, double y)
{
    std::vector<size_t> result;
    if (g.nodes.empty())
        return result;
    auto visit = [&](auto&& self, size_t i) -> void {
        const auto& n = g.nodes[i];
        if (view.classify(n.bounds, x, x, y, y) < 0)
            return;
        if (n.leaf()) {
            for (size_t c = n.begin; c < n.end; ++c)
                if (view.classify(g.chunks[c].bounds, x, x, y, y) >= 0)
                    result.push_back(c);
        } else {
            self(self, n.left);
            self(self, n.right);
        }
    };
    visit(visit, 0);
    // Keep the old source traversal order for equal-distance/depth pick ties.
    std::sort(result.begin(), result.end(), [&](size_t a, size_t b) {
        const auto& ca = g.chunks[a];
        const auto& cb = g.chunks[b];
        return std::tie(ca.fiber, ca.run, ca.first) < std::tie(cb.fiber, cb.run, cb.first);
    });
    return result;
}
}  // namespace vc3d::fibers3d::render
