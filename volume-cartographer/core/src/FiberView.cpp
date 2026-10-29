#include "vc/core/types/FiberView.hpp"
#include <vc_object_store/geometry/SegmentClip.hpp>

#include <algorithm>
#include <cmath>

namespace vc::fibers {
Point ViewRegion::project(const Point& point) const
{
    Point out{};
    for (size_t i = 0; i < 3; ++i) {
        const double d = point[i] - origin[i];
        out[0] += d * u[i];
        out[1] += d * v[i];
        out[2] += d * n[i];
    }
    return out;
}

bool ViewRegion::contains(const ViewRegion& other) const
{
    // Both regions are boxes in their own frames. A box lies inside a convex
    // prism exactly when its eight corners do, whatever the two frames' axes.
    for (const double a : {other.local.lower[0], other.local.upper[0]})
        for (const double b : {other.local.lower[1], other.local.upper[1]})
            for (const double c : {other.local.lower[2], other.local.upper[2]}) {
                Point corner{};
                for (int i = 0; i < 3; ++i)
                    corner[i] = other.origin[i] + a * other.u[i] + b * other.v[i] + c * other.n[i];
                const auto p = project(corner);
                for (int i = 0; i < 3; ++i)
                    if (!(p[i] >= local.lower[i] && p[i] <= local.upper[i]))
                        return false;
            }
    return true;
}

std::optional<ViewRegion> viewRegionFromCorners(const Point& origin, const Point& right, const Point& down,
                                                double depth)
{
    const auto dot = [](const Point& a, const Point& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
    const auto normalized = [&](Point p) -> std::optional<Point> {
        const double n = std::sqrt(dot(p, p));
        if (!(n > 0) || !std::isfinite(n)) return std::nullopt;
        for (auto& x : p) x /= n;
        return p;
    };
    const Point dx{right[0] - origin[0], right[1] - origin[1], right[2] - origin[2]};
    const Point dy{down[0] - origin[0], down[1] - origin[1], down[2] - origin[2]};
    const auto u = normalized(dx);
    if (!u) return std::nullopt;
    const double along = dot(dy, *u);
    const Point across{dy[0] - along * (*u)[0], dy[1] - along * (*u)[1], dy[2] - along * (*u)[2]};
    // Collinear corners leave only rounding across u.
    if (!(dot(across, across) > 1e-18 * dot(dy, dy))) return std::nullopt;
    const auto v = normalized(across);
    if (!v) return std::nullopt;
    const auto n = normalized({(*u)[1] * (*v)[2] - (*u)[2] * (*v)[1], (*u)[2] * (*v)[0] - (*u)[0] * (*v)[2],
                               (*u)[0] * (*v)[1] - (*u)[1] * (*v)[0]});
    if (!n) return std::nullopt;
    ViewRegion region;
    region.origin = origin;
    region.u = *u;
    region.v = *v;
    region.n = *n;
    region.local = {{0, 0, -depth}, {0, 0, depth}};
    for (const Point& corner : {dx, dy, Point{dx[0] + dy[0], dx[1] + dy[1], dx[2] + dy[2]}})
        for (int i = 0; i < 2; ++i) {
            const double p = dot(corner, i == 0 ? region.u : region.v);
            region.local.lower[i] = std::min(region.local.lower[i], p);
            region.local.upper[i] = std::max(region.local.upper[i], p);
        }
    return region;
}

bool clipSegment(const Point& a, const Point& b, const Bounds& box, double& lo, double& hi)
{
    return vc::geometry::clipSegment(a, b,
        {box.lower[0], box.lower[1], box.lower[2], box.upper[0], box.upper[1], box.upper[2]}, lo, hi);
}
}
