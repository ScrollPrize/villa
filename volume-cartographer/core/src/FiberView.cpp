#include "vc/core/types/FiberView.hpp"
#include <vc_object_store/geometry/SegmentClip.hpp>

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

bool clipSegment(const Point& a, const Point& b, const Bounds& box, double& lo, double& hi)
{
    return vc::geometry::clipSegment(a, b,
        {box.lower[0], box.lower[1], box.lower[2], box.upper[0], box.upper[1], box.upper[2]}, lo, hi);
}
}
