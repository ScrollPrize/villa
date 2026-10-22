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
    if (u != other.u || v != other.v || n != other.n)
        return false;
    const auto offset = project(other.origin);
    for (int i = 0; i < 3; ++i)
        if (!(other.local.lower[i] + offset[i] >= local.lower[i] && other.local.upper[i] + offset[i] <= local.upper[i]))
            return false;
    return true;
}

bool clipSegment(const Point& a, const Point& b, const Bounds& box, double& lo, double& hi)
{
    return vc::geometry::clipSegment(a, b,
        {box.lower[0], box.lower[1], box.lower[2], box.upper[0], box.upper[1], box.upper[2]}, lo, hi);
}
}
