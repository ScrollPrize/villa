#include "vc/core/types/FiberView.hpp"
#include <algorithm>

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
    lo = 0;
    hi = 1;
    for (int i = 0; i < 3; ++i) {
        const double d = b[i] - a[i];
        if (d == 0) {
            if (a[i] < box.lower[i] || a[i] > box.upper[i])
                return false;
        } else {
            double x = (box.lower[i] - a[i]) / d, y = (box.upper[i] - a[i]) / d;
            if (x > y)
                std::swap(x, y);
            lo = std::max(lo, x);
            hi = std::min(hi, y);
            if (lo > hi)
                return false;
        }
    }
    return true;
}
}
