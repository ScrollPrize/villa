#pragma once

#include "Fiber3DScene.hpp"
#include "vc/core/types/FiberCollection.hpp"
#include <algorithm>
#include <cmath>
#include <utility>

namespace vc3d::fibers3d
{

inline bool finite(const Point& p)
{
    return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}

inline Point interpolate(const Point& a, const Point& b, double t)
{
    return {a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t};
}

// Validate display input, then reuse the AFV reader's exact slab clipper.
// The original parameter interval is retained for source-point picking.
inline bool clipSegment(const Point& a, const Point& b, const Bounds& bounds, double& begin, double& end)
{
    begin = 0.0;
    end = 1.0;
    if (!finite(a) || !finite(b) || !finite(bounds.lower) || !finite(bounds.upper))
        return false;
    for (int axis = 0; axis < 3; ++axis)
        if (bounds.lower[axis] > bounds.upper[axis])
            return false;
    return vc::fibers::clipSegment(a, b, {bounds.lower, bounds.upper}, begin, end);
}

inline std::pair<double, double> lengthRange(const Scene& scene, const DisplaySettings& settings)
{
    if (settings.fixedLengthRange && std::isfinite(settings.lengthMinMm) && std::isfinite(settings.lengthMaxMm) &&
        settings.lengthMaxMm > settings.lengthMinMm)
        return {settings.lengthMinMm, settings.lengthMaxMm};
    double lo = std::numeric_limits<double>::infinity(), hi = -lo;
    for (const auto& fiber : scene.fibers) {
        if (!std::isfinite(fiber.lengthMm) || fiber.lengthMm < 0)
            continue;
        lo = std::min(lo, fiber.lengthMm);
        hi = std::max(hi, fiber.lengthMm);
    }
    if (!std::isfinite(lo))
        return {0.0, 1.0};
    return {lo, hi};
}

// Axial orientation: reversing a fiber must not change its colors.
inline Point axialDirection(const Point& a, const Point& b)
{
    Point t{b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    const double norm = std::hypot(t[0], t[1], t[2]);
    if (norm == 0.0 || !std::isfinite(norm))
        return {};
    for (auto& component : t)
        component = std::abs(component / norm);
    return t;
}

}  // namespace vc3d::fibers3d
