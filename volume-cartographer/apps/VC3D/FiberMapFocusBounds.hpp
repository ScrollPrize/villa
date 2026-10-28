#pragma once

#include "FiberNetworkLayout.hpp"
#include "vc/core/util/Rect3D.hpp"
#include <QPainterPath>
#include <algorithm>
#include <cmath>

namespace vc3d::fiber_map {

struct FocusPaths {
    QPainterPath inside;
    QPainterPath outside;
};

// Clip in the original 3D frame, then carry the segment fractions into the
// flattened display. A segment can cross the box with both endpoints outside.
inline FocusPaths focusRunPaths(const PlacedFiber& fiber, std::size_t runIndex,
                               const Rect3D& bounds)
{
    FocusPaths result;
    const auto points = displayRunPoints(fiber, runIndex);
    const auto world = displayRunWorldPoints(fiber, runIndex);
    const auto append = [](QPainterPath& path, const QPointF& a, const QPointF& b) {
        if (path.isEmpty() || path.currentPosition() != a) path.moveTo(a);
        path.lineTo(b);
    };
    for (std::size_t i = 1; i < points.size(); ++i) {
        double enter = 0.0, leave = 1.0;
        bool intersects = world.size() == points.size();
        for (int axis = 0; axis < 3 && intersects; ++axis) {
            const double a = world[i - 1][axis], b = world[i][axis];
            if (!std::isfinite(a) || !std::isfinite(b)) {
                intersects = false;
            } else if (a == b) {
                intersects = a >= bounds.low[axis] && a <= bounds.high[axis];
            } else {
                const double t0 = (bounds.low[axis] - a) / (b - a);
                const double t1 = (bounds.high[axis] - a) / (b - a);
                enter = std::max(enter, std::min(t0, t1));
                leave = std::min(leave, std::max(t0, t1));
                intersects = enter <= leave;
            }
        }
        const QPointF a = points[i - 1], b = points[i];
        if (!intersects) {
            append(result.outside, a, b);
            continue;
        }
        const QPointF first = a + enter * (b - a), last = a + leave * (b - a);
        if (enter > 0.0) append(result.outside, a, first);
        if (leave > enter) append(result.inside, first, last);
        if (leave < 1.0) append(result.outside, last, b);
    }
    return result;
}

} // namespace vc3d::fiber_map
