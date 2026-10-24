#pragma once
#include <algorithm>
#include <array>
#include <cmath>

namespace vc::geometry {
// Double-precision slab clipping. Coordinates remain untouched; first/last
// describe the closed parameter interval inside minXYZ,maxXYZ bounds.
inline bool clipSegment(const std::array<double, 3>& a, const std::array<double, 3>& b,
                        const std::array<double, 6>& bounds, double& first, double& last)
{
    first = 0;
    last = 1;
    for (size_t i = 0; i < 3; ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return false;
        const double d = b[i] - a[i];
        if (d == 0) {
            if (a[i] < bounds[i] || a[i] > bounds[i + 3]) return false;
        } else {
            double lo = (bounds[i] - a[i]) / d, hi = (bounds[i + 3] - a[i]) / d;
            if (lo > hi) std::swap(lo, hi);
            first = std::max(first, lo);
            last = std::min(last, hi);
            if (first > last) return false;
        }
    }
    return true;
}
}
