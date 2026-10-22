#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include "vc/fiber_tracer/detail/PortableTraceMath.hpp"

namespace vc::fiber_tracer {

// Trace-local committed state, never speculative beam state.
struct FiberWidthMean {
    double sum = 0;
    uint64_t count = 0;
    void add(double width) {
        if (!std::isfinite(width) || width <= 0)
            throw std::runtime_error("committed trace point has no positive predicted width");
        sum += width;
        ++count;
    }
    [[nodiscard]] double mean() const { return count ? sum / count : 0; }
};

inline float fiberWidthPresenceScale(double width, double mean, double tolerance, bool cosine = false) {
    if (tolerance == 0) return 1;
    if (!std::isfinite(mean) || mean <= 0 || !std::isfinite(tolerance) || tolerance <= 0)
        throw std::invalid_argument("invalid width reference/tolerance");
    return portable::widthPresenceScale(width, mean, tolerance, cosine);
}
}
