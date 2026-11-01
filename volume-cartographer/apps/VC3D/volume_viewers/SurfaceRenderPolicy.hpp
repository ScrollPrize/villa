#pragma once

#include <algorithm>

namespace vc::render {

// Plane views share the chunk scheduler with flattened-surface tile fills.
// Keep their coarse demand bounded in a tile-fill-preferring workspace even
// though planes never have a SurfaceCache. Direct surface fallback still needs
// coarse demand when its cache is disabled, unavailable, or out of band.
// Line annotation viewers retain unrestricted fallback demand in either case.
inline constexpr int surfaceQueuedFallbackLevels(
    int requestedLevels, bool preferSurfaceTileFills, bool surfaceCacheUsable,
    bool planeView = false, bool annotationView = false)
{
    return !annotationView && preferSurfaceTileFills && (planeView || surfaceCacheUsable)
        ? std::min(requestedLevels, 1)
        : requestedLevels;
}

} // namespace vc::render
