#pragma once

#include <algorithm>

namespace vc::render {

// A workspace preference must not limit direct sampling when its surface
// cache is disabled, unavailable, or cannot cover the frame's normal band.
inline constexpr int surfaceQueuedFallbackLevels(
    int requestedLevels, bool preferSurfaceTileFills, bool surfaceCacheUsable)
{
    return preferSurfaceTileFills && surfaceCacheUsable
        ? std::min(requestedLevels, 1)
        : requestedLevels;
}

} // namespace vc::render
