#pragma once

#include <string>

namespace vc3d::annotation {

// The inputs the cached scroll umbilicus was built from: it is read in the
// volume's voxels and its per-slice centres sized to the volume, and the
// volume-centre fallback depends on the volume itself.
struct UmbilicusCacheInputs {
    // The active volume, deliberately conservative: any volume switch reloads
    // rather than proving the previous geometry equivalent.
    std::string volumeId;
    // The stored umbilicus (uuid@revision), or why there is none.
    std::string dependencyToken;
};

// Whether a cached umbilicus may be reused for `current`, or must be
// re-resolved. Extracted from the branch it drives so it is asserted rather
// than read: the caller sits behind a package, a loaded volume and pane state.
inline bool umbilicusReloadNeeded(bool cachedLoadAttempted,
                                  const UmbilicusCacheInputs& cached,
                                  const UmbilicusCacheInputs& current)
{
    if (!cachedLoadAttempted) {
        return true;
    }
    if (cached.volumeId != current.volumeId) {
        return true;
    }
    return cached.dependencyToken != current.dependencyToken;
}

// One pane, as the stale-view refresh sees it.
struct GeneratedViewsPaneState {
    bool hasSession = false;
    // Sessions that suppress ordinary generated views (intersection sides)
    // are rebuilt through the inspection, never through the pane walk.
    bool suppressesGeneratedViews = false;
    // Nothing materialized means nothing stale on screen — and the builder
    // rejects an empty model, which during initial tracing turned a successful
    // attach into a modal complaint about views the user never asked for.
    bool hasGeneratedSurfaces = false;
    bool hasLinePoints = false;
    // The controller's orientation epoch when the views were built.
    int orientationEpoch = -1;
};

// Whether the refresh pass rebuilds this pane's generated views.
inline bool paneNeedsOrientationRefresh(const GeneratedViewsPaneState& pane,
                                        int currentEpoch)
{
    if (!pane.hasSession || pane.suppressesGeneratedViews) {
        return false;
    }
    if (!pane.hasGeneratedSurfaces || !pane.hasLinePoints) {
        return false;
    }
    return pane.orientationEpoch != currentEpoch;
}

} // namespace vc3d::annotation
