#pragma once

#include <string>

#include "AnnotationFrame.hpp"

namespace vc3d::annotation {

// The inputs the cached, already-scaled scroll umbilicus was built from. Not
// the file's contents: the cached points are multiplied by a frame-dependent
// factor and its per-slice centres sized to that frame's extent, and the
// volume-centre fallback and legacy reading depend on the volume itself — so
// all of these are the cache's key, not just where the file was found.
struct UmbilicusCacheInputs {
    // The package root the resolver searched from.
    std::string root;
    // The active volume, deliberately conservative: two volumes deriving one
    // annotation frame can still need different geometry (raw shape, a
    // registration transform), so any volume switch re-resolves rather than
    // proving the previous geometry equivalent.
    std::string volumeId;
    // Stat fingerprint over every resolver candidate and the registration
    // transform the legacy reading would consult.
    std::string dependencyToken;
    // The frame the cached points were scaled into.
    AnnotationFrame frame;
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
    if (cached.root != current.root) {
        return true;
    }
    if (cached.volumeId != current.volumeId) {
        return true;
    }
    if (cached.dependencyToken != current.dependencyToken) {
        return true;
    }
    return !sameAnnotationFrame(cached.frame, current.frame);
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
