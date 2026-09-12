#pragma once

#include <cstdint>

#include <opencv2/core.hpp>

class Surface;

// The two halves of a viewer's volume-point-to-scene-point mapping.
//
// Projecting a volume point onto the displayed surface is a nearest-point
// search over that surface; turning the result into a scene point is a handful
// of multiply-adds. The halves also differ in what they depend on: the search
// depends on the surface, the patch index and the displayed depth band, while
// the scene mapping depends on the camera. Splitting them lets an overlay that
// rebuilds on every pan and zoom cache the expensive half.
//
// These live in their own header because VolumeViewerBase.hpp and the overlay
// controller base include each other.

// Surface-space projection of a volume point: surface coordinates plus the
// signed offset along the surface normal. `applyVolumetricW` records whether
// the normal offset participates in the scene mapping (it does for quad
// surfaces under the volumetric camera, never for plane surfaces).
struct SurfaceProjection {
    float u{0.0f};
    float v{0.0f};
    float w{0.0f};
    bool applyVolumetricW{false};
};

// The inputs, other than the point itself, that a viewer's projection depends
// on. A cache of projections stays valid exactly as long as this compares
// equal, so callers do not have to enumerate those inputs themselves and
// cannot drift out of sync with the viewer.
struct SurfaceProjectionContext {
    const Surface* surface{nullptr};
    std::uint64_t patchIndexGeneration{0};
    float depthLo{0.0f};
    float depthHi{0.0f};
    // A plane can be moved while keeping the same Surface pointer, so its
    // frame has to be part of the identity as well.
    cv::Vec3f planeOrigin{0.0f, 0.0f, 0.0f};
    cv::Vec3f planeBasisX{0.0f, 0.0f, 0.0f};
    cv::Vec3f planeBasisY{0.0f, 0.0f, 0.0f};

    bool operator==(const SurfaceProjectionContext& other) const
    {
        return surface == other.surface &&
               patchIndexGeneration == other.patchIndexGeneration &&
               depthLo == other.depthLo &&
               depthHi == other.depthHi &&
               planeOrigin == other.planeOrigin &&
               planeBasisX == other.planeBasisX &&
               planeBasisY == other.planeBasisY;
    }
};
