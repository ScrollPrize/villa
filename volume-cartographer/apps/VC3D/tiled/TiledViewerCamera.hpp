#pragma once

#include <cstdint>
#include <cmath>
#include <QPointF>
#include <opencv2/core.hpp>

class Surface;
class PlaneSurface;
class QuadSurface;
class TileScene;
class SurfacePatchIndex;

// Camera state for the tiled volume viewer.
// All fields are main-thread only (no locking needed).
struct TiledViewerCamera {
    // Center of view in surface parameter space
    cv::Vec3f surfacePtr{0, 0, 0};

    // Zoom level (same 0.03125..4.0 range as CVolumeViewer::_scale)
    float scale = 0.5f;

    // Normal offset (shift+wheel slice navigation)
    float zOff = 0.0f;

    // Pyramid level index (0 = full res, higher = coarser)
    int dsScaleIdx = 1;

    // Derived: 2^(-dsScaleIdx), e.g., level 2 -> 0.25
    float dsScale = 0.5f;

    // Monotonic invalidation counter, bumped on every state change
    uint64_t epoch = 0;

    // Additional override for downscale (from settings)
    int downscaleOverride = 0;

    // Zoom limits
    static constexpr float MIN_SCALE = 0.03125f;
    static constexpr float MAX_SCALE = 4.0f;

    // Bump epoch to invalidate all in-flight renders
    void invalidate() { ++epoch; }

    // Recalculate pyramid level from current scale.
    // numScales = volume->numScales()
    void recalcPyramidLevel(int numScales);

    // Snap scale to the nearest predefined zoom stop
    static float roundScale(float s);

    // Step to the next (+1) or previous (-1) zoom stop from current scale.
    // |steps| > 1 skips multiple stops.  Returns the new scale.
    static float stepScale(float current, int steps);
};

// ---------------------------------------------------------------------------
// Coordinate-transform helpers
// ---------------------------------------------------------------------------
// These live alongside TiledViewerCamera because they depend on the camera's
// tile-scene layout, but they are free functions so overlay controllers (or
// tests) can call them without a full CTiledVolumeViewer instance.

// Map a volume (world) coordinate to scene pixel coordinates.
// Returns a null QPointF if the surface is nullptr.
QPointF tiledVolumeToScene(Surface* surf, TileScene* tileScene,
                           SurfacePatchIndex* patchIndex,
                           const cv::Vec3f& volPoint);

// Map scene pixel coordinates to a volume (world) position + surface normal.
// Returns false if the conversion fails (null surface, out-of-range, etc.).
bool tiledSceneToVolume(Surface* surf, TileScene* tileScene,
                        const QPointF& scenePos,
                        cv::Vec3f& outPos, cv::Vec3f& outNormal);
