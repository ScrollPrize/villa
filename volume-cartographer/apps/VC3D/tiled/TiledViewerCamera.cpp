#include "TiledViewerCamera.hpp"
#include "TileScene.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include <algorithm>
#include <stdexcept>

void TiledViewerCamera::recalcPyramidLevel(int numScales)
{
    if (numScales <= 0) {
        dsScaleIdx = 0;
        dsScaleIdxFine = 0;
        dsBlendFactor = 0.0f;
        dsScale = 1.0f;
        return;
    }

    const float maxScale = (numScales >= 2) ? 0.5f : 1.0f;
    const float minScale = std::pow(2.0f, 1.0f - numScales);

    if (scale >= maxScale) {
        dsScaleIdx = 0;
    } else if (scale < minScale) {
        dsScaleIdx = numScales - 1;
    } else {
        dsScaleIdx = static_cast<int>(std::round(-std::log2(scale)));
    }

    if (downscaleOverride > 0) {
        dsScaleIdx += downscaleOverride;
        dsScaleIdx = std::min(dsScaleIdx, numScales - 1);
    }

    dsScale = std::pow(2.0f, -dsScaleIdx);

    // Compute blend factor between current level and an adjacent level.
    // The camera scale falls between two pyramid level scales; we blend
    // between them for a smooth transition.
    //
    // dsScaleIdxFine = the adjacent level to blend toward.
    // dsBlendFactor  = 0.0 means fully current level, 1.0 means fully adjacent.
    //
    // When scale > dsScale (finer than current), blend toward the finer level.
    // When scale < dsScale (coarser than current), blend toward the coarser level.

    if (scale > dsScale && dsScaleIdx > 0) {
        // Blend toward finer level
        dsScaleIdxFine = dsScaleIdx - 1;
        float fineScale = std::pow(2.0f, -dsScaleIdxFine);
        dsBlendFactor = std::clamp(
            (scale - dsScale) / (fineScale - dsScale), 0.0f, 1.0f);
    } else if (scale < dsScale && dsScaleIdx < numScales - 1) {
        // Blend toward coarser level
        dsScaleIdxFine = dsScaleIdx + 1;
        float coarseScale = std::pow(2.0f, -dsScaleIdxFine);
        dsBlendFactor = std::clamp(
            (scale - dsScale) / (coarseScale - dsScale), 0.0f, 1.0f);
    } else {
        // At exact level scale or at boundary — no blending
        dsScaleIdxFine = dsScaleIdx;
        dsBlendFactor = 0.0f;
    }
}

// Predefined zoom stops where TILE_PX/scale is a "nice" number, eliminating
// sub-pixel tile-boundary seams.  Covers MIN_SCALE..MAX_SCALE roughly
// 12 steps per octave (≈6% apart).
static constexpr float kZoomStops[] = {
    // Extended range: 0.01 .. 10.0
    0.01f,
    0.015625f, // 32768
    0.03125f,  // 16384
    0.0625f,   // 8192
    0.125f,    // 4096
    0.1875f,   // 2730.7  (close to 3/16)
    0.25f,     // 2048
    0.3125f,   // 1638.4  (5/16)
    0.375f,    // 1365.3  (3/8)
    0.4375f,   // 1170.3  (7/16)
    0.5f,      // 1024
    0.5625f,   // 910.2   (9/16)
    0.625f,    // 819.2   (5/8)
    0.75f,     // 682.7   (3/4)
    0.875f,    // 585.1   (7/8)
    1.0f,      // 512
    1.25f,     // 409.6   (5/4)
    1.5f,      // 341.3   (3/2)
    1.75f,     // 292.6   (7/4)
    2.0f,      // 256
    2.5f,      // 204.8   (5/2)
    3.0f,      // 170.7
    3.5f,      // 146.3
    4.0f,      // 128
    5.0f,
    6.0f,
    8.0f,
    10.0f,
};
static constexpr int kNumStops = sizeof(kZoomStops) / sizeof(kZoomStops[0]);

// Find the index of the closest zoom stop to s.
static int closestStopIndex(float s)
{
    int lo = 0, hi = kNumStops - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (kZoomStops[mid] < s)
            lo = mid + 1;
        else
            hi = mid;
    }
    if (lo > 0) {
        float dLo = s - kZoomStops[lo - 1];
        float dHi = kZoomStops[lo] - s;
        if (dLo < dHi) --lo;
    }
    return lo;
}

float TiledViewerCamera::roundScale(float s)
{
    s = std::clamp(s, MIN_SCALE, MAX_SCALE);
    return kZoomStops[closestStopIndex(s)];
}

float TiledViewerCamera::stepScale(float current, int steps)
{
    // Continuous zoom: 1% per step at all zoom levels.
    float result = current * std::pow(1.01f, static_cast<float>(steps));
    return std::clamp(result, MIN_SCALE, MAX_SCALE);
}

// ---------------------------------------------------------------------------
// Coordinate-transform free functions
// ---------------------------------------------------------------------------

QPointF tiledVolumeToScene(Surface* surf, TileScene* tileScene,
                           SurfacePatchIndex* patchIndex,
                           const cv::Vec3f& volPoint)
{
    if (!surf || !tileScene) return QPointF();

    if (auto* plane = dynamic_cast<PlaneSurface*>(surf)) {
        cv::Vec3f surfPos = plane->project(volPoint, 1.0, 1.0);
        return tileScene->surfaceToScene(surfPos[0], surfPos[1]);
    }

    if (auto* quad = dynamic_cast<QuadSurface*>(surf)) {
        cv::Vec3f ptr(0, 0, 0);
        surf->pointTo(ptr, volPoint, 4.0, 100, patchIndex);
        cv::Vec3f loc = surf->loc(ptr);
        return tileScene->surfaceToScene(loc[0], loc[1]);
    }

    return QPointF();
}

bool tiledSceneToVolume(Surface* surf, TileScene* tileScene,
                        const QPointF& scenePos,
                        cv::Vec3f& outPos, cv::Vec3f& outNormal)
{
    if (!surf || !tileScene) {
        outPos = cv::Vec3f(0, 0, 0);
        outNormal = cv::Vec3f(0, 0, 1);
        return false;
    }

    try {
        cv::Vec2f surfParam = tileScene->sceneToSurface(scenePos);
        cv::Vec3f surfLoc = {surfParam[0], surfParam[1], 0};
        cv::Vec3f ptr(0, 0, 0);
        outNormal = surf->normal(ptr, surfLoc);
        outPos = surf->coord(ptr, surfLoc);
    } catch (const std::exception&) {
        return false;
    }
    return true;
}
