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
}

// Predefined zoom stops where 256/scale is a "nice" number, eliminating
// sub-pixel tile-boundary seams.  Covers MIN_SCALE..MAX_SCALE roughly
// 12 steps per octave (≈6% apart).
static constexpr float kZoomStops[] = {
    // 256 / scale → integer or near-integer tile world size
    0.03125f, // 8192
    0.0625f,  // 4096
    0.125f,   // 2048
    0.1875f,  // 1365.3  (close to 3/16)
    0.25f,    // 1024
    0.3125f,  // 819.2   (5/16)
    0.375f,   // 682.7   (3/8)
    0.4375f,  // 585.1   (7/16)
    0.5f,     // 512
    0.5625f,  // 455.1   (9/16)
    0.625f,   // 409.6   (5/8)
    0.75f,    // 341.3   (3/4)
    0.875f,   // 292.6   (7/8)
    1.0f,     // 256
    1.25f,    // 204.8   (5/4)
    1.5f,     // 170.7   (3/2)
    1.75f,    // 146.3   (7/4)
    2.0f,     // 128
    2.5f,     // 102.4   (5/2)
    3.0f,     // 85.3
    3.5f,     // 73.1
    4.0f,     // 64
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
    int idx = closestStopIndex(std::clamp(current, MIN_SCALE, MAX_SCALE));
    idx = std::clamp(idx + steps, 0, kNumStops - 1);
    return kZoomStops[idx];
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
