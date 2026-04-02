#include "vc/core/render/TiledViewerCamera.hpp"
#include <algorithm>

void TiledViewerCamera::recalcPyramidLevel(int numScales) noexcept
{
    if (numScales <= 0) {
        dsScaleIdx = 0;
        dsScale = 1.0f;
        return;
    }

    const float minScale = std::pow(2.0f, -static_cast<float>(numScales));

    if (scale < minScale) {
        dsScaleIdx = numScales - 1;
    } else {
        // Interpolate DOWN: always use the finer (higher-res) pyramid level.
        // ceil(-log2(scale)) gives the coarser level; subtract 1 for finer.
        // e.g. scale 0.75 → ceil(0.415) = 1, minus 1 = 0 → native
        // e.g. scale 0.3  → ceil(1.737) = 2, minus 1 = 1 → 2x downsample
        // e.g. scale 2.0  → ceil(-1.0)  = -1, minus 1 = -2 → clamped to 0
        // Epsilon prevents float rounding near exact powers of 2.
        float rawLevel = -std::log2(std::max(scale, 1e-6f));
        dsScaleIdx = std::max(0, static_cast<int>(std::ceil(rawLevel - 0.001f)) - 1);
    }

    dsScaleIdx = std::clamp(dsScaleIdx, 0, numScales - 1);

    if (downscaleOverride > 0) {
        dsScaleIdx += downscaleOverride;
        dsScaleIdx = std::min(dsScaleIdx, numScales - 1);
    }

    dsScale = std::pow(2.0f, -dsScaleIdx);
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

float TiledViewerCamera::roundScale(float s) noexcept
{
    s = std::clamp(s, MIN_SCALE, MAX_SCALE);
    return kZoomStops[closestStopIndex(s)];
}

float TiledViewerCamera::stepScale(float current, int steps) noexcept
{
    // Continuous zoom: 1% per step at all zoom levels.
    float result = current * std::pow(1.01f, static_cast<float>(steps));
    return std::clamp(result, MIN_SCALE, MAX_SCALE);
}
