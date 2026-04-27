#include "vc/core/render/AdaptiveCamera.hpp"
#include <algorithm>

void AdaptiveCamera::recalcPyramidLevel(int numScales, const float* scaleFactors) noexcept
{
    if (numScales <= 0) {
        dsScaleIdx = 0;
        dsScale = 1.0f;
        return;
    }

    if (scaleFactors) {
        // Pick the finest level whose scale factor <= 1/scale (i.e. the level
        // that can provide at least one data voxel per screen pixel).
        // scale=0.125 → need sf<=8 → pick level with sf=8.
        // scale=0.03  → need sf<=33 → pick level with sf=32.
        float needed = 1.0f / std::max(scale, 1e-6f);
        dsScaleIdx = 0;
        for (int i = 0; i < numScales; i++) {
            if (scaleFactors[i] <= needed + 0.001f)
                dsScaleIdx = i;
            else
                break;
        }
    } else {
        const float minScale = std::pow(2.0f, -static_cast<float>(numScales));
        if (scale < minScale) {
            dsScaleIdx = numScales - 1;
        } else {
            float rawLevel = -std::log2(std::max(scale, 1e-6f));
            dsScaleIdx = std::max(0, static_cast<int>(std::ceil(rawLevel - 0.001f)) - 1);
        }
    }

    dsScaleIdx = std::clamp(dsScaleIdx, 0, numScales - 1);

    if (downscaleOverride > 0) {
        dsScaleIdx += downscaleOverride;
        dsScaleIdx = std::min(dsScaleIdx, numScales - 1);
    }

    if (scaleFactors && dsScaleIdx >= 0 && dsScaleIdx < numScales) {
        dsScale = 1.0f / scaleFactors[dsScaleIdx];
    } else {
        static constexpr float kInvPow2[24] = {
            1.0f,        1.0f/2,      1.0f/4,       1.0f/8,
            1.0f/16,     1.0f/32,     1.0f/64,      1.0f/128,
            1.0f/256,    1.0f/512,    1.0f/1024,    1.0f/2048,
            1.0f/4096,   1.0f/8192,   1.0f/16384,   1.0f/32768,
            1.0f/65536,  1.0f/131072, 1.0f/262144,  1.0f/524288,
            1.0f/1048576,1.0f/2097152,1.0f/4194304, 1.0f/8388608,
        };
        dsScale = kInvPow2[std::clamp(dsScaleIdx, 0, 23)];
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

float AdaptiveCamera::roundScale(float s) noexcept
{
    s = std::clamp(s, MIN_SCALE, MAX_SCALE);
    return kZoomStops[closestStopIndex(s)];
}

float AdaptiveCamera::stepScale(float current, int steps) noexcept
{
    // Continuous zoom: 1% per step at all zoom levels.
    float result = current * std::pow(1.01f, static_cast<float>(steps));
    return std::clamp(result, MIN_SCALE, MAX_SCALE);
}
