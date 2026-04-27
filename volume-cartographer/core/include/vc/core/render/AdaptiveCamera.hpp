#pragma once

#include <cstdint>
#include <cmath>
#include <opencv2/core.hpp>

// Camera state for the tiled volume viewer.
// All fields are main-thread only (no locking needed).
struct AdaptiveCamera {
    // Center of view in surface parameter space
    cv::Vec3f surfacePtr{0, 0, 0};

    // Zoom level (same 0.03125..4.0 range as CVolumeViewer::_scale)
    float scale = 0.5f;

    // Normal offset (shift+wheel slice navigation)
    float zOff = 0.0f;

    // Pyramid level index (0 = full res, higher = coarser).
    // Always the coarsest level that covers the current zoom — never finer
    // than needed.  The renderer upscales from this single level.
    int dsScaleIdx = 1;

    // Derived: 2^(-dsScaleIdx), e.g., level 2 -> 0.25
    float dsScale = 0.5f;

    // Additional override for downscale (from settings)
    int downscaleOverride = 0;

    // Zoom limits
    static constexpr float MIN_SCALE = 0.01f;
    static constexpr float MAX_SCALE = 10.0f;

    constexpr void invalidate() noexcept { }

    // Recalculate pyramid level from current scale.
    // numScales = volume->numScales()
    // scaleFactors: optional per-level OME-Zarr scale factors (e.g. [8,16,32,...]).
    //   When non-null, dsScale = 1/scaleFactors[dsScaleIdx] instead of 1/2^idx.
    void recalcPyramidLevel(int numScales, const float* scaleFactors = nullptr) noexcept;

    // Snap scale to the nearest predefined zoom stop
    static float roundScale(float s) noexcept;

    // Step to the next (+1) or previous (-1) zoom stop from current scale.
    // |steps| > 1 skips multiple stops.  Returns the new scale.
    static float stepScale(float current, int steps) noexcept;
};
