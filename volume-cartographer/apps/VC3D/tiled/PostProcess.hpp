#pragma once

#include <QImage>
#include <opencv2/core.hpp>
#include <string>
#include <cstdint>

#include "vc/core/util/PostProcess.hpp"

// App-layer post-processing parameters.
// Extends core vc::PostProcessParams with colormap (Qt-dependent).
struct PostProcessParams {
    // Window/level: map [windowLow, windowHigh] -> [0, 255]
    float windowLow = 0.0f;
    float windowHigh = 255.0f;

    // Stretch: auto-stretch to full range (overrides window/level)
    bool stretchValues = false;

    // Composite-specific post-processing
    bool postStretchValues = false;
    bool removeSmallComponents = false;
    int minComponentSize = 50;

    // ISO cutoff: zero out values below threshold (0 = disabled)
    uint8_t isoCutoff = 0;

    // Colormap (empty = grayscale) — Qt-dependent, stays in app layer
    std::string colormapId;

    // Convert to core params (without colormap)
    vc::PostProcessParams toCoreParams() const {
        vc::PostProcessParams p;
        p.windowLow = windowLow;
        p.windowHigh = windowHigh;
        p.stretchValues = stretchValues;
        p.postStretchValues = postStretchValues;
        p.removeSmallComponents = removeSmallComponents;
        p.minComponentSize = minComponentSize;
        p.isoCutoff = isoCutoff;
        return p;
    }
};

// Apply all post-processing steps and produce a QImage::Format_RGB32 directly.
// The input gray mat is modified in-place (caller should not reuse it).
//   1. ISO cutoff
//   2. Composite post-stretch (if enabled)
//   3. Composite component removal (if enabled)
//   4. Window/level or value stretch
//   5. Colormap or grayscale → RGB32
QImage applyPostProcess(cv::Mat_<uint8_t>& gray,
                        const PostProcessParams& params);
