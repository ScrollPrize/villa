#pragma once

#include <opencv2/core.hpp>
#include <cstdint>

namespace vc {

// Grayscale post-processing parameters (no colormap — Qt-free).
struct PostProcessParams {
    float windowLow = 0.0f;
    float windowHigh = 255.0f;
    bool stretchValues = false;
    bool postStretchValues = false;
    bool removeSmallComponents = false;
    int minComponentSize = 50;
    uint8_t isoCutoff = 0;
};

// Apply grayscale post-processing pipeline in canonical order:
//   1. ISO cutoff
//   2. Composite post-stretch (if enabled)
//   3. Composite component removal (if enabled)
//   4. Window/level or value stretch
// Input/output: single-channel uint8 grayscale.
void applyPostProcess(cv::Mat_<uint8_t>& img, const PostProcessParams& params);

}  // namespace vc
