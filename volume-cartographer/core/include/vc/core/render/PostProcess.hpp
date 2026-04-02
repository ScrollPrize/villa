#pragma once

#include <opencv2/core.hpp>
#include <array>
#include <string>
#include <cstdint>

#include "vc/core/util/PostProcess.hpp"
#include "vc/core/render/Colormaps.hpp"

namespace vc {

// Render-layer post-processing parameters.
// Extends core vc::PostProcessParams with colormap.
struct RenderPostProcessParams {
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

    // Colormap (empty = grayscale)
    std::string colormapId;

    // Convert to core params (without colormap)
    PostProcessParams toCoreParams() const {
        PostProcessParams p;
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

// Apply all post-processing steps and write ARGB32 pixels into outBuf.
// outBuf must point to gray.rows * outStride uint32_t elements.
// outStride is in uint32_t units (pixels per row including padding).
// The input gray mat is modified in-place (caller should not reuse it).
//   1. ISO cutoff
//   2. Composite post-stretch (if enabled)
//   3. Composite component removal (if enabled)
//   4. Window/level or value stretch
//   5. Colormap or grayscale -> ARGB32
void applyRenderPostProcess(cv::Mat_<uint8_t>& gray,
                            const RenderPostProcessParams& params,
                            uint32_t* outBuf, int outStride);

// Build a window/level LUT mapping uint8 -> ARGB32 for the fused sampling path.
// Only valid for non-stretch grayscale mode (no colormap, no stretchValues).
// An optional lightFactor (0..1) scales voxel values before the window/level
// mapping, fusing directional lighting into the same LUT.
void buildWindowLevelLut(std::array<uint32_t, 256>& lut,
                         float windowLow, float windowHigh,
                         float lightFactor = 1.0f);

}  // namespace vc
