#pragma once

#include "vc/core/render/PostProcess.hpp"

// App-layer backward-compatible alias.
using PostProcessParams = vc::RenderPostProcessParams;

inline void applyPostProcess(cv::Mat_<uint8_t>& gray,
                             const PostProcessParams& params,
                             uint32_t* outBuf, int outStride) {
    vc::applyRenderPostProcess(gray, params, outBuf, outStride);
}

inline void buildWindowLevelLut(std::array<uint32_t, 256>& lut,
                                float windowLow, float windowHigh,
                                float lightFactor = 1.0f) {
    vc::buildWindowLevelLut(lut, windowLow, windowHigh, lightFactor);
}
