#include "vc/core/render/PostProcess.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cstdint>

namespace vc {

// Build a fused uint32_t[256] LUT: window/level (or stretch) + gray->ARGB32.
// Eliminates the intermediate uint8 Mat that cv::LUT would produce.
static void buildFusedGrayLut(std::array<uint32_t, 256>& lut,
                              const RenderPostProcessParams& params,
                              const cv::Mat_<uint8_t>& gray)
{
    if (params.stretchValues) {
        double minVal, maxVal;
        cv::minMaxLoc(gray, &minVal, &maxVal);
        const double range = std::max(1.0, maxVal - minVal);
        for (int i = 0; i < 256; ++i) {
            uint8_t v = static_cast<uint8_t>(
                std::clamp((i - minVal) / range * 255.0, 0.0, 255.0));
            lut[i] = 0xFF000000u | (static_cast<uint32_t>(v) << 16)
                                 | (static_cast<uint32_t>(v) << 8)
                                 | static_cast<uint32_t>(v);
        }
    } else {
        const int lo = static_cast<int>(
            std::clamp(params.windowLow, 0.0f, 255.0f));
        const int hi = static_cast<int>(
            std::clamp(params.windowHigh, static_cast<float>(lo + 1), 255.0f));
        const float span = std::max(1.0f, static_cast<float>(hi - lo));
        for (int i = 0; i < 256; ++i) {
            uint8_t v = static_cast<uint8_t>(
                std::clamp((static_cast<float>(i) - static_cast<float>(lo))
                           / span * 255.0f, 0.0f, 255.0f));
            lut[i] = 0xFF000000u | (static_cast<uint32_t>(v) << 16)
                                 | (static_cast<uint32_t>(v) << 8)
                                 | static_cast<uint32_t>(v);
        }
    }
}

// Check whether core preprocessing steps 1-3 are all no-ops.
static bool corePreprocessIsNoop(const RenderPostProcessParams& params)
{
    return params.isoCutoff == 0
        && !params.postStretchValues
        && !params.removeSmallComponents;
}

// Apply the fused LUT to convert grayscale pixels to ARGB32.
// Handles contiguous and row-strided layouts, processes 4 pixels at a time.
static void applyFusedLut(const cv::Mat_<uint8_t>& gray,
                          const std::array<uint32_t, 256>& lut,
                          uint32_t* bits, int stride)
{
    const int rows = gray.rows;
    const int cols = gray.cols;

    for (int y = 0; y < rows; ++y) {
        const auto* __restrict__ src = gray.ptr<uint8_t>(y);
        auto* __restrict__ dst = bits + y * stride;

        int x = 0;
        // Process 4 pixels per iteration -- scalar gather, batch store.
        // The LUT is 1KB and stays hot in L1. Unrolling 4x cuts loop
        // overhead and lets the CPU pipeline overlapping loads/stores.
        const int cols4 = cols - 3;
        for (; x < cols4; x += 4) {
            nt_store_u32(&dst[x + 0], lut[src[x + 0]]);
            nt_store_u32(&dst[x + 1], lut[src[x + 1]]);
            nt_store_u32(&dst[x + 2], lut[src[x + 2]]);
            nt_store_u32(&dst[x + 3], lut[src[x + 3]]);
        }
        for (; x < cols; ++x) {
            nt_store_u32(&dst[x], lut[src[x]]);
        }
    }
}

void applyRenderPostProcess(cv::Mat_<uint8_t>& gray,
                            const RenderPostProcessParams& params,
                            uint32_t* outBuf, int outStride)
{
    const OverlayColormapSpec* spec = nullptr;
    if (!params.colormapId.empty()) {
        spec = &vc::resolve(params.colormapId);
    }

    if (spec && spec->kind == OverlayColormapKind::DiscreteLut) {
        auto preservedParams = params.toCoreParams();
        preservedParams.windowLow = 0.0f;
        preservedParams.windowHigh = 255.0f;
        preservedParams.stretchValues = false;
        vc::applyPostProcess(gray, preservedParams);

        const int cutoff = static_cast<int>(std::clamp(params.windowLow, 0.0f, 255.0f));
        if (cutoff > 0) {
            cv::threshold(gray, gray, cutoff - 1, 0, cv::THRESH_TOZERO);
        }

        vc::makeColors(gray, *spec, outBuf, outStride);
        return;
    }

    // Colormap path: full core pipeline then colormap conversion.
    if (spec) {
        vc::applyPostProcess(gray, params.toCoreParams());
        vc::makeColors(gray, *spec, outBuf, outStride);
        return;
    }

    // Grayscale path: fused window/level + gray->ARGB32 in a single pass.
    // Skip core preprocessing entirely when all steps are no-ops (common case
    // for volume tiles: isoCutoff=0, no postStretch, no component removal).
    if (!corePreprocessIsNoop(params)) {
        auto coreParams = params.toCoreParams();
        coreParams.windowLow = 0.0f;
        coreParams.windowHigh = 255.0f;
        coreParams.stretchValues = false;
        vc::applyPostProcess(gray, coreParams);
    }

    // Build fused LUT: window/level (or stretch) -> ARGB32
    // Thread-local cache: when stretchValues is false (common case), the LUT
    // depends only on windowLow/windowHigh and is identical for every tile.
    thread_local std::array<uint32_t, 256> cachedLut;
    thread_local float cachedLow = -1.0f;
    thread_local float cachedHigh = -1.0f;
    thread_local bool cachedStretch = true;  // force rebuild on first call

    if (params.stretchValues || params.windowLow != cachedLow || params.windowHigh != cachedHigh || cachedStretch != params.stretchValues) {
        buildFusedGrayLut(cachedLut, params, gray);
        cachedLow = params.windowLow;
        cachedHigh = params.windowHigh;
        cachedStretch = params.stretchValues;
    }

    // Single-pass: gray -> ARGB32 via fused LUT (4x unrolled)
    applyFusedLut(gray, cachedLut, outBuf, outStride);
}

void buildWindowLevelLut(std::array<uint32_t, 256>& lut,
                         float windowLow, float windowHigh,
                         float lightFactor) noexcept
{
    const int lo = static_cast<int>(std::clamp(windowLow, 0.0f, 255.0f));
    const int hi = static_cast<int>(
        std::clamp(windowHigh, static_cast<float>(lo + 1), 255.0f));
    const float span = std::max(1.0f, static_cast<float>(hi - lo));
    const bool applyLight = lightFactor < 1.0f;
    for (int i = 0; i < 256; ++i) {
        float lit = applyLight
            ? std::clamp(static_cast<float>(i) * lightFactor, 0.0f, 255.0f)
            : static_cast<float>(i);
        uint8_t v = static_cast<uint8_t>(
            std::clamp((lit - static_cast<float>(lo))
                       / span * 255.0f, 0.0f, 255.0f));
        lut[i] = 0xFF000000u | (static_cast<uint32_t>(v) << 16)
                              | (static_cast<uint32_t>(v) << 8)
                              | static_cast<uint32_t>(v);
    }
}

void buildWindowLevelColormapLut(std::array<uint32_t, 256>& lut,
                                 float windowLow, float windowHigh,
                                 const std::string& colormapId,
                                 float lightFactor)
{
    // First build the window/level grayscale ramp.
    std::array<uint32_t, 256> wl;
    buildWindowLevelLut(wl, windowLow, windowHigh, lightFactor);

    if (colormapId.empty()) {
        lut = wl;
        return;
    }

    const OverlayColormapSpec& spec = vc::resolve(colormapId);

    if (spec.kind == OverlayColormapKind::DiscreteLut && spec.discreteLut) {
        // Apply window/level to index, then look up discrete palette.
        for (int i = 0; i < 256; ++i) {
            uint8_t mapped = static_cast<uint8_t>(wl[i] & 0xFF);  // gray value
            lut[i] = spec.discreteLut[mapped];
        }
        return;
    }

    if (spec.kind == OverlayColormapKind::Tint) {
        const float r = spec.tint[0], g = spec.tint[1], b = spec.tint[2];
        for (int i = 0; i < 256; ++i) {
            uint8_t v = static_cast<uint8_t>(wl[i] & 0xFF);
            uint8_t R = static_cast<uint8_t>(v * r);
            uint8_t G = static_cast<uint8_t>(v * g);
            uint8_t B = static_cast<uint8_t>(v * b);
            lut[i] = 0xFF000000u
                   | (static_cast<uint32_t>(R) << 16)
                   | (static_cast<uint32_t>(G) << 8)
                   | static_cast<uint32_t>(B);
        }
        return;
    }

    // OpenCV colormap: apply to an 8-bit ramp, then fuse with wl.
    cv::Mat ramp(1, 256, CV_8UC1);
    for (int i = 0; i < 256; ++i) ramp.at<uint8_t>(0, i) = uint8_t(i);
    cv::Mat colored;
    cv::applyColorMap(ramp, colored, spec.opencvCode);
    // colored is BGR. Index it by post-WL gray value.
    for (int i = 0; i < 256; ++i) {
        uint8_t v = static_cast<uint8_t>(wl[i] & 0xFF);
        const auto& bgr = colored.at<cv::Vec3b>(0, v);
        lut[i] = 0xFF000000u
               | (static_cast<uint32_t>(bgr[2]) << 16)
               | (static_cast<uint32_t>(bgr[1]) << 8)
               | static_cast<uint32_t>(bgr[0]);
    }
}

}  // namespace vc
