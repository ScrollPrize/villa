#include "PostProcess.hpp"
#include "VolumeViewerCmaps.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cstdint>

// Build a fused uint32_t[256] LUT: window/level (or stretch) + gray→ARGB32.
// Eliminates the intermediate uint8 Mat that cv::LUT would produce.
static void buildFusedGrayLut(std::array<uint32_t, 256>& lut,
                              const PostProcessParams& params,
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

QImage applyPostProcess(cv::Mat_<uint8_t>& gray,
                        const PostProcessParams& params)
{
    const volume_viewer_cmaps::OverlayColormapSpec* spec = nullptr;
    if (!params.colormapId.empty()) {
        spec = &volume_viewer_cmaps::resolve(params.colormapId);
    }

    if (spec && spec->kind == volume_viewer_cmaps::OverlayColormapKind::DiscreteLut) {
        auto preservedParams = params.toCoreParams();
        preservedParams.windowLow = 0.0f;
        preservedParams.windowHigh = 255.0f;
        preservedParams.stretchValues = false;
        vc::applyPostProcess(gray, preservedParams);

        const int cutoff = static_cast<int>(std::clamp(params.windowLow, 0.0f, 255.0f));
        if (cutoff > 0) {
            cv::threshold(gray, gray, cutoff - 1, 0, cv::THRESH_TOZERO);
        }

        return volume_viewer_cmaps::makeColors(gray, *spec);
    }

    // Colormap path: full core pipeline then colormap conversion.
    if (spec) {
        vc::applyPostProcess(gray, params.toCoreParams());
        return volume_viewer_cmaps::makeColors(gray, *spec);
    }

    // Grayscale path: fused window/level + gray→RGB32 in a single pass.
    // Run core steps 1-3 (ISO cutoff, post-stretch, component removal)
    // with identity window so step 4 is a no-op.
    auto coreParams = params.toCoreParams();
    coreParams.windowLow = 0.0f;
    coreParams.windowHigh = 255.0f;
    coreParams.stretchValues = false;
    vc::applyPostProcess(gray, coreParams);

    // Build fused LUT: window/level (or stretch) → ARGB32
    std::array<uint32_t, 256> lut;
    buildFusedGrayLut(lut, params, gray);

    // Single-pass: gray → RGB32 via fused LUT
    const int rows = gray.rows;
    const int cols = gray.cols;
    QImage result(cols, rows, QImage::Format_RGB32);

    auto* bits = reinterpret_cast<uint32_t*>(result.bits());
    const int stride = result.bytesPerLine() / 4;
    for (int y = 0; y < rows; ++y) {
        const auto* src = gray.ptr<uint8_t>(y);
        auto* dst = bits + y * stride;
        for (int x = 0; x < cols; ++x) {
            dst[x] = lut[src[x]];
        }
    }
    return result;
}
