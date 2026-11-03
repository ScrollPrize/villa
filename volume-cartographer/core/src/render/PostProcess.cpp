#include "vc/core/render/PostProcess.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cstdint>

#include "vc/core/render/Colormaps.hpp"

namespace vc {

void buildWindowLevelLut(std::array<uint32_t, 256>& lut,
                         float windowLow, float windowHigh) noexcept
{
    const int lo = static_cast<int>(std::clamp(windowLow, 0.0f, 255.0f));
    const int hi = static_cast<int>(
        std::clamp(windowHigh, static_cast<float>(lo + 1), 255.0f));
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

namespace {

// buildWindowLevelColormapLut, also returning the window/level ramp in wl and
// the resolved spec (nullptr for grayscale).
const OverlayColormapSpec* buildColormapLut(std::array<uint32_t, 256>& lut,
                                            std::array<uint32_t, 256>& wl,
                                            float windowLow, float windowHigh,
                                            const std::string& colormapId)
{
    // First build the window/level grayscale ramp.
    buildWindowLevelLut(wl, windowLow, windowHigh);

    if (colormapId.empty()) {
        lut = wl;
        return nullptr;
    }

    const OverlayColormapSpec& spec = vc::resolve(colormapId);
    constexpr uint32_t kBlack = 0xFF000000u;

    if (spec.kind == OverlayColormapKind::DiscreteLut && spec.discreteLut) {
        // Apply window/level to index, then look up discrete palette.
        for (int i = 0; i < 256; ++i) {
            uint8_t mapped = static_cast<uint8_t>(wl[i] & 0xFF);  // gray value
            lut[i] = spec.discreteLut[mapped];
        }
        lut[0] = kBlack;
        return &spec;
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
        lut[0] = kBlack;
        return &spec;
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
    lut[0] = kBlack;
    return &spec;
}

}  // namespace

void buildWindowLevelColormapLut(std::array<uint32_t, 256>& lut,
                                 float windowLow, float windowHigh,
                                 const std::string& colormapId)
{
    std::array<uint32_t, 256> wl;
    buildColormapLut(lut, wl, windowLow, windowHigh, colormapId);
}

void buildOverlayLut(std::array<uint32_t, 256>& lut,
                     float windowLow, float windowHigh,
                     const std::string& colormapId,
                     bool intensityAsAlpha)
{
    std::array<uint32_t, 256> color, wl;
    const OverlayColormapSpec* spec =
        buildColormapLut(color, wl, windowLow, windowHigh, colormapId);
    // Label values are not intensities, so labels always use the default mode.
    const bool discrete = spec && spec->kind == OverlayColormapKind::DiscreteLut;

    if (!intensityAsAlpha || discrete) {
        for (int i = 0; i < 256; ++i) {
            const float v = static_cast<float>(i);
            lut[i] = (v >= windowLow && v <= windowHigh) ? color[i] : 0u;
        }
        return;
    }

    // Empty id is grayscale, which is black at zero like a tint.
    const bool premultiplied = !spec || spec->premultiplied;

    for (int i = 0; i < 256; ++i) {
        const uint32_t t = wl[i] & 0xFFu;
        if (t == 0) {
            lut[i] = 0u;
            continue;
        }
        const uint32_t r = (color[i] >> 16) & 0xFFu;
        const uint32_t g = (color[i] >> 8) & 0xFFu;
        const uint32_t b = color[i] & 0xFFu;
        if (premultiplied) {
            // Alpha is the brightest channel, so rgb <= alpha holds and the
            // result over black is exactly the opaque colormap.
            const uint32_t alpha = std::max({r, g, b});
            lut[i] = alpha == 0 ? 0u : ((alpha << 24) | (color[i] & 0x00FFFFFFu));
        } else {
            const auto scale = [t](uint32_t c) {
                return (c * t + 127u) / 255u;
            };
            lut[i] = (t << 24) | (scale(r) << 16) | (scale(g) << 8) | scale(b);
        }
    }
}

}  // namespace vc
