#include "vc/core/render/PostProcess.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cstdint>

#include "vc/core/render/Colormaps.hpp"

namespace vc {

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
    constexpr uint32_t kBlack = 0xFF000000u;

    if (spec.kind == OverlayColormapKind::DiscreteLut && spec.discreteLut) {
        // Apply window/level to index, then look up discrete palette.
        for (int i = 0; i < 256; ++i) {
            uint8_t mapped = static_cast<uint8_t>(wl[i] & 0xFF);  // gray value
            lut[i] = spec.discreteLut[mapped];
        }
        lut[0] = kBlack;
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
        lut[0] = kBlack;
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
    lut[0] = kBlack;
}

}  // namespace vc
