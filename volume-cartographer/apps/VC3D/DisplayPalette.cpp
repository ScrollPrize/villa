#include "DisplayPalette.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cstdint>

#include "Colormaps.hpp"

namespace vc {

namespace {

// buildWindowLevelColormapLut, also returning the window/level ramp in wl and
// the resolved spec (nullptr for grayscale).
const OverlayColormapSpec* buildColormapLut(std::array<uint32_t, 256>& lut,
                                            std::array<uint32_t, 256>& wl,
                                            float windowLow, float windowHigh,
                                            const std::string& colormapId)
{
    // First build the window/level grayscale ramp.
    vcr::display::windowLevelLut(wl, windowLow, windowHigh);

    if (colormapId.empty()) {
        lut = wl;
        return nullptr;
    }

    const OverlayColormapSpec& spec = vc::resolve(colormapId);
    std::array<uint32_t, 256> palette;

    if (spec.kind == OverlayColormapKind::DiscreteLut && spec.discreteLut) {
        std::copy_n(spec.discreteLut, 256, palette.begin());
        vcr::display::paletteLut(lut, palette, windowLow, windowHigh);
        return &spec;
    }

    if (spec.kind == OverlayColormapKind::Tint) {
        const float r = spec.tint[0], g = spec.tint[1], b = spec.tint[2];
        for (int i = 0; i < 256; ++i) {
            uint8_t v = static_cast<uint8_t>(i);
            uint8_t R = static_cast<uint8_t>(v * r);
            uint8_t G = static_cast<uint8_t>(v * g);
            uint8_t B = static_cast<uint8_t>(v * b);
            palette[i] = 0xFF000000u
                   | (static_cast<uint32_t>(R) << 16)
                   | (static_cast<uint32_t>(G) << 8)
                   | static_cast<uint32_t>(B);
        }
        vcr::display::paletteLut(lut, palette, windowLow, windowHigh);
        return &spec;
    }

    // OpenCV colormap: apply to an 8-bit ramp, then fuse with wl.
    cv::Mat ramp(1, 256, CV_8UC1);
    for (int i = 0; i < 256; ++i) ramp.at<uint8_t>(0, i) = uint8_t(i);
    cv::Mat colored;
    cv::applyColorMap(ramp, colored, spec.opencvCode);
    // Palette generation stays with application color selection.
    for (int i = 0; i < 256; ++i) {
        uint8_t v = static_cast<uint8_t>(i);
        const auto& bgr = colored.at<cv::Vec3b>(0, v);
        palette[i] = 0xFF000000u
               | (static_cast<uint32_t>(bgr[2]) << 16)
               | (static_cast<uint32_t>(bgr[1]) << 8)
               | static_cast<uint32_t>(bgr[0]);
    }
    vcr::display::paletteLut(lut, palette, windowLow, windowHigh);
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

    vcr::display::overlayLut(lut, color, windowLow, windowHigh,
                             intensityAsAlpha && !discrete, !spec || spec->premultiplied);
}

} // namespace vc
