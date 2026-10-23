#pragma once

#include "Fiber3DScene.hpp"
#include "vc/core/render/Colormaps.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>

namespace vc3d::fibers3d
{

// Use the same complete OpenCV colormaps as the CT viewers, not approximate
// hand-picked colors carrying the names of standard scientific palettes.
inline const std::array<QColor, 256>& paletteStops(Palette palette)
{
    static const auto tables = [] {
        std::array<std::array<QColor, 256>, 4> result;
        cv::Mat_<uint8_t> values(1, 256);
        for (int i = 0; i < 256; ++i)
            values(0, i) = uint8_t(i);
        for (int p = 0; p < 4; ++p) {
            auto spec = vc::resolve(p == 1 ? "magma" : "viridis");
            if (p == 2)
                spec.opencvCode = cv::COLORMAP_CIVIDIS;
            if (p == 3)
                spec.opencvCode = cv::COLORMAP_TURBO;
            std::array<uint32_t, 256> colors;
            vc::makeColors(values, spec, colors.data(), 256);
            for (int i = 0; i < 256; ++i)
                result[p][i] = QColor::fromRgba(colors[i]);
        }
        return result;
    }();
    return tables[std::clamp(int(palette), 0, 3)];
}

inline QColor paletteColor(Palette palette, double value)
{
    const auto& stops = paletteStops(palette);
    const double scaled = std::clamp(value, 0.0, 1.0) * 255.0;
    const int index = std::min(254, int(scaled));
    const double fraction = scaled - index;
    const auto& a = stops[index];
    const auto& b = stops[index + 1];
    return QColor::fromRgbF(a.redF() + (b.redF() - a.redF()) * fraction, a.greenF() + (b.greenF() - a.greenF()) * fraction, a.blueF() + (b.blueF() - a.blueF()) * fraction);
}

}  // namespace vc3d::fibers3d
