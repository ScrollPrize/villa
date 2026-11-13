#pragma once

#include "Colormaps.hpp"

#include <QImage>

#include <opencv2/core/mat.hpp>

#include <cstdint>
#include <string>

namespace vc3d {

// An 8-bit scalar image as an overlay image: gray, or the colors of `colormapId`,
// with each pixel's value as its alpha, so 0 is transparent. The ink detection
// overlays draw single-channel predictions with it.
inline QImage scalarOverlayImage(const cv::Mat_<uint8_t>& scalar, const std::string& colormapId)
{
    if (scalar.empty()) {
        return {};
    }

    QImage image(scalar.cols, scalar.rows, QImage::Format_ARGB32);
    if (colormapId.empty()) {
        for (int y = 0; y < scalar.rows; ++y) {
            auto* out = reinterpret_cast<QRgb*>(image.scanLine(y));
            const auto* in = scalar.ptr<uint8_t>(y);
            for (int x = 0; x < scalar.cols; ++x) {
                const int v = in[x];
                out[x] = qRgba(v, v, v, v);
            }
        }
        return image;
    }

    vc::makeColors(
        scalar,
        vc::resolve(colormapId),
        reinterpret_cast<uint32_t*>(image.bits()),
        image.bytesPerLine() / 4);

    for (int y = 0; y < scalar.rows; ++y) {
        auto* out = reinterpret_cast<QRgb*>(image.scanLine(y));
        const auto* in = scalar.ptr<uint8_t>(y);
        for (int x = 0; x < scalar.cols; ++x) {
            const QRgb color = out[x];
            out[x] = qRgba(qRed(color), qGreen(color), qBlue(color), in[x]);
        }
    }
    return image;
}

} // namespace vc3d
