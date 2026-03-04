#include "VolumeViewerCmaps.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdint>

namespace volume_viewer_cmaps
{

namespace
{
const std::vector<OverlayColormapSpec>& buildSpecs()
{
    // tint values are now R, G, B order (matching output format)
    static const std::vector<OverlayColormapSpec> specs = {
        {"fire", QStringLiteral("Fire"), OverlayColormapKind::OpenCv, cv::COLORMAP_HOT, {}},
        {"viridis", QStringLiteral("Viridis"), OverlayColormapKind::OpenCv, cv::COLORMAP_VIRIDIS, {}},
        {"magma", QStringLiteral("Magma"), OverlayColormapKind::OpenCv, cv::COLORMAP_MAGMA, {}},
        {"red", QStringLiteral("Red"), OverlayColormapKind::Tint, 0, cv::Vec3f(1.0f, 0.0f, 0.0f)},
        {"green", QStringLiteral("Green"), OverlayColormapKind::Tint, 0, cv::Vec3f(0.0f, 1.0f, 0.0f)},
        {"blue", QStringLiteral("Blue"), OverlayColormapKind::Tint, 0, cv::Vec3f(0.0f, 0.0f, 1.0f)},
        {"cyan", QStringLiteral("Cyan"), OverlayColormapKind::Tint, 0, cv::Vec3f(0.0f, 1.0f, 1.0f)},
        {"magenta", QStringLiteral("Magenta"), OverlayColormapKind::Tint, 0, cv::Vec3f(1.0f, 0.0f, 1.0f)}
    };
    return specs;
}
} // namespace

const std::vector<OverlayColormapSpec>& specs()
{
    static const std::vector<OverlayColormapSpec>& specsRef = buildSpecs();
    return specsRef;
}

const OverlayColormapSpec& resolve(const std::string& id)
{
    const auto& allSpecs = specs();
    auto it = std::find_if(allSpecs.begin(), allSpecs.end(), [&id](const auto& spec) {
        return spec.id == id;
    });
    if (it != allSpecs.end()) {
        return *it;
    }
    return allSpecs.front();
}

QImage makeColors(const cv::Mat_<uint8_t>& values, const OverlayColormapSpec& spec)
{
    if (values.empty()) {
        return {};
    }

    const int rows = values.rows;
    const int cols = values.cols;
    QImage result(cols, rows, QImage::Format_RGB32);

    if (spec.kind == OverlayColormapKind::OpenCv) {
        // OpenCV colormaps produce BGR output. Build a 256-entry XRGB LUT
        // once, then map pixels directly — avoids applyColorMap's iterator overhead.
        cv::Mat lut_input(1, 256, CV_8UC1);
        for (int i = 0; i < 256; ++i)
            lut_input.at<uint8_t>(0, i) = static_cast<uint8_t>(i);

        cv::Mat lut_bgr;
        cv::applyColorMap(lut_input, lut_bgr, spec.opencvCode);

        // Convert BGR LUT to packed XRGB32
        uint32_t lut[256];
        const auto* bgrRow = lut_bgr.ptr<cv::Vec3b>(0);
        for (int i = 0; i < 256; ++i) {
            lut[i] = 0xFF000000u
                   | (static_cast<uint32_t>(bgrRow[i][2]) << 16)
                   | (static_cast<uint32_t>(bgrRow[i][1]) << 8)
                   |  static_cast<uint32_t>(bgrRow[i][0]);
        }

        for (int y = 0; y < rows; ++y) {
            const auto* src = values.ptr<uint8_t>(y);
            auto* dst = reinterpret_cast<uint32_t*>(result.scanLine(y));
            for (int x = 0; x < cols; ++x) {
                dst[x] = lut[src[x]];
            }
        }
    } else {
        // Tint colormap: value * tint[channel]
        // tint is R, G, B in [0,1]
        const float tR = spec.tint[0] * 255.0f;
        const float tG = spec.tint[1] * 255.0f;
        const float tB = spec.tint[2] * 255.0f;

        // Build 256-entry LUT for each channel
        uint32_t lut[256];
        for (int i = 0; i < 256; ++i) {
            float f = static_cast<float>(i) / 255.0f;
            auto r = static_cast<uint32_t>(std::min(f * tR, 255.0f));
            auto g = static_cast<uint32_t>(std::min(f * tG, 255.0f));
            auto b = static_cast<uint32_t>(std::min(f * tB, 255.0f));
            lut[i] = 0xFF000000u | (r << 16) | (g << 8) | b;
        }

        for (int y = 0; y < rows; ++y) {
            const auto* src = values.ptr<uint8_t>(y);
            auto* dst = reinterpret_cast<uint32_t*>(result.scanLine(y));
            for (int x = 0; x < cols; ++x) {
                dst[x] = lut[src[x]];
            }
        }
    }
    return result;
}

const std::vector<OverlayColormapEntry>& entries()
{
    static std::vector<OverlayColormapEntry> cachedEntries;
    static bool initialized = false;
    if (!initialized) {
        const auto& allSpecs = specs();
        cachedEntries.reserve(allSpecs.size());
        for (const auto& spec : allSpecs) {
            cachedEntries.push_back({spec.label, spec.id});
        }
        initialized = true;
    }
    return cachedEntries;
}

} // namespace volume_viewer_cmaps
