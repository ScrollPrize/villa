#include "VolumeViewerCmaps.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>

namespace volume_viewer_cmaps
{

namespace
{
const std::vector<OverlayColormapSpec>& buildSpecs()
{
    static const std::vector<OverlayColormapSpec> specs = {
        {"fire", QStringLiteral("Fire"), OverlayColormapKind::OpenCv, cv::COLORMAP_HOT, {}},
        {"viridis", QStringLiteral("Viridis"), OverlayColormapKind::OpenCv, cv::COLORMAP_VIRIDIS, {}},
        {"magma", QStringLiteral("Magma"), OverlayColormapKind::OpenCv, cv::COLORMAP_MAGMA, {}},
        {"red", QStringLiteral("Red"), OverlayColormapKind::Tint, 0, cv::Vec3f(0.0f, 0.0f, 1.0f)},
        {"green", QStringLiteral("Green"), OverlayColormapKind::Tint, 0, cv::Vec3f(0.0f, 1.0f, 0.0f)},
        {"blue", QStringLiteral("Blue"), OverlayColormapKind::Tint, 0, cv::Vec3f(1.0f, 0.0f, 0.0f)},
        {"cyan", QStringLiteral("Cyan"), OverlayColormapKind::Tint, 0, cv::Vec3f(1.0f, 1.0f, 0.0f)},
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

cv::Mat makeColors(const cv::Mat_<uint8_t>& values, const OverlayColormapSpec& spec)
{
    if (values.empty()) {
        return {};
    }

    cv::Mat colored;
    if (spec.kind == OverlayColormapKind::OpenCv) {
        cv::applyColorMap(values, colored, spec.opencvCode);
    } else {
        cv::Mat valuesFloat;
        values.convertTo(valuesFloat, CV_32F, 1.0f / 255.0f);
        std::vector<cv::Mat> channels(3);
        for (int c = 0; c < 3; ++c) {
            channels[c] = valuesFloat * (spec.tint[c] * 255.0f);
        }
        cv::merge(channels, colored);
        colored.convertTo(colored, CV_8UC3);
    }
    return colored;
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
