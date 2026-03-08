#include "VolumeViewerCmaps.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cstdint>

namespace volume_viewer_cmaps
{

namespace
{
const std::vector<OverlayColormapSpec>& buildSpecs();

QImage applyPackedLut(const cv::Mat_<uint8_t>& values, const uint32_t* lut)
{
    const int rows = values.rows;
    const int cols = values.cols;
    QImage result(cols, rows, QImage::Format_RGB32);

    for (int y = 0; y < rows; ++y) {
        const auto* src = values.ptr<uint8_t>(y);
        auto* dst = reinterpret_cast<uint32_t*>(result.scanLine(y));
        for (int x = 0; x < cols; ++x) {
            dst[x] = lut[src[x]];
        }
    }

    return result;
}

// Fixed Glasbey-like categorical palette generated offline from a maximin RGB
// search, with label 0 reserved for black.
constexpr std::array<uint32_t, 256> kGlasbeyBlack0Lut = {
    0xFF000000u, 0xFFFFFFFFu, 0xFFFF0000u, 0xFF00FF00u, 0xFF0000FFu, 0xFFFFFF00u, 0xFFFF00FFu, 0xFF00FFFFu,
    0xFF929292u, 0xFF006D6Du, 0xFF6D006Du, 0xFF6D6D00u, 0xFF4992FFu, 0xFF49FF92u, 0xFFFF4992u, 0xFF9224DBu,
    0xFF92DB24u, 0xFFDB6D24u, 0xFFDB92FFu, 0xFFDBFF92u, 0xFF92FFFFu, 0xFF000092u, 0xFF009200u, 0xFF920000u,
    0xFF00B6B6u, 0xFF4949B6u, 0xFF49B649u, 0xFFFFB66Du, 0xFFDB006Du, 0xFF924949u, 0xFF242449u, 0xFF006DFFu,
    0xFF00FF6Du, 0xFFDBB6B6u, 0xFF92B6DBu, 0xFF49DBDBu, 0xFFDB49DBu, 0xFF926DFFu, 0xFF92FF6Du, 0xFFDBB600u,
    0xFF4924FFu, 0xFF49FF24u, 0xFF244900u, 0xFF92FFB6u, 0xFFFFFF49u, 0xFF00B6FFu, 0xFF00FFB6u, 0xFFFF00B6u,
    0xFF4992B6u, 0xFFB64992u, 0xFF929249u, 0xFF006DB6u, 0xFF00B66Du, 0xFF496D6Du, 0xFF4900B6u, 0xFF49B600u,
    0xFF490000u, 0xFFDB6D6Du, 0xFFB600B6u, 0xFFB64900u, 0xFFB6B66Du, 0xFFB600FFu, 0xFFB6FF00u, 0xFFFF4949u,
    0xFFFF4900u, 0xFFDB2424u, 0xFF6D2424u, 0xFFDBDBDBu, 0xFF2449DBu, 0xFF24DB49u, 0xFF929200u, 0xFFFFB6DBu,
    0xFFFFDBB6u, 0xFFFF6DFFu, 0xFFB6DB92u, 0xFFDBDB6Du, 0xFFFF9292u, 0xFF6DDBB6u, 0xFFDB6DB6u, 0xFFB6DB49u,
    0xFFDBDB24u, 0xFFDB9249u, 0xFF6D49FFu, 0xFF6DB692u, 0xFF6DDB6Du, 0xFF926DB6u, 0xFFFF9224u, 0xFF2492DBu,
    0xFF24DB92u, 0xFF496DDBu, 0xFFDB2492u, 0xFFB6B624u, 0xFF6D6D92u, 0xFF6D926Du, 0xFF249292u, 0xFF6D00DBu,
    0xFF6DDB00u, 0xFF922492u, 0xFFB6246Du, 0xFF6D9224u, 0xFF926D24u, 0xFF244992u, 0xFF249249u, 0xFF2424B6u,
    0xFF24B624u, 0xFF920049u, 0xFF494924u, 0xFF006D24u, 0xFFB6DBFFu, 0xFFB6FFDBu, 0xFF49FFFFu, 0xFF6DDBFFu,
    0xFF6DFFDBu, 0xFFB692DBu, 0xFF24DBFFu, 0xFF24FFDBu, 0xFFB649FFu, 0xFFDB24FFu, 0xFFFF24DBu, 0xFF6D92DBu,
    0xFF00DBDBu, 0xFFDB00DBu, 0xFF6DFF49u, 0xFFFF246Du, 0xFF926D6Du, 0xFFB66D49u, 0xFFFF0049u, 0xFF6D24B6u,
    0xFF6D496Du, 0xFF6D6D49u, 0xFF0024DBu, 0xFF00DB24u, 0xFF2400DBu, 0xFF24DB00u, 0xFF492492u, 0xFFB60024u,
    0xFF004949u, 0xFF490049u, 0xFF00246Du, 0xFF24006Du, 0xFF000049u, 0xFF002424u, 0xFF240024u, 0xFFB6B6FFu,
    0xFF92DBDBu, 0xFF9292FFu, 0xFF6DB6FFu, 0xFFB6B6B6u, 0xFFB66DDBu, 0xFF49B6DBu, 0xFF9249DBu, 0xFF49FF6Du,
    0xFF49DB49u, 0xFFB6496Du, 0xFFDB2449u, 0xFF2424FFu, 0xFF24FF24u, 0xFF6D0092u, 0xFF494949u, 0xFF24496Du,
    0xFF246D49u, 0xFF49246Du, 0xFF6D2449u, 0xFF922424u, 0xFFB62400u, 0xFF242400u, 0xFFDBFFFFu, 0xFFFFDBFFu,
    0xFFFFFFDBu, 0xFFDBDBFFu, 0xFFDBFFDBu, 0xFFFFDBDBu, 0xFFB6FFFFu, 0xFFFFB6FFu, 0xFFFFFFB6u, 0xFFDBB6FFu,
    0xFFDBFFB6u, 0xFFFF92FFu, 0xFFFFFF92u, 0xFF92DBFFu, 0xFF92FFDBu, 0xFFB6DBDBu, 0xFFDBB6DBu, 0xFFDBDBB6u,
    0xFFFF92DBu, 0xFFFFDB92u, 0xFF6DFFFFu, 0xFFB6FFB6u, 0xFFFFB6B6u, 0xFFFFFF6Du, 0xFFDB92DBu, 0xFFDBDB92u,
    0xFF92B6FFu, 0xFFB692FFu, 0xFFB6B6DBu, 0xFFB6DBB6u, 0xFFB6FF92u, 0xFFDB6DFFu, 0xFFDBFF6Du, 0xFFFF49FFu,
    0xFFFF6DDBu, 0xFFFF92B6u, 0xFFFFB692u, 0xFFFFDB6Du, 0xFF49DBFFu, 0xFF49FFDBu, 0xFF6DDBDBu, 0xFF92DBB6u,
    0xFF92FF92u, 0xFFDB49FFu, 0xFFDB6DDBu, 0xFFDB92B6u, 0xFFDBB692u, 0xFFDBFF49u, 0xFFFF49DBu, 0xFFFFDB49u,
    0xFF24FFFFu, 0xFF6DFFB6u, 0xFFB66DFFu, 0xFFB6FF6Du, 0xFFFF24FFu, 0xFFFF6DB6u, 0xFFFFFF24u, 0xFF9292DBu,
    0xFF92DB92u, 0xFFDB9292u, 0xFFDBDB49u, 0xFF49B6FFu, 0xFF49FFB6u, 0xFF6D92FFu, 0xFF6DB6DBu, 0xFF6DFF92u,
    0xFF92B6B6u, 0xFFB692B6u, 0xFFB6B692u, 0xFFB6DB6Du, 0xFFB6FF49u, 0xFFDBB66Du, 0xFFDBFF24u, 0xFFFF49B6u,
    0xFFFF6D92u, 0xFFFF926Du, 0xFFFFB649u, 0xFFFFDB24u, 0xFF00DBFFu, 0xFF00FFDBu, 0xFF24DBDBu, 0xFF49DBB6u,
    0xFF6DDB92u, 0xFF9249FFu, 0xFF926DDBu, 0xFF9292B6u, 0xFF92B692u, 0xFF92DB6Du, 0xFF92FF49u, 0xFFB649DBu,
};

std::vector<OverlayColormapEntry> buildEntries(const EntryScope scope)
{
    std::vector<OverlayColormapEntry> out;
    for (const auto& spec : buildSpecs()) {
        if (scope == EntryScope::SharedOnly && spec.audience == ColormapAudience::OverlayOnly) {
            continue;
        }
        out.push_back({spec.label, spec.id});
    }
    return out;
}

const std::vector<OverlayColormapSpec>& buildSpecs()
{
    // tint values are now R, G, B order (matching output format)
    static const std::vector<OverlayColormapSpec> specs = {
        {"fire", QStringLiteral("Fire"), OverlayColormapKind::OpenCv, ColormapAudience::Shared, cv::COLORMAP_HOT, {}, nullptr},
        {"viridis", QStringLiteral("Viridis"), OverlayColormapKind::OpenCv, ColormapAudience::Shared, cv::COLORMAP_VIRIDIS, {}, nullptr},
        {"magma", QStringLiteral("Magma"), OverlayColormapKind::OpenCv, ColormapAudience::Shared, cv::COLORMAP_MAGMA, {}, nullptr},
        {"red", QStringLiteral("Red"), OverlayColormapKind::Tint, ColormapAudience::Shared, 0, cv::Vec3f(1.0f, 0.0f, 0.0f), nullptr},
        {"green", QStringLiteral("Green"), OverlayColormapKind::Tint, ColormapAudience::Shared, 0, cv::Vec3f(0.0f, 1.0f, 0.0f), nullptr},
        {"blue", QStringLiteral("Blue"), OverlayColormapKind::Tint, ColormapAudience::Shared, 0, cv::Vec3f(0.0f, 0.0f, 1.0f), nullptr},
        {"cyan", QStringLiteral("Cyan"), OverlayColormapKind::Tint, ColormapAudience::Shared, 0, cv::Vec3f(0.0f, 1.0f, 1.0f), nullptr},
        {"magenta", QStringLiteral("Magenta"), OverlayColormapKind::Tint, ColormapAudience::Shared, 0, cv::Vec3f(1.0f, 0.0f, 1.0f), nullptr},
        {"glasbey_black0", QStringLiteral("Glasbey (0=Black)"), OverlayColormapKind::DiscreteLut, ColormapAudience::OverlayOnly, 0, {}, kGlasbeyBlack0Lut.data()},
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

    if (spec.kind == OverlayColormapKind::OpenCv) {
        // OpenCV colormaps produce BGR output. Build a 256-entry XRGB LUT
        // once, then map pixels directly — avoids applyColorMap's iterator overhead.
        cv::Mat lut_input(1, 256, CV_8UC1);
        for (int i = 0; i < 256; ++i)
            lut_input.at<uint8_t>(0, i) = static_cast<uint8_t>(i);

        cv::Mat lut_bgr;
        cv::applyColorMap(lut_input, lut_bgr, spec.opencvCode);

        // Convert BGR LUT to packed XRGB32
        std::array<uint32_t, 256> lut{};
        const auto* bgrRow = lut_bgr.ptr<cv::Vec3b>(0);
        for (int i = 0; i < 256; ++i) {
            lut[i] = 0xFF000000u
                   | (static_cast<uint32_t>(bgrRow[i][2]) << 16)
                   | (static_cast<uint32_t>(bgrRow[i][1]) << 8)
                   |  static_cast<uint32_t>(bgrRow[i][0]);
        }
        return applyPackedLut(values, lut.data());
    }

    if (spec.kind == OverlayColormapKind::DiscreteLut && spec.discreteLut != nullptr) {
        return applyPackedLut(values, spec.discreteLut);
    }

    {
        // Tint colormap: value * tint[channel]
        // tint is R, G, B in [0,1]
        const float tR = spec.tint[0] * 255.0f;
        const float tG = spec.tint[1] * 255.0f;
        const float tB = spec.tint[2] * 255.0f;

        // Build 256-entry LUT for each channel
        std::array<uint32_t, 256> lut{};
        for (int i = 0; i < 256; ++i) {
            float f = static_cast<float>(i) / 255.0f;
            auto r = static_cast<uint32_t>(std::min(f * tR, 255.0f));
            auto g = static_cast<uint32_t>(std::min(f * tG, 255.0f));
            auto b = static_cast<uint32_t>(std::min(f * tB, 255.0f));
            lut[i] = 0xFF000000u | (r << 16) | (g << 8) | b;
        }
        return applyPackedLut(values, lut.data());
    }
}

const std::vector<OverlayColormapEntry>& entries(const EntryScope scope)
{
    static const std::vector<OverlayColormapEntry> sharedEntries = buildEntries(EntryScope::SharedOnly);
    static const std::vector<OverlayColormapEntry> overlayEntries = buildEntries(EntryScope::OverlayCompatible);
    return scope == EntryScope::SharedOnly ? sharedEntries : overlayEntries;
}

} // namespace volume_viewer_cmaps
