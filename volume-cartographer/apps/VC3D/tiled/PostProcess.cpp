#include "PostProcess.hpp"
#include "VolumeViewerCmaps.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdint>

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

    // Steps 1-4: core grayscale pipeline (modifies gray in-place)
    vc::applyPostProcess(gray, params.toCoreParams());

    // Step 5: Colormap or grayscale → QImage::Format_RGB32
    if (spec) {
        return volume_viewer_cmaps::makeColors(gray, *spec);
    }

    // Grayscale: write 0xffGGGGGG directly into RGB32 buffer.
    // This replaces: cvtColor(GRAY2BGR) + cvtColor(BGR2RGB) + QImage(RGB888) + RGB888→RGB32
    const int rows = gray.rows;
    const int cols = gray.cols;
    QImage result(cols, rows, QImage::Format_RGB32);

    auto* bits = reinterpret_cast<uint32_t*>(result.bits());
    const int stride = result.bytesPerLine() / 4;
    for (int y = 0; y < rows; ++y) {
        const auto* src = gray.ptr<uint8_t>(y);
        auto* dst = bits + y * stride;
        for (int x = 0; x < cols; ++x) {
            uint32_t v = src[x];
            dst[x] = 0xFF000000u | (v << 16) | (v << 8) | v;
        }
    }
    return result;
}
