#include "PostProcess.hpp"
#include "VolumeViewerCmaps.hpp"

#include <opencv2/imgproc.hpp>
#include <utils/slab_allocator.hpp>

#include <algorithm>
#include <array>
#include <cstdint>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

// Non-temporal store for write-only ARGB32 output — bypasses cache,
// freeing cache lines for read-heavy LUT/chunk data.
static inline void nt_store_u32(uint32_t* dst, uint32_t val) {
#if defined(__x86_64__)
    _mm_stream_si32(reinterpret_cast<int*>(dst), static_cast<int>(val));
#elif defined(__aarch64__) && __has_builtin(__builtin_nontemporal_store)
    __builtin_nontemporal_store(val, dst);
#else
    *dst = val;
#endif
}

// Global lock-free slab pool for 512x512 ARGB32 tile buffers (1 MB each).
QImage allocTileImage(int cols, int rows)
{
    return QImage(cols, rows, QImage::Format_RGB32);
}

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

// Check whether core preprocessing steps 1-3 are all no-ops.
static bool corePreprocessIsNoop(const PostProcessParams& params)
{
    return params.isoCutoff == 0
        && !params.postStretchValues
        && !params.removeSmallComponents;
}

// Apply the fused LUT to convert grayscale pixels to ARGB32.
// Handles contiguous and row-strided layouts, processes 4 pixels at a time.
static void applyFusedLut(const cv::Mat_<uint8_t>& gray,
                          const std::array<uint32_t, 256>& lut,
                          uint32_t* bits, int stride)
{
    const int rows = gray.rows;
    const int cols = gray.cols;

    for (int y = 0; y < rows; ++y) {
        const auto* __restrict__ src = gray.ptr<uint8_t>(y);
        auto* __restrict__ dst = bits + y * stride;

        int x = 0;
        // Process 4 pixels per iteration — scalar gather, batch store.
        // The LUT is 1KB and stays hot in L1. Unrolling 4x cuts loop
        // overhead and lets the CPU pipeline overlapping loads/stores.
        const int cols4 = cols - 3;
        for (; x < cols4; x += 4) {
            nt_store_u32(&dst[x + 0], lut[src[x + 0]]);
            nt_store_u32(&dst[x + 1], lut[src[x + 1]]);
            nt_store_u32(&dst[x + 2], lut[src[x + 2]]);
            nt_store_u32(&dst[x + 3], lut[src[x + 3]]);
        }
        for (; x < cols; ++x) {
            nt_store_u32(&dst[x], lut[src[x]]);
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
    // Skip core preprocessing entirely when all steps are no-ops (common case
    // for volume tiles: isoCutoff=0, no postStretch, no component removal).
    if (!corePreprocessIsNoop(params)) {
        auto coreParams = params.toCoreParams();
        coreParams.windowLow = 0.0f;
        coreParams.windowHigh = 255.0f;
        coreParams.stretchValues = false;
        vc::applyPostProcess(gray, coreParams);
    }

    // Build fused LUT: window/level (or stretch) → ARGB32
    std::array<uint32_t, 256> lut;
    buildFusedGrayLut(lut, params, gray);

    // Single-pass: gray → RGB32 via fused LUT (4x unrolled)
    const int rows = gray.rows;
    const int cols = gray.cols;
    QImage result = allocTileImage(cols, rows);

    auto* bits = reinterpret_cast<uint32_t*>(result.bits());
    const int stride = result.bytesPerLine() / 4;
    applyFusedLut(gray, lut, bits, stride);

    return result;
}

void buildWindowLevelLut(std::array<uint32_t, 256>& lut,
                         float windowLow, float windowHigh,
                         float lightFactor)
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
