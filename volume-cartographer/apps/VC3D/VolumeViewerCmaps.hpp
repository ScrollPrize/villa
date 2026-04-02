#pragma once

#include <QString>
#include <opencv2/core.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace volume_viewer_cmaps
{

enum class OverlayColormapKind { OpenCv, Tint, DiscreteLut };
enum class ColormapAudience { Shared, OverlayOnly };
enum class EntryScope { SharedOnly, OverlayCompatible };

struct OverlayColormapSpec
{
    std::string id;
    QString label;
    OverlayColormapKind kind;
    ColormapAudience audience;
    int opencvCode;
    cv::Vec3f tint; // R, G, B in [0,1]
    const uint32_t* discreteLut;
};

struct OverlayColormapEntry
{
    QString label;
    std::string id;
};

const std::vector<OverlayColormapSpec>& specs();
const OverlayColormapSpec& resolve(const std::string& id);

// Apply colormap and write directly into a caller-provided ARGB32 buffer.
// outBuf must point to rows*outStride uint32_t elements.
// outStride is in uint32_t units (pixels per row including padding).
void makeColors(const cv::Mat_<uint8_t>& values, const OverlayColormapSpec& spec,
                uint32_t* outBuf, int outStride);

const std::vector<OverlayColormapEntry>& entries(EntryScope scope = EntryScope::OverlayCompatible);

} // namespace volume_viewer_cmaps
