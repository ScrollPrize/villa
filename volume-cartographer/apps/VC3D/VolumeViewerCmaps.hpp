#pragma once

#include <QString>
#include <opencv2/core.hpp>

#include <cstdint>
#include <string>
#include <vector>

#include "vc/core/render/Colormaps.hpp"

namespace volume_viewer_cmaps
{

// Re-export core types under the app-level namespace for backward compatibility.
using vc::OverlayColormapKind;
using vc::ColormapAudience;
using vc::EntryScope;
using vc::OverlayColormapSpec;

// Qt-specific entry with QString label for UI comboboxes.
struct OverlayColormapEntry
{
    QString label;
    std::string id;
};

// Re-export core functions.
inline const std::vector<OverlayColormapSpec>& specs() { return vc::specs(); }
inline const OverlayColormapSpec& resolve(const std::string& id) { return vc::resolve(id); }
inline void makeColors(const cv::Mat_<uint8_t>& values, const OverlayColormapSpec& spec,
                       uint32_t* outBuf, int outStride) {
    vc::makeColors(values, spec, outBuf, outStride);
}

// Build Qt-specific entry lists with QString labels for UI comboboxes.
const std::vector<OverlayColormapEntry>& entries(EntryScope scope = EntryScope::OverlayCompatible);

} // namespace volume_viewer_cmaps
