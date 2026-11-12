#pragma once

#include "vc/lasagna/Manifest.hpp"
#include <stdexcept>
#include <string_view>

namespace vc3d::line_annotation {

// A normal source is independent of the interpolation/tracing backend. In
// particular, compound AFV nx/ny encode fiber DIRECTION, not the sheet normal.
inline bool usePredictedFiberNormals(
    const vc::lasagna::LasagnaDatasetManifest& manifest,
    std::string_view preference)
{
    if (preference == "lasagna") return false;
    if (preference != "fiber")
        throw std::invalid_argument("Unknown line annotation normal source");
    const bool available = manifest.groupForChannel("sheet_nx") &&
        manifest.groupForChannel("sheet_ny") &&
        manifest.groupForChannel("sheet_grad_mag");
    if (!available)
        throw std::invalid_argument(
            "The selected fiber dataset has no complete predicted fiber-normal channels "
            "(sheet_nx, sheet_ny, sheet_grad_mag). Choose Lasagna normals.");
    return available;
}

} // namespace vc3d::line_annotation
