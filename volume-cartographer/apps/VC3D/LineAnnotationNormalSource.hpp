#pragma once

#include "vc/lasagna/Manifest.hpp"
#include <stdexcept>
#include <string_view>

namespace vc3d::line_annotation {

inline bool hasPredictedFiberNormals(const vc::lasagna::LasagnaDatasetManifest& manifest)
{
    return manifest.groupForChannel("sheet_nx") && manifest.groupForChannel("sheet_ny");
}

// A normal source is independent of the interpolation/tracing backend. In
// particular, compound AFV nx/ny encode fiber DIRECTION, not the sheet normal.
inline bool usePredictedFiberNormals(
    const vc::lasagna::LasagnaDatasetManifest& manifest,
    std::string_view preference)
{
    if (preference == "lasagna") return false;
    if (preference != "fiber")
        throw std::invalid_argument("Unknown line annotation normal source");
    if (!hasPredictedFiberNormals(manifest))
        throw std::invalid_argument(
            "The selected fiber dataset has no complete predicted fiber-normal channels "
            "(sheet_nx, sheet_ny). Select a fiber dataset that has them, or choose Lasagna "
            "normals explicitly in the Tracing / annotation normals menu.");
    return true;
}

} // namespace vc3d::line_annotation
