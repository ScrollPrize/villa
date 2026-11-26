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
// Predicted normals are used whenever they exist and Lasagna was not chosen;
// a fiber dataset without them falls back to Lasagna.
inline bool usePredictedFiberNormals(
    const vc::lasagna::LasagnaDatasetManifest& manifest,
    std::string_view preference)
{
    if (preference == "lasagna") return false;
    if (preference != "fiber")
        throw std::invalid_argument("Unknown line annotation normal source");
    return hasPredictedFiberNormals(manifest);
}

} // namespace vc3d::line_annotation
