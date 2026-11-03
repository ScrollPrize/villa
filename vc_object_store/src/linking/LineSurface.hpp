#pragma once
#include "linking/StoredGeometry.hpp"
#include "vc_object_store/object_links/ObjectLinks.hpp"
#include <functional>
#include <opencv2/core.hpp>

namespace vc::object_links::detail {
struct LineData {
    std::string uuid;
    double sourceScale = 1.0;
    std::vector<cv::Vec3d> line; // working XYZ, raw input order
    std::vector<double> arclength;
    std::vector<cv::Vec3d> controls; // fibers only
    size_t firstLine = 0, lastLine = 0; // inclusive eligible span, raw indices
    objects::ObjectRecord object;
    std::filesystem::path stripCacheFile; // empty for inputs without persistent strip caching
    bool rebuildStrip = false;
};
using LineLoader = std::function<std::optional<LineData>(size_t, DetectorReport&)>;
// An incremental run (ObjectLinks.hpp): the caller loads the new lines and
// those near a new surface, and names the surfaces to load.
struct LineIncremental {
    std::vector<objects::ObjectRecord> surfaces;  // new surfaces and those near a new line
    std::set<std::string> fresh;                  // new lines and surfaces
    std::vector<std::string> selected;            // every selected line and surface
};
void validateLineSurfaceParams(const LineSurfaceParams& params, double sourceScale);
// Load one line at a time so track coordinates need not all be copied into RAM.
DetectorReport linkLinesToSurfaces(objects::ObjectStore& db, StoredGeometry& inputs, size_t lineCount, const LineLoader& load,
    const std::filesystem::path& ctVolume,
    const LineSurfaceParams& params, double sourceScale, const char* detector, int version,
    const std::atomic_bool* cancel, DetectorReport report, const LineIncremental* incremental = nullptr);
} // namespace vc::object_links::detail
