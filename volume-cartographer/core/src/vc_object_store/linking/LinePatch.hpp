#pragma once
#include "vc/core/object_links/ObjectLinks.hpp"
#include <functional>
#include <opencv2/core.hpp>

namespace vc::object_links::detail {
struct LineData {
    std::filesystem::path path; // diagnostic label; not necessarily a file
    std::string uuid;
    std::vector<cv::Vec3d> line; // working XYZ, raw input order
    std::vector<double> arclength;
    std::vector<cv::Vec3d> controls; // fibers only
    size_t firstLine = 0, lastLine = 0; // inclusive eligible span, raw indices
    objects::ObjectRecord object;
    std::filesystem::path stripCacheFile; // empty for inputs without persistent strip caching
    bool rebuildStrip = false;
};
using LineLoader = std::function<std::optional<LineData>(size_t, DetectorReport&)>;
void validateLinePatchParams(const LinePatchParams& params, double sourceScale);
// Load one line at a time so track coordinates need not all be copied into RAM.
DetectorReport linkLinesToPatches(objects::ObjectStore& db, size_t lineCount, const LineLoader& load,
    const std::filesystem::path& patchesDir, const std::filesystem::path& ctVolume,
    const LinePatchParams& params, double sourceScale, const char* detector, int version,
    const std::atomic_bool* cancel);
} // namespace vc::object_links::detail
