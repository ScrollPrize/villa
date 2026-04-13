#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

class VolumePkg;
class Volume;
class SurfaceTreeWidget;

struct AreaResult {
    std::string segmentId;
    double areaVx2{0.0};
    double areaCm2{0.0};
    bool success{false};
    std::string errorReason;
};

namespace SurfaceAreaCalculator {

// Calculate area for multiple segments. Pure computation, no UI.
std::vector<AreaResult> calculateAreas(
    const std::shared_ptr<VolumePkg>& vpkg,
    const std::shared_ptr<Volume>& volume,
    const std::vector<std::string>& ids);

} // namespace SurfaceAreaCalculator
