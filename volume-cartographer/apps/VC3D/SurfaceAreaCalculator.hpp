#pragma once

#include <string>
#include <vector>

class SurfaceCatalog;

struct AreaResult {
    std::string segmentId;
    double areaVx2{0.0};
    double areaCm2{0.0};
    bool success{false};
    std::string errorReason;
};

namespace SurfaceAreaCalculator {

// Area of each surface's valid quads in the current volume's voxels, recorded
// in its record meta (area_vx2, and area_cm2 when the voxel size is known).
std::vector<AreaResult> calculateAreas(SurfaceCatalog& catalog, const std::vector<std::string>& ids);

} // namespace SurfaceAreaCalculator
