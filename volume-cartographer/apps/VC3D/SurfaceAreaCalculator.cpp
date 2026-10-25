#include "SurfaceAreaCalculator.hpp"
#include "SurfaceCatalog.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfaceArea.hpp"
#include <cmath>

std::vector<AreaResult> SurfaceAreaCalculator::calculateAreas(SurfaceCatalog& catalog, const std::vector<std::string>& ids)
{
    std::vector<AreaResult> results;
    results.reserve(ids.size());
    const double voxelSize = catalog.volume() ? catalog.volume()->voxelSize() : 0.0;
    for (const auto& id : ids) {
        AreaResult result;
        result.segmentId = id;
        try {
            const auto opened = catalog.open(id);
            if (!opened.editable) {
                throw std::runtime_error("not editable at the current volume's scale");
            }
            const double areaVx2 = vc::surface::computeSurfaceAreaVox2(*opened.surface);
            if (!std::isfinite(areaVx2)) {
                throw std::runtime_error("non-finite area");
            }
            const auto areaCm2 = vc::surface::areaCm2FromVox2(areaVx2, voxelSize);
            catalog.updateMeta(id, [&](nlohmann::json& meta) {
                meta["area_vx2"] = areaVx2;
                if (areaCm2) meta["area_cm2"] = *areaCm2;
                else meta.erase("area_cm2");
            });
            result.areaVx2 = areaVx2;
            result.areaCm2 = areaCm2.value_or(0.0);
            result.success = true;
        } catch (const std::exception& error) {
            result.errorReason = error.what();
        }
        results.push_back(result);
    }
    return results;
}
