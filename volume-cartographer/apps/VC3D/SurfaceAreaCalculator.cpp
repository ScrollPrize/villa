#include "SurfaceAreaCalculator.hpp"
#include "SurfaceCatalog.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfaceArea.hpp"
#include <vc_object_store/objects/GeometryCoordinates.hpp>
#include <vc_object_store/store/GeometryViews.hpp>
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
                throw std::runtime_error("the object store is read-only");
            }
            // The physical area from the open volume; area_vx2 in voxels of the
            // volume the surface is bound to, whatever volume is open.
            const double viewedVx2 = vc::surface::computeSurfaceAreaVox2(*opened.surface);
            double areaVx2 = viewedVx2;
            if (opened.view.mapping != vc::objects::VolumeAffine::identity()) {
                // The same revision read in the bound volume's voxels, through the store's view of it.
                catalog.session()->read(opened.pin.attachment, [&](const auto& db, const auto& txn, const auto&) {
                    const auto meta = nlohmann::json::parse(db.objectByUuid(txn, id)->meta);
                    const auto bound = vc::objects::geometryCoordinates(
                        meta.value("coordinates", nlohmann::json::object()))["volume_uuid"];
                    if (!bound.is_string()) throw std::runtime_error("the surface is not bound to a volume");
                    const auto view = vc::objects::GeometrySnapshot(db, txn)
                                          .surface(id, opened.pin.geometryRevision)
                                          .viewTransformedAt(bound.template get<std::string>());
                    const auto stored = load_quad_from_store(db, txn, id, opened.pin.geometryRevision, true, &view);
                    areaVx2 = vc::surface::computeSurfaceAreaVox2(*stored);
                });
            }
            if (!std::isfinite(areaVx2) || !std::isfinite(viewedVx2)) {
                throw std::runtime_error("non-finite area");
            }
            const auto areaCm2 = vc::surface::areaCm2FromVox2(viewedVx2, voxelSize);
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
