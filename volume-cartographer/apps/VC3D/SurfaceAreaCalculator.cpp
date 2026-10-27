#include "SurfaceAreaCalculator.hpp"
#include "SurfaceCatalog.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/SurfaceArea.hpp"
#include <vc_object_store/objects/GeometryCoordinates.hpp>
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
            // A surface of the user's store (perhaps just created, not listed yet).
            const auto* row = catalog.row(id);
            if (!catalog.attachment() || !catalog.volume()) {
                throw std::runtime_error("attach an object store and open a volume first");
            }
            if ((row && row->catalog) || !catalog.session()->writable(*catalog.attachment())) {
                throw std::runtime_error("the object store is read-only");
            }
            // The physical area from the open volume; area_vx2 in voxels of the
            // volume the surface is bound to, whatever volume is open.
            double viewedVx2 = 0.0;
            double areaVx2 = 0.0;
            catalog.session()->read(*catalog.attachment(), [&](const auto& db, const auto& txn, const auto&) {
                const auto object = db.objectByUuid(txn, id);
                if (!object) throw std::runtime_error("the surface is no longer in the object store");
                const auto meta = nlohmann::json::parse(object->meta);
                const auto bound = vc::objects::geometryCoordinates(
                    meta.value("coordinates", nlohmann::json::object()))["volume_uuid"];
                if (!bound.is_string()) throw std::runtime_error("the surface is not bound to a volume");
                viewedVx2 = db.surfaceAreaInVolume(txn, id, catalog.volume()->id());
                areaVx2 = db.surfaceAreaInVolume(txn, id, bound.template get<std::string>());
            });
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
