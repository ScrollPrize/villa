#include "ProjectResources.hpp"

#include <cmath>

#include "CState.hpp"

#include "vc/core/types/Volume.hpp"

#include <vc_object_store/objects/StoreReader.hpp>
#include <vc_object_store/objects/VolumeUmbilicus.hpp>

namespace vc3d {
namespace {
// The attached volume an umbilicus stamp names by store directory name, so the
// stamp can be checked against it. Remote volumes are skipped: the check needs
// local shape metadata.
std::shared_ptr<Volume> attachedVolumeByStoreName(const VolumePkg& pkg, const std::string& storeName)
{
    const std::string wanted = std::filesystem::path(storeName).filename().string();
    if (wanted.empty()) return nullptr;
    for (const std::string& id : pkg.volumeIDs()) {
        const std::shared_ptr<Volume> volume = pkg.volume(id);
        if (volume && !volume->isRemote() && volume->path().filename().string() == wanted) return volume;
    }
    return nullptr;
}

vc::objects::UmbilicusTargetGrid grid(const Volume& v)
{
    const double voxelSize = v.voxelSize();
    return vc::objects::UmbilicusTargetGrid{
        {double(v.sliceWidth()), double(v.sliceHeight()), double(v.numSlices())},
        std::isfinite(voxelSize) && voxelSize > 0 ? std::optional(voxelSize) : std::nullopt};
}
}

UmbilicusImportPlan planUmbilicusImport(const CState& state, const std::filesystem::path& file)
{
    const auto attachment = state.storeFor("umbilicus");
    const auto volume = state.currentVolume();
    if (!attachment || !volume || !state.vpkg() || !state.annotationVolume())
        throw std::runtime_error("Open a project and a volume first.");
    UmbilicusImportPlan plan;
    plan.file = file;
    try {
        plan.umbilicus = vc::objects::readUmbilicusFile(file);
    } catch (const std::exception& error) {
        throw UmbilicusFileUnreadable(error.what());
    }
    // The store decides the frame: the file's stamp against the volume the
    // umbilicus is bound to, and against an attached volume the stamp names.
    std::optional<vc::objects::UmbilicusTargetGrid> named;
    if (plan.umbilicus.volume)
        if (const auto volumeNamed = attachedVolumeByStoreName(*state.vpkg(), *plan.umbilicus.volume)) named = grid(*volumeNamed);
    plan.binding = vc::objects::umbilicusBinding(plan.umbilicus, grid(*volume), named);
    plan.store = *attachment;
    plan.target = *state.annotationVolume();
    state.objectStoreSession()->readStore(plan.store, [&](const vc::objects::StoreReader& reader, const auto&) {
        plan.replaces = !vc::objects::volumeUmbilici(reader, plan.target.volume).empty();
    });
    return plan;
}

std::string importUmbilicus(const CState& state, const UmbilicusImportPlan& plan, double storedUnitsPerVoxel)
{
    // The umbilicus belongs to the open volume's store volume: units per voxel
    // of the open volume are units per voxel of that one at its scale.
    return vc::objects::importVolumeUmbilicus(*state.objectStoreSession(), plan.store, plan.target.volume, plan.umbilicus,
        storedUnitsPerVoxel * plan.target.scale, plan.file.stem().string(),
        {{"tool", "vc3d-import-umbilicus"},
         {"parameters", {{"file", plan.file.string()}, {"units_per_voxel", storedUnitsPerVoxel},
                         {"frame", plan.binding.frame.empty() ? std::string("entered") : plan.binding.frame}}},
         {"sources", nlohmann::json::array()}});
}

std::optional<AnnotationUmbilici> annotationUmbilici(const CState& state)
{
    const auto attachment = state.storeFor("umbilicus");
    if (!attachment || !state.annotationVolume()) return std::nullopt;
    AnnotationUmbilici result{*attachment, state.annotationVolume()->volume, {}};
    state.objectStoreSession()->readStore(result.store, [&](const vc::objects::StoreReader& reader, const auto&) {
        for (const auto& record : vc::objects::volumeUmbilici(reader, result.volume)) result.uuids.push_back(record.uuid);
    });
    return result;
}

std::vector<std::string> retireUmbilici(const CState& state, const AnnotationUmbilici& umbilici)
{
    (void)state.objectStoreSession()->editCurrent(umbilici.store, umbilici.uuids,
        [](const vc::objects::StoreReader&, const std::string& uuid) {
            return std::vector<vc::objects::TransactionOperation>{vc::objects::RetireObjectOp{uuid}};
        });
    return umbilici.uuids;
}

}  // namespace vc3d
