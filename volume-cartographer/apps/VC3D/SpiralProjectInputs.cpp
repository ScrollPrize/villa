#include "SpiralProjectInputs.hpp"

#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/lasagna/Manifest.hpp"

#include <vc_object_store/objects/VolumeUmbilicus.hpp>

#include <QJsonArray>

#include <cmath>

namespace vc3d::spiral {

ProjectLasagna projectLasagna(const vc::lasagna::LasagnaDatasetManifest& manifest,
                              const std::string& location)
{
    ProjectLasagna result;
    result.location = location;
    result.baseShapeZYX = manifest.baseShapeZYX;
    const std::pair<const char*, const char*> inputs[] = {
        {"normal_x", "nx"}, {"normal_y", "ny"}, {"gradient_magnitude", "grad_mag"}};
    std::optional<int> scaledown;
    for (const auto& [input, channel] : inputs) {
        const auto* group = manifest.groupForChannel(channel);
        if (!group) continue;
        if (group->isRemote()) {
            result.error = "Spiral reads Lasagna stores from local paths; " + location + " is remote";
            return result;
        }
        if (group->channels.size() != 1) {
            result.error = "Spiral reads single-channel Lasagna arrays; " + location + "'s channel " +
                           channel + " shares group " + group->name;
            return result;
        }
        // The group is an array of an OME-Zarr store: the fit reads
        // <store>/<array>.
        const auto store = group->zarrPath.parent_path().string();
        const auto array = group->zarrPath.filename().string();
        if ((!result.group.empty() && array != result.group) || (scaledown && *scaledown != group->scaledown)) {
            result.error = "Spiral reads the Lasagna channels at one array; " + location +
                           " stores them at different ones";
            return result;
        }
        result.stores[input] = store;
        result.group = array;
        scaledown = group->scaledown;
    }
    if (result.stores.empty())
        result.error = location + " has no nx, ny or grad_mag channel";
    else
        result.baseVoxelsPerArrayVoxel = manifest.sourceToBase * std::ldexp(1.0, *scaledown);
    return result;
}

std::optional<ProjectLasagna> projectLasagna(const VolumePkg& pkg)
{
    const std::string location = pkg.selectedLasagnaDataset();
    if (location.empty()) return std::nullopt;
    if (vc::project::isLocationRemote(location)) {
        ProjectLasagna remote;
        remote.location = location;
        remote.error = "Spiral reads Lasagna stores from local paths; " + location + " is remote";
        return remote;
    }
    const auto path = vc::project::resolveLocalPath(location, pkg.directory());
    try {
        return projectLasagna(vc::lasagna::LasagnaDatasetManifest::parseFile(path), path.string());
    } catch (const std::exception& error) {
        ProjectLasagna unreadable;
        unreadable.location = path.string();
        unreadable.error = "cannot read " + path.string() + ": " + error.what();
        return unreadable;
    }
}

std::map<std::string, ProjectVolumeFacts> projectVolumeFacts(const VolumePkg& pkg)
{
    std::map<std::string, ProjectVolumeFacts> result;
    for (const auto& id : pkg.volumeIDs()) {
        const auto volume = pkg.volume(id);
        if (!volume) continue;
        const auto annotation = vc::project::annotationVolume(&pkg, *volume);
        ProjectVolumeFacts facts;
        facts.voxelSizeUm = annotation.voxelSizeUm;
        if (const auto view = pkg.catalogView(id)) {
            const auto* sample = pkg.catalog()->findSample(view->sampleId);
            if (const auto* catalogVolume = sample ? sample->findVolume(view->volumeId) : nullptr) {
                facts.baseShapeZYX = catalogVolume->shapeZYX;
                facts.zTopToBottom = catalogVolume->orientation.zTopToBottom;
                facts.leftHandedCoordinates = catalogVolume->orientation.leftHandedCoordinates;
            }
        } else if (annotation.scale == 1) {
            const auto shape = volume->shape();
            facts.baseShapeZYX = std::array<std::size_t, 3>{
                std::size_t(shape[0]), std::size_t(shape[1]), std::size_t(shape[2])};
        }
        // Every working level of one catalog volume states the same facts;
        // the first that knows one gives it.
        auto& known = result[annotation.volume];
        if (!known.voxelSizeUm) known.voxelSizeUm = facts.voxelSizeUm;
        if (!known.baseShapeZYX) known.baseShapeZYX = facts.baseShapeZYX;
        if (!known.zTopToBottom) known.zTopToBottom = facts.zTopToBottom;
        if (!known.leftHandedCoordinates) known.leftHandedCoordinates = facts.leftHandedCoordinates;
    }
    return result;
}

QJsonObject projectBlock(const std::string& name,
                         const std::map<std::string, ProjectVolumeFacts>& volumes,
                         const std::optional<ProjectLasagna>& lasagna,
                         vc::objects::ObjectStoreSession* session,
                         const std::optional<vc::objects::AttachmentIdentity>& umbilicusStore)
{
    const bool readable = lasagna && lasagna->error.empty();
    QJsonObject entries;
    for (const auto& [uuid, facts] : volumes) {
        QJsonObject entry;
        if (facts.voxelSizeUm) entry[QStringLiteral("voxel_size_um")] = *facts.voxelSizeUm;
        if (facts.baseShapeZYX)
            entry[QStringLiteral("base_shape_zyx")] = QJsonArray{
                qint64((*facts.baseShapeZYX)[0]), qint64((*facts.baseShapeZYX)[1]),
                qint64((*facts.baseShapeZYX)[2])};
        if (facts.zTopToBottom) entry[QStringLiteral("z_direction_is_top_to_bottom")] = *facts.zTopToBottom;
        if (facts.leftHandedCoordinates)
            entry[QStringLiteral("left_handed_coordinates")] = *facts.leftHandedCoordinates;
        if (readable && lasagna->baseShapeZYX && facts.baseShapeZYX) {
            // An array voxel in this volume's voxels: a whole number, or the
            // volume is not a dyadic level of the dataset's base.
            try {
                const double basePerVolumeVoxel = vc::lasagna::dyadicCoordinateScaleBetweenShapes(
                    *facts.baseShapeZYX, *lasagna->baseShapeZYX);
                const double scale = lasagna->baseVoxelsPerArrayVoxel / basePerVolumeVoxel;
                if (scale >= 1 && scale == std::round(scale))
                    entry[QStringLiteral("lasagna_scale")] = qint64(scale);
            } catch (const std::exception&) {
            }
        }
        if (session && umbilicusStore) {
            // In the volume's own voxels, as the fit reads it.
            if (const auto umbilicus = vc::objects::volumeUmbilicus(
                    *session, *umbilicusStore, vc::objects::VolumeVoxels{uuid})) {
                QJsonArray points;
                for (const auto& [x, y, z] : umbilicus->points) points.append(QJsonArray{z, y, x});
                entry[QStringLiteral("umbilicus")] = QJsonObject{
                    {QStringLiteral("uuid"), QString::fromStdString(umbilicus->uuid)},
                    {QStringLiteral("revision"), QString::fromStdString(umbilicus->objectRevision)},
                    {QStringLiteral("points_zyx"), points}};
            }
        }
        entries[QString::fromStdString(uuid)] = entry;
    }
    QJsonObject block{{QStringLiteral("name"), QString::fromStdString(name)},
                      {QStringLiteral("volumes"), entries}};
    if (lasagna) {
        QJsonObject value{{QStringLiteral("location"), QString::fromStdString(lasagna->location)}};
        if (!readable) {
            value[QStringLiteral("error")] = QString::fromStdString(lasagna->error);
        } else {
            value[QStringLiteral("group")] = QString::fromStdString(lasagna->group);
            for (const auto& [input, store] : lasagna->stores)
                value[QString::fromStdString(input)] = QString::fromStdString(store);
        }
        block[QStringLiteral("lasagna")] = value;
    }
    return block;
}

} // namespace vc3d::spiral
