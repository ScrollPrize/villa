#include "SpiralProjectInputs.hpp"

#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"

#include <vc_object_store/objects/VolumeUmbilicus.hpp>

#include <QJsonArray>

namespace vc3d::spiral {

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
                         vc::objects::ObjectStoreSession* session,
                         const std::optional<vc::objects::AttachmentIdentity>& umbilicusStore)
{
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
    return {{QStringLiteral("name"), QString::fromStdString(name)},
            {QStringLiteral("volumes"), entries}};
}

} // namespace vc3d::spiral
