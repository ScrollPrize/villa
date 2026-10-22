#pragma once
#include "vc/core/types/Volume.hpp"
#include <vc_object_store/objects/GeometryCoordinates.hpp>
#include <QUuid>

namespace vc3d {
inline std::string objectStoreVolumeUuid(const Volume& volume)
{
    try { return vc::objects::util::canonicalUuid(volume.id()); }
    catch (const std::invalid_argument&) {
        // Stable identity for native volume IDs that are not UUIDs.
        return QUuid::createUuidV5(QUuid("{6ba7b811-9dad-11d1-80b4-00c04fd430c8}"),
            QByteArray::fromStdString(volume.id())).toString(QUuid::WithoutBraces).toStdString();
    }
}
inline nlohmann::json coordinateBindingForVolume(const Volume& volume, double storedUnitsPerVoxel)
{
    return {{"coordinates", vc::objects::geometryCoordinates({{"volume_uuid", objectStoreVolumeUuid(volume)},
                                                             {"scale_factor", storedUnitsPerVoxel}})},
            {"coordinate_binding", {{"active_volume_id", volume.id()}, {"shape_zyx", volume.shape()},
                                    {"resolution_um", volume.voxelSize()}}}};
}
inline std::optional<double> geometryScaleForVolume(const nlohmann::json& metadata, const Volume& volume)
{
    return vc::objects::geometryScaleToVolume(metadata, objectStoreVolumeUuid(volume));
}
}
