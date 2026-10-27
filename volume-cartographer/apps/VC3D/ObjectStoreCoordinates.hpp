#pragma once
#include "vc/core/types/Volume.hpp"
#include <vc_object_store/objects/GeometryCoordinates.hpp>

namespace vc3d {
inline nlohmann::json coordinateBindingForVolume(const Volume& volume, double storedUnitsPerVoxel)
{
    return {{"coordinates", vc::objects::geometryCoordinates({{"volume_uuid", volume.id()},
                                                             {"scale_factor", storedUnitsPerVoxel}})},
            {"coordinate_binding", {{"active_volume_id", volume.id()}, {"shape_zyx", volume.shape()},
                                    {"resolution_um", volume.voxelSize()}}}};
}
inline std::optional<double> geometryScaleForVolume(const nlohmann::json& metadata, const Volume& volume)
{
    return vc::objects::geometryScaleToVolume(metadata, volume.id());
}
}
