#pragma once
#include "vc_object_store/objects/GeometryCoordinates.hpp"

namespace vc::objects::detail {
inline std::string coordinateFilterValue(const std::string& json)
{
    return geometryCoordinates(CoordinateJson::parse(json)).dump();
}
inline std::string coordinateFilterSql(const std::string& alias, const std::string& parameter)
{
    return " AND json_extract(" + alias + ".meta,'$.coordinates.volume_uuid') IS json_extract(" +
        parameter + ",'$.volume_uuid') AND json_extract(" + alias +
        ".meta,'$.coordinates.scale_factor') IS json_extract(" + parameter + ",'$.scale_factor')";
}
}
