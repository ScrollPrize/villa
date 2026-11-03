#pragma once
#include <string>
namespace vc::objects::detail {
inline std::string sequenceIdArray(const std::string& kind, const std::string& name)
{
    if (kind == "point_collection" && name == "points") return "point_ids";
    if (kind == "fiber" && name == "control_points") return "control_point_ids";
    return {};
}
}
