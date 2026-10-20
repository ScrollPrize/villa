#pragma once
#include "vc_object_store/geometry/Geometry.hpp"
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <nlohmann/json.hpp>

namespace vc::objects::detail {
// Decode native fiber point arrays without changing point order.
inline std::vector<std::array<double, 3>> fiberPoints(const nlohmann::json& root, const char* key)
{
    std::vector<std::array<double, 3>> points;
    if (!root.contains(key)) return points;
    const auto& entries = root.at(key);
    if (!entries.is_array()) throw std::runtime_error(std::string("invalid array: ") + key);
    for (const auto& entry : entries) {
        const auto& xyz = entry.is_object() ? entry.at("position") : entry;
        if (!xyz.is_array() || xyz.size() < 3)
            throw std::runtime_error(std::string("invalid point in ") + key);
        std::array<double, 3> point;
        for (size_t axis = 0; axis < 3; ++axis) {
            point[axis] = xyz.at(axis).get<double>();
            if (!std::isfinite(point[axis])) throw std::runtime_error(std::string("non-finite point in ") + key);
        }
        points.push_back(point);
    }
    return points;
}

inline double fiberPolylineLength(const std::vector<std::array<double, 3>>& points)
{
    return vc::geometry::polylineLength(points);
}
} // namespace vc::objects::detail
