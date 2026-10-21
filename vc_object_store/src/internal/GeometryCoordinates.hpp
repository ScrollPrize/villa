#pragma once

#include "vc_object_store/Uuid.hpp"
#include <nlohmann/json.hpp>
#include <cmath>

namespace vc::objects::detail {
using CoordinateJson = nlohmann::json;
inline bool hasGeometryCoordinates(const std::string& kind)
{
    return kind == "point_collection" || kind == "patch" || kind == "fiber" || kind == "track";
}
// The only persistent coordinate declarations are volume UUID and the factor
// mapping source-volume voxels to the geometry's annotation coordinates.
inline CoordinateJson geometryCoordinates(CoordinateJson value)
{
    if (!value.is_object())
        throw std::invalid_argument("geometry coordinates must be an object");
    for (const auto& [key, field] : value.items())
        if (key != "volume_uuid" && key != "scale_factor")
            throw std::invalid_argument("unknown geometry coordinate field: " + key);
    if (!value.contains("volume_uuid")) value["volume_uuid"] = nullptr;
    if (!value.contains("scale_factor")) value["scale_factor"] = nullptr;
    if (!value["volume_uuid"].is_null()) {
        if (!value["volume_uuid"].is_string())
            throw std::invalid_argument("geometry volume_uuid must be a UUID string or null");
        value["volume_uuid"] = util::canonicalUuid(value["volume_uuid"].get<std::string>());
    }
    const auto& scale = value["scale_factor"];
    if (!scale.is_null() &&
        (!scale.is_number() || !std::isfinite(scale.get<double>()) || scale.get<double>() <= 0))
        throw std::invalid_argument("geometry scale_factor must be finite and positive or null");
    return value;
}

inline CoordinateJson mergeGeometryCoordinates(const CoordinateJson& a, const CoordinateJson& b)
{
    auto result = geometryCoordinates(a);
    const auto incoming = geometryCoordinates(b);
    for (const auto* key : {"volume_uuid", "scale_factor"}) {
        if (incoming[key].is_null()) continue;
        if (result[key].is_null()) {
            result[key] = incoming[key];
            continue;
        }
        bool equal = result[key] == incoming[key];
        // JSON's ordinary mixed float/integer comparison can round a large
        // integer. Do not accept conflicting exact scale declarations that way.
        const auto& left = result[key];
        const auto& right = incoming[key];
        if (left.is_number() && right.is_number() &&
            left.is_number_float() != right.is_number_float()) {
            const auto& integer = left.is_number_integer() ? left : right;
            const double floating = (left.is_number_float() ? left : right).get<double>();
            equal = floating < std::ldexp(1.0, 64) && std::floor(floating) == floating &&
                    static_cast<uint64_t>(floating) == integer.get<uint64_t>();
        }
        if (!equal) throw std::invalid_argument(std::string("geometry coordinate conflict: ") + key);
    }
    return result;
}

inline CoordinateJson sourceGeometryCoordinates(const CoordinateJson& source)
{
    if (!source.is_object())
        throw std::invalid_argument("geometry coordinate source must be an object");
    auto result = geometryCoordinates(source.value("coordinates", CoordinateJson::object()));
    CoordinateJson flat = CoordinateJson::object();
    for (const auto* key : {"volume_uuid", "scale_factor"})
        if (source.contains(key)) flat[key] = source[key];
    return mergeGeometryCoordinates(result, flat);
}

inline void eraseSourceGeometryCoordinates(CoordinateJson& source)
{
    for (const auto* key : {"coordinates", "volume_uuid", "scale_factor"}) source.erase(key);
}
} // namespace vc::objects::detail
